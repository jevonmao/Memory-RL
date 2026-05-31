"""Random Network Distillation (Burda et al., 2018).

Two MLPs of identical architecture:
  target    : φ → R^d   — frozen, random initialisation (never trained)
  predictor : φ → R^d   — trained to match target(φ) on visited states

Intrinsic reward at step t:
    r_i(s_t) = ‖ predictor(φ(s_t)) − target(φ(s_t)) ‖²

Key properties (vs ICM):
  * No dependence on action, so noisy-TV problem doesn't arise.
  * Predictor only fits *visited* states ⇒ never saturates on *unseen*
    states. Pathak ICM's intrinsic decayed 0.49 → 0.016 in 250 k steps
    on BinFill — that decay can't happen on unseen states under RND.
  * Frozen random target ⇒ very cheap to evaluate (one forward).

We run RND on top of the *shared* RobommeCNNExtractor features (576-d), so
the random target operates in a meaningful visual space rather than on raw
pixels. We do NOT propagate predictor gradients into the extractor — RND's
encoder is *random*, and forcing the policy's encoder to fit a random target
would just regularise the policy toward noise.

Numerical guards:
  * Feature normalisation: running mean/std of the input φ before both target
    and predictor. Without this the inputs' scale matters → predictor either
    diverges (large features) or learns the constant target (small features).
  * Intrinsic-reward normalisation: running std of the *intrinsic returns*,
    then divide intrinsic by that std. This is the form used in the
    Burda paper (and Stable-Baselines3-contrib's RND impl, where it exists).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class _RunningMeanStd(nn.Module):
    """Online mean / variance tracker.

    Stored as buffers so it moves with .to(device) and is included in
    state_dict(). Update is in-place on the buffer tensors (no gradients).
    Algorithm: Welford / parallel-Welford from the Burda RND code.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("mean",  torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var",   torch.ones (shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.detach().to(self.mean.dtype)
        batch_mean = x.mean(dim=0)
        batch_var  = x.var (dim=0, unbiased=False)
        batch_n    = torch.tensor(float(x.shape[0]), dtype=self.mean.dtype, device=self.mean.device)

        delta   = batch_mean - self.mean
        tot_n   = self.count + batch_n
        new_mean = self.mean + delta * (batch_n / tot_n)

        m_a = self.var * self.count
        m_b = batch_var * batch_n
        M2  = m_a + m_b + (delta ** 2) * (self.count * batch_n / tot_n)
        new_var = M2 / tot_n

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_n)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var.clamp_min(1e-8)).to(torch.float32)


# ---------------------------------------------------------------------------
class _RNDHead(nn.Module):
    """4-layer MLP from feat_dim → out_dim used for both target and predictor."""

    def __init__(self, feat_dim: int, out_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
            nn.Linear(hidden,   out_dim),
        )
        # Burda et al.'s orthogonal init with √2 gain
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
class RND(nn.Module):
    """Owns target + predictor + running stats.

    Public surface used by PPOWithRND:
      .intrinsic(feat)       -> (B,)   normalised intrinsic reward
      .loss(feat)            -> scalar predictor MSE loss
      .update_obs_rms(feat)  -> in-place; call once per env step batch
      .update_ret_rms(returns) -> in-place; call once per rollout
      .parameters()          -> predictor params only (target frozen)
    """

    def __init__(self, feat_dim: int = 576, out_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim  = out_dim

        self.target    = _RNDHead(feat_dim, out_dim, hidden)
        self.predictor = _RNDHead(feat_dim, out_dim, hidden)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

        self.obs_rms = _RunningMeanStd(shape=(feat_dim,))
        # Discounted intrinsic-return std (Burda et al. §2.5).
        self.ret_rms = _RunningMeanStd(shape=())

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        self.target.eval()         # frozen forever
        return self

    # ------------------------------------------------------------------
    def _normalise(self, feat: torch.Tensor) -> torch.Tensor:
        m  = self.obs_rms.mean.to(feat.device, feat.dtype)
        s  = self.obs_rms.std.to(feat.device,  feat.dtype)
        # Clip to ±5σ — matches Burda et al.; protects against feature outliers
        # right after a fresh RMS init when std is ≈1.
        return ((feat - m) / s).clamp(-5.0, 5.0)

    # ------------------------------------------------------------------
    def intrinsic(self, feat: torch.Tensor) -> torch.Tensor:
        """Per-sample intrinsic reward (B,) — no_grad caller is expected."""
        with torch.no_grad():
            z = self._normalise(feat)
            t = self.target(z)
            p = self.predictor(z)
            r = (p - t).pow(2).mean(dim=-1)
            # Scale by running std of intrinsic returns (skip if uninitialised).
            std = self.ret_rms.std.to(feat.device, feat.dtype)
            r = r / std.clamp_min(1e-8)
        return r

    def loss(self, feat: torch.Tensor) -> torch.Tensor:
        """Predictor regression loss against the (no-grad) target output."""
        z = self._normalise(feat)
        with torch.no_grad():
            t = self.target(z)
        p = self.predictor(z)
        return F.mse_loss(p, t)

    # ------------------------------------------------------------------
    def update_obs_rms(self, feat: torch.Tensor) -> None:
        self.obs_rms.update(feat)

    def update_ret_rms(self, returns: torch.Tensor) -> None:
        # Burda passes the per-timestep discounted intrinsic-return flat tensor.
        self.ret_rms.update(returns.reshape(-1, 1).squeeze(-1).unsqueeze(-1).squeeze(-1)
                            if returns.dim() == 0 else returns.reshape(-1))

    # ------------------------------------------------------------------
    def parameters(self, recurse: bool = True):
        # Only the predictor needs gradients (target is frozen).
        return self.predictor.parameters(recurse=recurse)
