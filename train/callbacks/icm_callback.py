"""
ICM callback for SB3 PPO.

After each rollout collection the callback:
  1. Extracts CNN features for s_t and s_{t+1} from the rollout buffer.
  2. Adds the ICM intrinsic reward to the extrinsic rewards in-place.
  3. Runs an ICM gradient step on the combined forward + inverse loss.

The callback carries its own optimizer (Adam, separate from the PPO optimizer).
"""

from __future__ import annotations

import torch
import torch.optim as optim
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ICMCallback(BaseCallback):
    """
    Intrinsic Curiosity Module callback.

    Args:
        icm:         ICM module (train.models.icm.ICM)
        extractor:   Feature extractor (RobommeCNNExtractor) — used to encode obs
        eta:         Intrinsic reward scale (η)
        lr:          ICM learning rate
        beta:        ICM loss weighting (β * L_fwd + (1-β) * L_inv)
        device:      torch device
    """

    def __init__(self,
                 icm,
                 extractor,
                 eta: float = 0.01,
                 lr: float = 3e-4,
                 beta: float = 0.2,
                 device: str = "auto",
                 verbose: int = 0):
        super().__init__(verbose)
        self.icm       = icm
        self.extractor = extractor
        self.eta       = eta
        self.lr        = lr
        self.beta      = beta
        self._device   = device

    def _on_training_start(self) -> None:
        dev = self.model.device if self._device == "auto" else torch.device(self._device)
        self.icm.to(dev)
        self.extractor.to(dev)
        self._opt = optim.Adam(
            list(self.icm.parameters()) + list(self.extractor.parameters()),
            lr=self.lr,
        )
        self._dev = dev

    def _on_rollout_end(self) -> None:
        buf   = self.model.rollout_buffer
        n_env = self.model.n_envs
        steps = buf.buffer_size          # steps per env

        # Reconstruct (s_t, a_t, s_{t+1}) triples from the flat rollout buffer.
        # SB3 stores observations as (steps, n_env, *obs_shape) in buf.observations
        # (or buf.observations["key"] for Dict obs).
        obs_dict    = buf.observations   # dict[str, np.ndarray (steps, n_env, ...)]
        actions_np  = buf.actions        # (steps, n_env, action_dim)

        # Convert to tensors — process env-by-env to keep GPU memory bounded
        all_intr = np.zeros((steps, n_env), dtype=np.float32)

        self._opt.zero_grad()
        total_loss = torch.tensor(0.0, device=self._dev)

        for e in range(n_env):
            # Build obs tensor for each step
            obs_t_list   = self._slice_obs(obs_dict, e, slice(0,   steps - 1))
            obs_tp1_list = self._slice_obs(obs_dict, e, slice(1,   steps))
            act_t        = torch.tensor(
                actions_np[:-1, e], dtype=torch.float32, device=self._dev
            )  # (T-1, action_dim)

            with torch.no_grad():
                feat_t   = self.extractor(obs_t_list)
                feat_tp1 = self.extractor(obs_tp1_list)

            phi_hat, act_hat, intr = self.icm(feat_t.detach(), act_t, feat_tp1.detach())
            loss = self.icm.loss(phi_hat, feat_tp1.detach(), act_hat, act_t, beta=self.beta)
            total_loss = total_loss + loss

            intr_np = intr.detach().cpu().numpy()              # (T-1,)
            all_intr[:-1, e] = intr_np
            all_intr[-1,  e] = intr_np[-1]                    # repeat last step

        total_loss = total_loss / n_env
        total_loss.backward()
        self._opt.step()

        # Add scaled intrinsic reward to the buffer (in-place)
        buf.rewards += self.eta * all_intr

        if self.verbose >= 1:
            self.logger.record("icm/loss",        total_loss.item())
            self.logger.record("icm/intr_reward", float(all_intr.mean()))

    def _on_step(self) -> bool:
        return True

    # ------------------------------------------------------------------
    def _slice_obs(self,
                   obs_dict: dict,
                   env_idx:  int,
                   sl:       slice) -> dict[str, torch.Tensor]:
        """Pull a time-slice from the rollout buffer and move to device."""
        out = {}
        for key, arr in obs_dict.items():
            # arr shape: (steps, n_env, *obs_shape)
            out[key] = torch.tensor(arr[sl, env_idx], dtype=torch.float32, device=self._dev)
        return out
