"""Implicit Q-Learning (IQL) — Kostrikov et al. 2021 (arXiv:2110.06169).

Offline RL algorithm that never queries out-of-distribution actions, making
it well-suited for learning from fixed demonstration datasets without a live
environment. Three networks:

  V(s)    — value network trained with expectile regression
  Q(s,a)  — twin critics trained with TD-backup using V(s')
  π(a|s)  — Gaussian actor trained with advantage-weighted regression (AWR)
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _mlp(dims: List[int], act: type = nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.net = _mlp([obs_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int):
        super().__init__()
        inp = obs_dim + action_dim
        self.q1 = _mlp([inp, hidden, hidden, 1])
        self.q2 = _mlp([inp, hidden, hidden, 1])

    def both(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(obs, act)
        return torch.minimum(q1, q2)


_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.trunk = _mlp([obs_dim, hidden, hidden])
        self.relu = nn.ReLU()
        self.mu  = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        h = self.relu(self.trunk(obs))
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return torch.distributions.Normal(mu, log_std.exp())

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).log_prob(action).sum(-1)

    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).sample()

    def mode(self, obs: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).mean


# ---------------------------------------------------------------------------
# Expectile loss
# ---------------------------------------------------------------------------

def _expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2:  E[ |τ - 𝟏(u<0)| · u² ]"""
    w = torch.where(diff >= 0, tau * torch.ones_like(diff), (1.0 - tau) * torch.ones_like(diff))
    return (w * diff.pow(2)).mean()


# ---------------------------------------------------------------------------
# IQL agent
# ---------------------------------------------------------------------------

class IQL:
    """Implicit Q-Learning agent (offline, no env interaction needed)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        temperature: float = 3.0,
        advantage_clip: float = 100.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.adv_clip = advantage_clip

        self.value  = ValueNet(obs_dim, hidden).to(self.device)
        self.qnet   = TwinQ(obs_dim, action_dim, hidden).to(self.device)
        self.q_tgt  = copy.deepcopy(self.qnet)
        self.actor  = GaussianActor(obs_dim, action_dim, hidden).to(self.device)

        for p in self.q_tgt.parameters():
            p.requires_grad_(False)

        self.v_opt     = Adam(self.value.parameters(), lr=lr)
        self.q_opt     = Adam(self.qnet.parameters(),  lr=lr)
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)

    # ------------------------------------------------------------------
    def _soft_update(self) -> None:
        for p, pt in zip(self.qnet.parameters(), self.q_tgt.parameters()):
            pt.data.lerp_(p.data, self.tau)

    # ------------------------------------------------------------------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs      = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        actions  = batch["actions"].to(self.device)
        rewards  = batch["rewards"].to(self.device)
        dones    = batch["dones"].to(self.device)

        # ---- Value update: expectile regression on min-Q ----
        with torch.no_grad():
            q_ref = self.q_tgt.min(obs, actions)
        v_pred = self.value(obs)
        v_loss = _expectile_loss(q_ref - v_pred, self.expectile)

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # ---- Q update: Bellman backup with V(s') ----
        with torch.no_grad():
            v_next  = self.value(next_obs)
            q_backup = rewards + self.gamma * (1.0 - dones) * v_next

        q1, q2 = self.qnet.both(obs, actions)
        q_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        self._soft_update()

        # ---- Actor update: advantage-weighted regression (AWR) ----
        with torch.no_grad():
            adv = (self.q_tgt.min(obs, actions) - self.value(obs))
            adv = adv.clamp(-self.adv_clip, self.adv_clip)
            weights = (self.temperature * adv).exp().clamp(max=self.adv_clip)

        log_pi   = self.actor.log_prob(obs, actions)
        actor_loss = -(weights * log_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return {
            "v_loss":     v_loss.item(),
            "q_loss":     q_loss.item(),
            "actor_loss": actor_loss.item(),
            "v_mean":     v_pred.mean().item(),
            "q_mean":     q_ref.mean().item(),
            "adv_mean":   adv.mean().item(),
        }

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        torch.save({
            "value":  self.value.state_dict(),
            "qnet":   self.qnet.state_dict(),
            "q_tgt":  self.q_tgt.state_dict(),
            "actor":  self.actor.state_dict(),
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.value.load_state_dict(ckpt["value"])
        self.qnet.load_state_dict(ckpt["qnet"])
        self.q_tgt.load_state_dict(ckpt["q_tgt"])
        self.actor.load_state_dict(ckpt["actor"])

    # ------------------------------------------------------------------
    def predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return an action tensor given a batched obs tensor (on CPU)."""
        obs = obs.to(self.device)
        with torch.no_grad():
            return (self.actor.mode(obs) if deterministic else self.actor.sample(obs)).cpu()
