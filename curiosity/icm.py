"""Intrinsic Curiosity Module (Pathak et al., 2017).

Encoder phi maps obs -> feature; inverse model predicts a_t from (phi_t, phi_{t+1});
forward model predicts phi_{t+1} from (phi_t, a_t). Intrinsic reward is the forward
model's prediction error.

Supports Discrete and Box action spaces (CartPole and RoboMME-style continuous
control share one interface). For Box obs we assume a 1-D vector — RoboMME's
flatten_obs=True path already guarantees this.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from curiosity.base_curiosity import BaseCuriosity


def _mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class ICM(nn.Module, BaseCuriosity):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        eta: float = 0.01,
        beta: float = 0.2,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        nn.Module.__init__(self)
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise ValueError(f"ICM expects 1-D Box obs, got {observation_space}")
        obs_dim = int(observation_space.shape[0])

        self._discrete = isinstance(action_space, spaces.Discrete)
        if self._discrete:
            self._action_dim = int(action_space.n)
            act_in_dim = self._action_dim  # one-hot
        elif isinstance(action_space, spaces.Box) and len(action_space.shape) == 1:
            self._action_dim = int(action_space.shape[0])
            act_in_dim = self._action_dim
        else:
            raise ValueError(f"ICM expects Discrete or 1-D Box action, got {action_space}")

        self.feature_dim = feature_dim
        self.eta = float(eta)
        self.beta = float(beta)
        self.device = torch.device(device)

        self.encoder = _mlp([obs_dim, hidden_dim, feature_dim])
        self.inverse = _mlp([2 * feature_dim, hidden_dim, self._action_dim])
        self.forward_net = _mlp([feature_dim + act_in_dim, hidden_dim, feature_dim])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        if self._discrete:
            return F.one_hot(action.long(), num_classes=self._action_dim).float()
        return action.float()

    def _to_tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=dtype)
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    @torch.no_grad()
    def compute_intrinsic_reward(
        self,
        obs: Any,
        action: Any,
        next_obs: Any,
        memory_state: Optional[Any] = None,
    ):
        """Per-sample intrinsic reward. Accepts batched (B, ...) or single (...)."""
        single = np.asarray(obs).ndim == 1
        obs_t = self._to_tensor(np.atleast_2d(obs))
        next_t = self._to_tensor(np.atleast_2d(next_obs))
        act_arr = np.atleast_1d(action) if self._discrete else np.atleast_2d(action)
        act_t = self._to_tensor(act_arr, dtype=torch.long if self._discrete else torch.float32)

        phi = self.encoder(obs_t)
        phi_next = self.encoder(next_t)
        a_enc = self._encode_action(act_t)
        phi_pred = self.forward_net(torch.cat([phi, a_enc], dim=-1))
        r = 0.5 * self.eta * (phi_pred - phi_next).pow(2).sum(dim=-1)
        out = r.cpu().numpy().astype(np.float32)
        return float(out[0]) if single else out

    def update(self, batch: Mapping[str, Any]) -> Mapping[str, float]:
        obs = self._to_tensor(batch["obs"])
        next_obs = self._to_tensor(batch["next_obs"])
        if self._discrete:
            act = self._to_tensor(batch["action"], dtype=torch.long)
        else:
            act = self._to_tensor(batch["action"])

        phi = self.encoder(obs)
        phi_next = self.encoder(next_obs)
        a_enc = self._encode_action(act)

        phi_pred = self.forward_net(torch.cat([phi, a_enc], dim=-1))
        fwd_loss = 0.5 * F.mse_loss(phi_pred, phi_next.detach(), reduction="mean")

        inv_logits = self.inverse(torch.cat([phi, phi_next], dim=-1))
        if self._discrete:
            inv_loss = F.cross_entropy(inv_logits, act)
        else:
            inv_loss = F.mse_loss(inv_logits, act)

        loss = self.beta * fwd_loss + (1.0 - self.beta) * inv_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "icm/loss": float(loss.detach().cpu()),
            "icm/forward_loss": float(fwd_loss.detach().cpu()),
            "icm/inverse_loss": float(inv_loss.detach().cpu()),
        }
