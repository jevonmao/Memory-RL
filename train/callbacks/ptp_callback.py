"""
PTP (Past-Token Prediction) callback for SB3 PPO + MemoryTransformer.

After each rollout the callback:
  1. Iterates over timesteps in the rollout buffer.
  2. For each step, uses the history_state / history_action from obs to
     build PTP targets and runs a gradient step on the PTP head.

The callback carries its own optimizer (Adam, separate from PPO's).
"""

from __future__ import annotations

import torch
import torch.optim as optim
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from train.models.ptp_memory import PTPHead, ptp_loss


class PTPCallback(BaseCallback):
    """
    Past-Token Prediction auxiliary loss callback.

    Args:
        memory_transformer: MemoryTransformer (inside RobommeMemoryExtractor)
        ptp_head:           PTPHead module
        action_dim:         action space dimensionality (8)
        K:                  history length
        lr:                 learning rate for PTP parameters
        ptp_weight:         scalar multiplier for PTP loss contribution
        device:             torch device
    """

    def __init__(self,
                 memory_transformer,
                 ptp_head:    PTPHead,
                 action_dim:  int = 8,
                 K:           int = 8,
                 lr:          float = 3e-4,
                 ptp_weight:  float = 0.1,
                 device:      str = "auto",
                 verbose:     int = 0):
        super().__init__(verbose)
        self.memory_transformer = memory_transformer
        self.ptp_head    = ptp_head
        self.action_dim  = action_dim
        self.K           = K
        self.lr          = lr
        self.ptp_weight  = ptp_weight
        self._device     = device

    def _on_training_start(self) -> None:
        dev = self.model.device if self._device == "auto" else torch.device(self._device)
        self.memory_transformer.to(dev)
        self.ptp_head.to(dev)
        self._opt = optim.Adam(
            list(self.memory_transformer.parameters()) + list(self.ptp_head.parameters()),
            lr=self.lr,
        )
        self._dev = dev

    def _on_rollout_end(self) -> None:
        buf   = self.model.rollout_buffer
        n_env = self.model.n_envs
        steps = buf.buffer_size
        obs   = buf.observations    # dict of (steps, n_env, ...)
        acts  = buf.actions         # (steps, n_env, action_dim)

        # history_state:  (steps, n_env, K, STATE_DIM)
        # history_action: (steps, n_env, K, ACTION_DIM)
        hist_s = torch.tensor(obs["history_state"],  dtype=torch.float32, device=self._dev)
        hist_a = torch.tensor(obs["history_action"], dtype=torch.float32, device=self._dev)
        acts_t = torch.tensor(acts,                  dtype=torch.float32, device=self._dev)

        # Flatten (steps, n_env) → batch
        T, E  = steps, n_env
        hist_s  = hist_s.view(T * E, self.K, -1)          # (T*E, K, STATE_DIM)
        hist_a  = hist_a.view(T * E, self.K, self.action_dim)
        acts_t  = acts_t.view(T * E, self.action_dim)     # (T*E, action_dim)

        # Compute PTP targets: next action is acts_t, history actions are hist_a
        # Build token sequence: (state ‖ action) per step
        tokens = torch.cat([hist_s, hist_a], dim=-1)       # (T*E, K, token_dim)

        self._opt.zero_grad()
        memory_feat = self.memory_transformer(tokens)      # (T*E, memory_dim)
        past_pred, future_pred = self.ptp_head(memory_feat)

        # Build mask: zero-padded entries in history_action have norm ≈ 0
        mask = (hist_a.abs().sum(-1) > 1e-6).float()      # (T*E, K)

        loss = ptp_loss(past_pred, future_pred, hist_a, acts_t, mask=mask)
        (self.ptp_weight * loss).backward()
        self._opt.step()

        if self.verbose >= 1:
            self.logger.record("ptp/loss", loss.item())

    def _on_step(self) -> bool:
        return True
