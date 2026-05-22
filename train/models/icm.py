"""
Intrinsic Curiosity Module (ICM) — Pathak et al., 2017.

Forward model:  f(φ(s_t), a_t)          → φ̂(s_{t+1})
Inverse model:  g(φ(s_t), φ(s_{t+1}))   → â_t

Intrinsic reward:  r_i = η · ‖φ(s_{t+1}) − φ̂(s_{t+1})‖²

Both models operate on the 576-d CNN feature vectors produced by
RobommeCNNExtractor, so no separate image encoder is needed here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    def __init__(self, feat_dim: int = 576, action_dim: int = 8, hidden: int = 256):
        super().__init__()
        self.feat_dim   = feat_dim
        self.action_dim = action_dim

        # Forward model: (φ_t ‖ a_t) → φ_{t+1}
        self.forward_model = nn.Sequential(
            nn.Linear(feat_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )

        # Inverse model: (φ_t ‖ φ_{t+1}) → â_t
        self.inverse_model = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self,
                feat_t:   torch.Tensor,
                action_t: torch.Tensor,
                feat_tp1: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            phi_hat_tp1:  predicted next-state feature  (B, feat_dim)
            action_hat:   predicted action               (B, action_dim)
            intr_reward:  per-sample intrinsic reward   (B,)
        """
        # Forward
        fwd_input    = torch.cat([feat_t, action_t], dim=-1)
        phi_hat_tp1  = self.forward_model(fwd_input)

        # Inverse
        inv_input    = torch.cat([feat_t, feat_tp1], dim=-1)
        action_hat   = self.inverse_model(inv_input)

        # Intrinsic reward = forward prediction error (detach target)
        intr_reward  = F.mse_loss(phi_hat_tp1, feat_tp1.detach(), reduction="none").mean(-1)

        return phi_hat_tp1, action_hat, intr_reward

    def loss(self,
             phi_hat_tp1: torch.Tensor,
             feat_tp1:    torch.Tensor,
             action_hat:  torch.Tensor,
             action_t:    torch.Tensor,
             beta: float = 0.2) -> torch.Tensor:
        """
        ICM loss = β * L_forward + (1-β) * L_inverse

        β=0.2 weights forward loss slightly lower (inverse is easier to learn).
        """
        L_fwd = F.mse_loss(phi_hat_tp1, feat_tp1.detach())
        L_inv = F.mse_loss(action_hat,  action_t.detach())
        return beta * L_fwd + (1.0 - beta) * L_inv
