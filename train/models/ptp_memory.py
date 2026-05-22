"""
Transformer memory module + Past-Token Prediction (PTP) auxiliary loss.

Architecture:
  - Learnable CLS token prepended to K history tokens
  - 2-layer transformer encoder (multi-head self-attention)
  - CLS token output → memory feature

PTP auxiliary loss (Torne et al., 2025):
  For each timestep t in a rollout, given the stored K-step history:
    L_past   = MSE(predict(history_action[:-j])  vs history_action[-j])   over j=1..K
    L_future = MSE(predict(next action)           vs actual next action)
  L_PTP = (L_past + L_future) / 2

The loss is computed in PTPCallback during the SB3 training update.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryTransformer(nn.Module):
    """
    Transformer over K history tokens → memory embedding.

    Input:  (B, K, token_dim)  where token_dim = state_dim + action_dim
    Output: (B, out_dim)       CLS token representation
    """

    def __init__(self, token_dim: int, out_dim: int,
                 n_heads: int = 4, n_layers: int = 2, K: int = 8):
        super().__init__()
        d_model = max(64, (token_dim // n_heads + 1) * n_heads)  # round up to head multiple

        self.token_proj = nn.Linear(token_dim, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed  = nn.Parameter(torch.zeros(1, K + 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.size(0)
        x = self.token_proj(tokens)                                # (B, K, d_model)
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                            # (B, K+1, d_model)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        return self.out_proj(x[:, 0])                              # CLS output


class PTPHead(nn.Module):
    """
    Past-Token Prediction head.

    Given the CLS memory embedding, predict:
      1. Each past action token a_{t-j}  (past prediction)
      2. The next action a_{t+1}         (future prediction)

    Both predictions share a 2-layer MLP.
    """

    def __init__(self, memory_dim: int, action_dim: int, K: int = 8):
        super().__init__()
        self.K = K
        self.action_dim = action_dim

        # Predict all K past actions in one shot: output (K + 1) * action_dim
        self.head = nn.Sequential(
            nn.Linear(memory_dim, 256),
            nn.ReLU(),
            nn.Linear(256, (K + 1) * action_dim),  # K past + 1 future
        )

    def forward(self, memory_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (past_pred, future_pred) both shape (B, K, action_dim) and (B, action_dim)."""
        out = self.head(memory_feat)                           # (B, (K+1)*action_dim)
        out = out.view(out.size(0), self.K + 1, self.action_dim)
        past_pred   = out[:, :self.K, :]                      # (B, K, action_dim)
        future_pred = out[:, self.K,  :]                      # (B, action_dim)
        return past_pred, future_pred


def ptp_loss(past_pred: torch.Tensor,
             future_pred: torch.Tensor,
             history_action: torch.Tensor,
             next_action: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute PTP loss.

    Args:
        past_pred:       (B, K, action_dim)  — predicted past actions
        future_pred:     (B, action_dim)     — predicted next action
        history_action:  (B, K, action_dim)  — ground-truth past actions
        next_action:     (B, action_dim)     — ground-truth next action (from rollout)
        mask:            (B, K) bool — True for valid (non-padded) history steps

    Returns:
        scalar loss
    """
    if mask is not None:
        # Only penalise on non-padded history tokens
        past_loss = (F.mse_loss(past_pred, history_action, reduction="none") * mask.unsqueeze(-1)).mean()
    else:
        past_loss = F.mse_loss(past_pred, history_action)

    future_loss = F.mse_loss(future_pred, next_action)
    return (past_loss + future_loss) * 0.5
