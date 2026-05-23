"""
Shared CNN feature extractor for all three baselines.

Used as the `features_extractor_class` in SB3 policies.
Encodes front + wrist RGB independently with a shared ResNet18 backbone,
then concatenates with the robot state vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
import gymnasium as gym


IMG_FEAT_DIM  = 256   # output dim per camera after projection
STATE_FEAT_DIM = 64   # robot state embedding dim
JOINT_FEAT_DIM = 64   # visual history embedding dim (Baseline 3)


def _to_chw(x: torch.Tensor) -> torch.Tensor:
    """Accept either (B, H, W, C) or (B, C, H, W) and return (B, C, H, W)."""
    if x.shape[-1] in (1, 3, 4):   # last dim is channels → HWC
        return x.permute(0, 3, 1, 2).contiguous()
    return x  # already CHW (SB3 VecTransposeImage already did it)


class _ResNetTrunk(nn.Module):
    """ResNet18 up to (but not including) the final FC, plus a projection head."""

    def __init__(self, out_dim: int = IMG_FEAT_DIM, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.body = nn.Sequential(*list(backbone.children())[:-1])  # drop FC
        if freeze_backbone:
            for p in self.body.parameters():
                p.requires_grad_(False)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W), values in [0, 255] uint8 → normalise
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)) \
          / torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return self.proj(self.body(x))


class RobommeCNNExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Baselines 1 & 2.

    Produces a flat feature vector:
        [front_feat(256) ‖ wrist_feat(256) ‖ state_feat(64)]  = 576-d
    """

    def __init__(self, observation_space: gym.spaces.Dict,
                 img_feat_dim: int = IMG_FEAT_DIM,
                 state_feat_dim: int = STATE_FEAT_DIM):

        features_dim = 2 * img_feat_dim + state_feat_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.front_enc = _ResNetTrunk(img_feat_dim)
        self.wrist_enc = _ResNetTrunk(img_feat_dim)

        # State: joint(7) + eef(6) + gripper(2) = 15
        state_in = (
            observation_space["joint_state"].shape[0]
            + observation_space["eef_state"].shape[0]
            + observation_space["gripper"].shape[0]
        )
        self.state_enc = nn.Sequential(
            nn.Linear(state_in, state_feat_dim),
            nn.LayerNorm(state_feat_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        front = _to_chw(obs["front_rgb"])
        wrist = _to_chw(obs["wrist_rgb"])
        state = torch.cat([obs["joint_state"], obs["eef_state"], obs["gripper"]], dim=-1)
        return torch.cat([
            self.front_enc(front),
            self.wrist_enc(wrist),
            self.state_enc(state),
        ], dim=-1)


class RobommeMemoryExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Baseline 3 (PPO + PTP Memory).

    Produces:
        [front_feat(256) ‖ wrist_feat(256) ‖ state_feat(64) ‖ memory_feat(256)]  = 832-d

    memory_feat comes from a 2-layer transformer over the K-step history of
    (state‖action) tokens, with a learned CLS token prepended.
    """

    def __init__(self, observation_space: gym.spaces.Dict,
                 img_feat_dim: int = IMG_FEAT_DIM,
                 state_feat_dim: int = STATE_FEAT_DIM,
                 memory_feat_dim: int = 256,
                 n_heads: int = 4,
                 n_layers: int = 2):

        features_dim = 2 * img_feat_dim + state_feat_dim + memory_feat_dim
        super().__init__(observation_space, features_dim=features_dim)

        from train.models.ptp_memory import MemoryTransformer
        K          = observation_space["history_state"].shape[0]
        state_dim  = observation_space["history_state"].shape[1]   # STATE_DIM = 15
        # PTP plan decision #4: input tokens are state-only so the past-action
        # prediction task isn't trivial (action would otherwise appear in the
        # input and be copyable). PTPHead still predicts past + future actions
        # from the CLS embedding; only the encoder's *input* changes.
        token_dim  = state_dim                                     # 15

        self.front_enc  = _ResNetTrunk(img_feat_dim)
        self.wrist_enc  = _ResNetTrunk(img_feat_dim)
        state_in = state_dim
        self.state_enc = nn.Sequential(
            nn.Linear(state_in, state_feat_dim),
            nn.LayerNorm(state_feat_dim),
            nn.ReLU(),
        )
        self.memory_transformer = MemoryTransformer(
            token_dim=token_dim,
            out_dim=memory_feat_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            K=K,
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        front = _to_chw(obs["front_rgb"])
        wrist = _to_chw(obs["wrist_rgb"])
        # current state (last entry of history, always valid)
        state = obs["history_state"][:, -1, :]   # (B, STATE_DIM)

        # State-only tokens (see __init__ comment for the why).
        tokens = obs["history_state"]            # (B, K, STATE_DIM)

        return torch.cat([
            self.front_enc(front),
            self.wrist_enc(wrist),
            self.state_enc(state),
            self.memory_transformer(tokens),
        ], dim=-1)
