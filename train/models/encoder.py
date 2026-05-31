"""
Shared CNN feature extractor for all three baselines.

Used as the `features_extractor_class` in SB3 policies.

Optimizations vs the v4 commit:
  * One shared ResNet18 backbone (frozen) processes front + wrist as a single
    (2B, 3, H, W) batch — previously two separate backbones doubled latency.
  * Per-camera projection heads still differ so the model can specialise.
  * Frozen backbone runs under torch.no_grad() so we don't build / store an
    autograd graph through 11M unused parameters every minibatch.
  * channels_last memory format on the conv input — Ada Lovelace + cuDNN
    routes NHWC through faster kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
import gymnasium as gym


IMG_FEAT_DIM   = 256   # output dim per camera after projection
STATE_FEAT_DIM = 64    # robot state embedding dim
JOINT_FEAT_DIM = 64    # visual history embedding dim (Baseline 3)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _to_chw(x: torch.Tensor) -> torch.Tensor:
    """Accept either (B, H, W, C) or (B, C, H, W) and return (B, C, H, W)."""
    if x.shape[-1] in (1, 3, 4):
        return x.permute(0, 3, 1, 2).contiguous()
    return x


class _SharedResNetBackbone(nn.Module):
    """Frozen ResNet18 trunk (no FC). Forwards under no_grad — never trained."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.body = nn.Sequential(*list(backbone.children())[:-1])
        for p in self.body.parameters():
            p.requires_grad_(False)
        self.body.eval()
        # Pre-bake normalization constants so we don't materialise tensors per call.
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std",  torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Keep the backbone in eval mode forever (BatchNorm stats are frozen).
        super().train(mode)
        self.body.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W) uint8 or float32. Always run under no_grad.
        with torch.no_grad():
            if x.dtype == torch.uint8:
                x = x.float().mul_(1.0 / 255.0)
            x = (x - self._mean) / self._std
            x = x.to(memory_format=torch.channels_last)
            feat = self.body(x)            # (N, 512, 1, 1)
        return feat.flatten(1)             # (N, 512); grad will flow into proj head only


class _ProjHead(nn.Module):
    """Per-camera trainable projection from the shared 512-d trunk feature."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


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

        self.backbone   = _SharedResNetBackbone()
        self.front_head = _ProjHead(img_feat_dim)
        self.wrist_head = _ProjHead(img_feat_dim)

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
        B = front.shape[0]
        # Single batched backbone pass over (2B, 3, H, W); split + project.
        stacked = torch.cat([front, wrist], dim=0)
        trunk = self.backbone(stacked)                    # (2B, 512)
        front_feat = self.front_head(trunk[:B])
        wrist_feat = self.wrist_head(trunk[B:])
        state = torch.cat([obs["joint_state"], obs["eef_state"], obs["gripper"]], dim=-1)
        return torch.cat([front_feat, wrist_feat, self.state_enc(state)], dim=-1)


# ---------------------------------------------------------------------------
# Legacy architecture (pre-refactor) — kept for loading old checkpoints like
# ppo_v2 / ppo_v3 / ppo_v4 that were saved before the shared-backbone change.
# DO NOT use for new training runs.
# ---------------------------------------------------------------------------

class _ResNetTrunkLegacy(nn.Module):
    """The original per-camera ResNet18 trunk + projection head.

    Matches state_dict keys: front_enc.body.*, front_enc.proj.*  (and wrist_enc).
    """

    def __init__(self, out_dim: int = IMG_FEAT_DIM, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.body = nn.Sequential(*list(backbone.children())[:-1])
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
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.proj(self.body(x))


class RobommeCNNExtractorLegacy(BaseFeaturesExtractor):
    """Pre-refactor RobommeCNNExtractor with two separate ResNet18 trunks.

    State_dict keys: front_enc.{body,proj}.*, wrist_enc.{body,proj}.*,
                     state_enc.{0,1}.*.
    Output is the same 576-d vector as the new shared-backbone version, so
    the downstream policy heads (action_net, value_net) are interchangeable.
    """

    def __init__(self, observation_space: gym.spaces.Dict,
                 img_feat_dim: int = IMG_FEAT_DIM,
                 state_feat_dim: int = STATE_FEAT_DIM):
        features_dim = 2 * img_feat_dim + state_feat_dim
        super().__init__(observation_space, features_dim=features_dim)
        self.front_enc = _ResNetTrunkLegacy(img_feat_dim)
        self.wrist_enc = _ResNetTrunkLegacy(img_feat_dim)
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


def detect_encoder_layout(state_dict_keys) -> str:
    """Inspect SB3 state_dict keys and return 'new' or 'legacy'.

    The SB3 model's policy state_dict prefixes extractor keys with
    'features_extractor.'. Both layouts share state_enc.* but differ in:
       new layout:    features_extractor.backbone.body.* + .front_head.* + .wrist_head.*
       legacy layout: features_extractor.front_enc.body.* + .wrist_enc.body.*
    """
    keys = list(state_dict_keys)
    has_legacy = any("front_enc.body" in k or "wrist_enc.body" in k for k in keys)
    has_new    = any("backbone.body"   in k for k in keys)
    if has_legacy and not has_new:
        return "legacy"
    if has_new and not has_legacy:
        return "new"
    if not has_legacy and not has_new:
        raise ValueError(
            "state dict has neither front_enc.body nor backbone.body keys; "
            "is this a RobommeCNNExtractor checkpoint?"
        )
    # Both present — shouldn't happen, but prefer new.
    return "new"


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

        self.backbone   = _SharedResNetBackbone()
        self.front_head = _ProjHead(img_feat_dim)
        self.wrist_head = _ProjHead(img_feat_dim)
        self.state_enc  = nn.Sequential(
            nn.Linear(state_dim, state_feat_dim),
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
        B = front.shape[0]
        trunk = self.backbone(torch.cat([front, wrist], dim=0))
        front_feat = self.front_head(trunk[:B])
        wrist_feat = self.wrist_head(trunk[B:])

        state = obs["history_state"][:, -1, :]
        tokens = obs["history_state"]

        return torch.cat([
            front_feat,
            wrist_feat,
            self.state_enc(state),
            self.memory_transformer(tokens),
        ], dim=-1)
