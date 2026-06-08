import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Mapping

from .memory import MemoryModule, MemoryAdapter, MemoryReadout


STATE_DIM = 15
ACTION_DIM = 8
D_MODEL = 512
NUM_MEMORY_SLOTS = 8


# =========================================================
# Frozen Encoders
# =========================================================

class FrozenVisionEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.vision = clip_model.vision_model
        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()

    def forward(self, images):
        with torch.no_grad():
            out = self.vision(pixel_values=images)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            return out.last_hidden_state[:, 0]


class FrozenTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text = clip_model.text_model
        for p in self.text.parameters():
            p.requires_grad = False
        self.text.eval()

    def forward(self, tokens):
        with torch.no_grad():
            if isinstance(tokens, Mapping) or hasattr(tokens, "keys"):
                out = self.text(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens.get("attention_mask", None),
                )
            else:
                out = self.text(input_ids=tokens)

            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            return out.last_hidden_state[:, 0]


# =========================================================
# State encoder
# =========================================================

class StateEncoder(nn.Module):
    def __init__(self, state_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# CLIP + Memory VLA
# =========================================================

class CLIPMemoryVLA(nn.Module):
    def __init__(
        self,
        clip_model,
        state_dim=15,
        action_dim=8,
        d_model=512,
        num_slots=8,
    ):
        super().__init__()

        self.vision = FrozenVisionEncoder(clip_model)
        self.text = FrozenTextEncoder(clip_model)

        vision_dim = clip_model.config.vision_config.hidden_size
        text_dim = clip_model.config.text_config.hidden_size

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_slots = num_slots

        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.state = StateEncoder(state_dim, d_model)

        self.memory = MemoryModule(d_model, num_slots)
        self.adapter = MemoryAdapter(d_model)
        self.readout = MemoryReadout(d_model)

        # policy = SINGLE STEP
        self.policy = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim),
        )

        self.ptp_head = nn.Linear(d_model, action_dim)

    def forward(self, images, states, text_tokens, memory=None, detach_memory=False, memory_update="all"):
        """
        images: (B,T,3,H,W)
        states: (B,T,15)
        text_tokens: (B,L) or a tokenizer dict with input_ids/attention_mask
        memory: optional incoming memory, (B,K,D). If None, reset memory.
        detach_memory: if True, detach incoming memory for truncated BPTT.
        """

        B, T, _ = states.shape

        if memory_update not in ("all", "last", "none"):
            raise ValueError(f"memory_update must be 'all', 'last', or 'none', got {memory_update}")

        instr = self.text(text_tokens)
        instr = self.text_proj(instr)
        instr = F.normalize(instr, dim=-1)

        if memory is None:
            # Fresh memory: use this at the beginning of each episode.
            mem = self.memory.init_memory(B).to(states.device)
        else:
            # Reuse memory across consecutive chunks/timesteps from the same episode.
            mem = memory.to(states.device)
            if detach_memory:
                mem = mem.detach()

        ptp_list = []
        action_pred = None

        for t in range(T):

            v = self.vision(images[:, t])
            v = self.vision_proj(v)

            s = self.state(states[:, t])

            v = F.normalize(v, dim=-1)
            s = F.normalize(s, dim=-1)

            x = v + s + instr

            should_update_memory = (
                memory_update == "all"
                or (memory_update == "last" and t == T - 1)
            )

            if should_update_memory:
                mem, mem_ctx = self.adapter(x.unsqueeze(1), mem, instr)
            else:
                mem_ctx = mem.mean(dim=1)

            ctx = self.readout(x, mem, instr)

            fused = torch.cat([x + mem_ctx, ctx], dim=-1)

            # ONLY last step produces action
            if t == T - 1:
                action_pred = self.policy(fused)

            ptp_list.append(self.ptp_head(x))

        ptp_pred = torch.stack(ptp_list, dim=1)

        return {
            "pred_action": action_pred,   # (B, A)
            "pred_ptp": ptp_pred,         # (B, T, A)
            "memory": mem,
        }


# =========================================================
# SimpleVLA (unchanged reference, correct shape)
# =========================================================

class SimpleVLA(nn.Module):

    def __init__(
        self,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
    ):
        super().__init__()

        self.state_enc = nn.Linear(state_dim, d_model)

        self.image_enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True
        )

        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.memory = MemoryModule(d_model, NUM_MEMORY_SLOTS)

        self.policy_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )

        self.ptp_head = nn.Linear(d_model, action_dim)

    def forward(self, states, images):

        B, T, _ = states.shape

        s = self.state_enc(states)

        B, T, C, H, W = images.shape
        img = images.view(B * T, C, H, W)
        img = self.image_enc(img).view(B, T, -1)

        x = s + img
        x = self.temporal(x)

        mem = self.memory.init_memory(B).to(states.device)

        for t in range(T):
            mem = self.memory.update(mem, x[:, t])

        memory_summary = mem.mean(dim=1)
        current = x[:, -1]

        action_pred = self.policy_head(torch.cat([current, memory_summary], dim=-1))
        ptp_pred = self.ptp_head(x)

        return action_pred, ptp_pred, mem