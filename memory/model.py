import torch
import torch.nn as nn

from memory import MemoryModule

# --------------------------------------------------
# Global Hyperparameters
# --------------------------------------------------

HISTORY_HORIZON = 8

STATE_DIM = 15
ACTION_DIM = 8

D_MODEL = 256
NUM_MEMORY_SLOTS = 8


# --------------------------------------------------
# Model
# --------------------------------------------------

class SimpleVLA(nn.Module):

    def __init__(
        self,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
    ):
        super().__init__()

        # -------------------------
        # Encoders
        # -------------------------
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

        # -------------------------
        # Temporal encoder
        # -------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True
        )

        self.temporal = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # -------------------------
        # Memory module
        # -------------------------
        self.memory = MemoryModule(
            dim=d_model,
            num_slots=NUM_MEMORY_SLOTS
        )

        # -------------------------
        # Policy head
        # -------------------------
        self.policy_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )

        # -------------------------
        # PTP head (past action prediction)
        # -------------------------
        self.ptp_head = nn.Linear(d_model, action_dim)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, states, images):
        """
        states: (B, T, 15)
        images: (B, T, 3, H, W)

        Returns:
            action_pred: (B, 8)
            ptp_pred: (B, T, 8)
            memory: (B, K, D)
        """

        B, T, _ = states.shape

        # -------------------------
        # Encode states
        # -------------------------
        s = self.state_enc(states)  # (B, T, D)

        # -------------------------
        # Encode images (vectorized)
        # -------------------------
        B, T, C, H, W = images.shape

        img = images.view(B * T, C, H, W)
        img = self.image_enc(img)
        img = img.view(B, T, -1)

        # -------------------------
        # Fuse modalities
        # -------------------------
        x = s + img  # (B, T, D)

        # -------------------------
        # Temporal encoding
        # -------------------------
        x = self.temporal(x)  # (B, T, D)

        # -------------------------
        # Initialize memory
        # -------------------------
        mem = self.memory.init_memory(B)

        # -------------------------
        # Sequential memory update
        # -------------------------
        for t in range(T):
            mem = self.memory.update(
                mem,
                x[:, t]
            )

        # -------------------------
        # Memory summary
        # -------------------------
        memory_summary = mem.mean(dim=1)  # (B, D)

        # -------------------------
        # Policy input
        # -------------------------
        current_repr = x[:, -1]  # last timestep

        policy_input = torch.cat(
            [current_repr, memory_summary],
            dim=-1
        )

        action_pred = self.policy_head(policy_input)

        # -------------------------
        # PTP prediction (past actions)
        # -------------------------
        ptp_pred = self.ptp_head(x)

        return action_pred, ptp_pred, mem