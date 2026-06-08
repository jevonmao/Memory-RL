import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Memory Module
# =========================================================

class MemoryModule(nn.Module):
    """
    Slot-based recurrent memory.
    """

    def __init__(self, dim=256, num_slots=8):
        super().__init__()

        self.num_slots = num_slots

        self.memory = nn.Parameter(
            torch.randn(num_slots, dim) * 0.02
        )

    def init_memory(self, batch_size):
        return self.memory.unsqueeze(0).repeat(batch_size, 1, 1).clone()

    def update(self, memory, x):
        """
        memory: (B, K, D)
        x: (B, D)
        """

        x = x.unsqueeze(1)  # (B,1,D)

        attn = torch.softmax(
            torch.bmm(memory, x.transpose(1, 2)) / (memory.size(-1) ** 0.5),
            dim=1
        )

        # stabilize update (IMPORTANT)
        memory = memory + attn * x
        memory = F.layer_norm(memory, memory.shape[-1:])

        return memory


# =========================================================
# Memory Adapter (WRITE)
# =========================================================

class MemoryAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, hidden_states, memory, instr_emb):
        """
        hidden_states: (B,T,D)
        memory: (B,K,D)
        instr_emb: (B,D)
        """

        B, T, D = hidden_states.shape

        instr = instr_emb.unsqueeze(1).expand(-1, T, -1)

        for t in range(T):
            token = self.input_proj(hidden_states[:, t] + instr[:, t])
            memory = self._update(memory, token)

        mem_ctx = memory.mean(dim=1)

        return memory, mem_ctx

    def _update(self, memory, x):
        """
        memory: (B,K,D)
        x: (B,D)
        """
        x = x.unsqueeze(1)  # (B,1,D)

        attn = torch.softmax(
            torch.bmm(memory, x.transpose(1, 2)) / (memory.size(-1) ** 0.5),
            dim=1
        )

        memory = memory + attn * x
        memory = F.layer_norm(memory, memory.shape[-1:])

        return memory

# =========================================================
# Memory Readout (READ)
# =========================================================

class MemoryReadout(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, query, memory, instr_emb):
        """
        query: (B,D)
        memory: (B,K,D)
        instr_emb: (B,D)
        """

        query = query + instr_emb

        q = self.q(query).unsqueeze(1)   # (B,1,D)
        k = self.k(memory)               # (B,K,D)
        v = self.v(memory)

        scale = query.size(-1) ** 0.5

        attn = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) / scale,
            dim=-1
        )

        return torch.bmm(attn, v).squeeze(1)
    
# class MemoryModule(nn.Module):
#     def __init__(self, dim=256, num_slots=8):
#         super().__init__()
#         self.num_slots = num_slots
#         self.memory = nn.Parameter(torch.randn(num_slots, dim))

#         self.readout = nn.Linear(dim, dim)

#     def init_memory(self, batch_size):
#         return self.memory.unsqueeze(0).repeat(batch_size, 1, 1)

#     def update(self, memory, x):
#         B, K, D = memory.shape
#         x = x.unsqueeze(1)

#         attn = torch.softmax(torch.bmm(memory, x.transpose(1, 2)), dim=1)
#         memory = memory + attn * x
#         return memory

#     def reconstruct(self, memory):
#         # (B, K, D)
#         return self.readout(memory)

