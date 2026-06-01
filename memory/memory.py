import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    def __init__(self, dim=256, num_slots=8):
        super().__init__()
        self.num_slots = num_slots
        self.memory = nn.Parameter(torch.randn(num_slots, dim))

        self.readout = nn.Linear(dim, dim)

    def init_memory(self, batch_size):
        return self.memory.unsqueeze(0).repeat(batch_size, 1, 1)

    def update(self, memory, x):
        B, K, D = memory.shape
        x = x.unsqueeze(1)

        attn = torch.softmax(torch.bmm(memory, x.transpose(1, 2)), dim=1)
        memory = memory + attn * x
        return memory

    def reconstruct(self, memory):
        # (B, K, D)
        return self.readout(memory)