from __future__ import annotations

import torch
from torch import nn


class BPNN(nn.Module):
    def __init__(self, input_dim: int, representation_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, representation_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(representation_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)
