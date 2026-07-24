from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = max(output_dim, min(256, input_dim * 2))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, output_dim), nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ModalityAttention(nn.Module):
    def __init__(self, representation_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(representation_dim, 32), nn.Tanh(), nn.Linear(32, 1, bias=False),
        )

    def forward(self, representations: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(representations, dim=1)
        weights = torch.softmax(self.score(stacked), dim=1)
        return (weights * stacked).sum(dim=1)


def contrastive_loss(
    first: torch.Tensor, second: torch.Tensor, temperature: float = 0.07,
) -> torch.Tensor:
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
