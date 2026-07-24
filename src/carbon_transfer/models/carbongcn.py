from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adjacency, self.linear(inputs))


class CarbonGCN(nn.Module):
    def __init__(self, input_dim: int, representation_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = GraphConvolution(input_dim, representation_dim)
        self.conv2 = GraphConvolution(representation_dim, representation_dim)
        self.dropout = dropout
        self.output = nn.Linear(representation_dim, 1)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv1(inputs, adjacency))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = F.relu(self.conv2(hidden, adjacency))
        return self.output(hidden).squeeze(-1)
