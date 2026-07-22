from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = inputs.mean((2, 3), keepdim=True)
        weights = F.relu(self.fc1(weights), inplace=True)
        weights = torch.sigmoid(self.fc2(weights))
        return inputs * weights


class POIEncoder(nn.Module):
    """OpenCarbon POI encoder adapted from its 17-channel London-style branch."""

    def __init__(self, channels: int = 17, representation_dim: int = 128, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=7, stride=3, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=7, stride=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=3, padding=1, bias=False)
        self.se1 = SEModule(32) if use_se else nn.Identity()
        self.se2 = SEModule(32) if use_se else nn.Identity()
        self.output = nn.Linear(100, representation_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.se1(F.relu(self.bn1(self.conv1(inputs)), inplace=True))
        hidden = self.se2(F.relu(self.bn2(self.conv2(hidden)), inplace=True))
        hidden = self.conv3(hidden).flatten(1)
        return self.output(hidden)


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


class OpenCarbonModel(nn.Module):
    def __init__(
        self,
        remote_dim: int,
        environment_dim: int,
        neighbor_dim: int,
        representation_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.poi_encoder = POIEncoder(17, representation_dim, use_se=True)
        self.remote_encoder = MLPEncoder(remote_dim, representation_dim, dropout)
        self.environment_encoder = MLPEncoder(environment_dim, representation_dim, dropout)
        self.modality_attention = ModalityAttention(representation_dim)
        self.neighbor_encoder = MLPEncoder(neighbor_dim, representation_dim, dropout)
        self.cross_gate = nn.Sequential(
            nn.Linear(representation_dim * 2, representation_dim), nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(representation_dim * 2, representation_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(representation_dim, 1),
        )

    def forward(
        self,
        poi: torch.Tensor,
        remote: torch.Tensor,
        environment: torch.Tensor,
        neighbor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poi_representation = self.poi_encoder(poi)
        remote_representation = self.remote_encoder(remote)
        environment_representation = self.environment_encoder(environment)
        grid_representation = self.modality_attention([
            poi_representation, remote_representation, environment_representation,
        ])
        neighbor_representation = self.neighbor_encoder(neighbor)
        gate = self.cross_gate(torch.cat([grid_representation, neighbor_representation], dim=-1))
        cross_representation = grid_representation + gate * neighbor_representation
        prediction = self.regressor(
            torch.cat([cross_representation, neighbor_representation], dim=-1)
        ).squeeze(-1)
        return prediction, poi_representation, remote_representation


def contrastive_loss(
    first: torch.Tensor, second: torch.Tensor, temperature: float = 0.07,
) -> torch.Tensor:
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
