from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .common import MLPEncoder, ModalityAttention


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


class NeighborhoodAggregator(nn.Module):
    """Encode a fixed 3x3 fused-representation neighborhood with an explicit mask."""

    def __init__(self, representation_dim: int, dropout: float) -> None:
        super().__init__()
        attention_heads = 4 if representation_dim % 4 == 0 else 1
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(representation_dim + 1, representation_dim, kernel_size=3),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.cross_attention = nn.MultiheadAttention(
            representation_dim, attention_heads, dropout=dropout, batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(representation_dim * 2, representation_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(representation_dim)

    def forward(
        self,
        fused_nodes: torch.Tensor,
        neighborhood_indices: torch.Tensor,
        neighborhood_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if neighborhood_indices.ndim != 2 or neighborhood_indices.shape[1] != 9:
            raise ValueError("neighborhood_indices must have shape [batch, 9]")
        if neighborhood_mask.shape != neighborhood_indices.shape:
            raise ValueError("neighborhood_mask must match neighborhood_indices")
        mask = neighborhood_mask.to(dtype=torch.bool)
        if not torch.all(mask[:, 4]):
            raise ValueError("Every 3x3 neighborhood must contain its center node")

        gathered = fused_nodes[neighborhood_indices]
        gathered = gathered * mask.unsqueeze(-1).to(gathered.dtype)
        center = gathered[:, 4]
        spatial = gathered.transpose(1, 2).reshape(
            gathered.shape[0], gathered.shape[2], 3, 3,
        )
        mask_channel = mask.to(gathered.dtype).reshape(-1, 1, 3, 3)
        spatial_context = self.spatial_encoder(torch.cat([spatial, mask_channel], dim=1)).flatten(1)
        attended, _ = self.cross_attention(
            center.unsqueeze(1), gathered, gathered, key_padding_mask=~mask,
            need_weights=False,
        )
        context = self.output(torch.cat([spatial_context, attended.squeeze(1)], dim=-1))
        return center, self.norm(center + context)


class MeanMLPGatedNeighborhoodAggregator(nn.Module):
    """Fuse the center with a masked mean of its fixed 3x3 neighborhood."""

    def __init__(self, representation_dim: int, dropout: float) -> None:
        super().__init__()
        self.neighborhood_mlp = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(representation_dim, representation_dim),
        )
        self.gate = nn.Linear(representation_dim * 2, representation_dim)

    def forward(
        self,
        fused_nodes: torch.Tensor,
        neighborhood_indices: torch.Tensor,
        neighborhood_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if neighborhood_indices.ndim != 2 or neighborhood_indices.shape[1] != 9:
            raise ValueError("neighborhood_indices must have shape [batch, 9]")
        if neighborhood_mask.shape != neighborhood_indices.shape:
            raise ValueError("neighborhood_mask must match neighborhood_indices")
        mask = neighborhood_mask.to(dtype=torch.bool)
        if not torch.all(mask[:, 4]):
            raise ValueError("Every 3x3 neighborhood must contain its center node")

        gathered = fused_nodes[neighborhood_indices]
        weights = mask.unsqueeze(-1).to(gathered.dtype)
        center = gathered[:, 4]
        mean = (gathered * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        neighborhood = self.neighborhood_mlp(mean)
        gate = torch.sigmoid(self.gate(torch.cat([center, neighborhood], dim=-1)))
        fused = gate * center + (1.0 - gate) * neighborhood
        return center, fused


class OpenCarbonModel(nn.Module):
    def __init__(
        self,
        remote_dim: int,
        environment_dim: int,
        representation_dim: int = 128,
        dropout: float = 0.2,
        neighborhood_aggregation: str = "spatial_attention",
        poi_input_mode: str = "dense",
    ) -> None:
        super().__init__()
        if poi_input_mode == "dense":
            self.poi_encoder = POIEncoder(17, representation_dim, use_se=True)
        elif poi_input_mode == "tabular":
            self.poi_encoder = MLPEncoder(17, representation_dim, dropout)
        else:
            raise ValueError(f"Unknown poi_input_mode={poi_input_mode}")
        self.remote_encoder = MLPEncoder(remote_dim, representation_dim, dropout)
        self.environment_encoder = (
            MLPEncoder(environment_dim, representation_dim, dropout)
            if environment_dim > 0 else None
        )
        self.modality_attention = ModalityAttention(representation_dim)
        aggregators = {
            "spatial_attention": NeighborhoodAggregator,
            "mean_mlp_gate": MeanMLPGatedNeighborhoodAggregator,
        }
        if neighborhood_aggregation not in aggregators:
            raise ValueError(
                f"Unknown neighborhood_aggregation: {neighborhood_aggregation}; "
                f"expected one of {sorted(aggregators)}"
            )
        self.neighborhood_aggregation = neighborhood_aggregation
        self.neighborhood_aggregator = aggregators[neighborhood_aggregation](
            representation_dim, dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(representation_dim * 2, representation_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(representation_dim, 1),
        )

    def forward_encoded(
        self,
        poi_representation: torch.Tensor,
        remote: torch.Tensor,
        environment: torch.Tensor,
        neighborhood_indices: torch.Tensor,
        neighborhood_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        remote_representation = self.remote_encoder(remote)
        representations = [poi_representation, remote_representation]
        if self.environment_encoder is not None:
            representations.append(self.environment_encoder(environment))
        grid_representation = self.modality_attention(representations)
        center_representation, neighborhood_representation = self.neighborhood_aggregator(
            grid_representation, neighborhood_indices, neighborhood_mask,
        )
        prediction = self.regressor(torch.cat([
            center_representation, neighborhood_representation,
        ], dim=-1)).squeeze(-1)
        center_indices = neighborhood_indices[:, 4]
        return prediction, poi_representation[center_indices], remote_representation[center_indices]

    def forward(
        self,
        poi: torch.Tensor,
        remote: torch.Tensor,
        environment: torch.Tensor,
        neighborhood_indices: torch.Tensor,
        neighborhood_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_encoded(
            self.poi_encoder(poi), remote, environment,
            neighborhood_indices, neighborhood_mask,
        )


class Stage1OpenCarbonModel(nn.Module):
    """POI-free OpenCarbon variant for same-month within-city grid transfer."""

    def __init__(
        self,
        feature_dim: int,
        representation_dim: int = 128,
        time_embedding_dim: int = 32,
        num_periods: int = 36,
        dropout: float = 0.2,
        neighborhood_aggregation: str = "mean_mlp_gate",
    ) -> None:
        super().__init__()
        self.feature_encoder = MLPEncoder(feature_dim, representation_dim, dropout)
        self.time_embedding = nn.Embedding(num_periods, time_embedding_dim)
        aggregators = {
            "spatial_attention": NeighborhoodAggregator,
            "mean_mlp_gate": MeanMLPGatedNeighborhoodAggregator,
        }
        if neighborhood_aggregation not in aggregators:
            raise ValueError(
                f"Unknown neighborhood_aggregation: {neighborhood_aggregation}; "
                f"expected one of {sorted(aggregators)}"
            )
        self.neighborhood_aggregation = neighborhood_aggregation
        self.neighborhood_aggregator = aggregators[neighborhood_aggregation](
            representation_dim, dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(representation_dim * 2 + time_embedding_dim, representation_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        period_ids: torch.Tensor,
        neighborhood_indices: torch.Tensor,
        neighborhood_mask: torch.Tensor,
    ) -> torch.Tensor:
        grid_representation = self.feature_encoder(features)
        center, neighborhood = self.neighborhood_aggregator(
            grid_representation, neighborhood_indices, neighborhood_mask,
        )
        time_representation = self.time_embedding(period_ids)
        return self.regressor(torch.cat([
            center, neighborhood, time_representation,
        ], dim=-1)).squeeze(-1)
