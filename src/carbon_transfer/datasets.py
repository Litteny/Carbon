from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class BalancedEnvironmentBatchSampler(Sampler[List[int]]):
    """Yield deterministic, environment-balanced batches of row indices."""

    def __init__(
        self,
        environment_ids: Sequence[int],
        batch_size: int,
        seed: int,
    ) -> None:
        self.environment_ids = np.asarray(environment_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.environments = sorted(np.unique(self.environment_ids).tolist())
        if len(self.environments) < 2:
            raise ValueError("VREx training requires at least two source environments")
        if self.batch_size < len(self.environments):
            raise ValueError(
                f"batch_size={self.batch_size} must be at least the number of "
                f"source environments ({len(self.environments)})"
            )
        self.indices = {
            environment: np.flatnonzero(self.environment_ids == environment)
            for environment in self.environments
        }
        if any(not len(values) for values in self.indices.values()):
            raise ValueError("Every source environment must contain at least one sample")
        self.num_batches = int(np.ceil(
            len(self.environment_ids) / self.batch_size,
        ))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        generator = np.random.default_rng(self.seed + self.epoch)
        base, remainder = divmod(self.batch_size, len(self.environments))
        per_environment = {
            environment: base + (position < remainder)
            for position, environment in enumerate(self.environments)
        }
        pools = {
            environment: generator.permutation(indices)
            for environment, indices in self.indices.items()
        }
        positions = {environment: 0 for environment in self.environments}
        for _ in range(self.num_batches):
            batch: List[int] = []
            for environment in self.environments:
                required = per_environment[environment]
                selected = []
                while len(selected) < required:
                    pool = pools[environment]
                    position = positions[environment]
                    available = min(required - len(selected), len(pool) - position)
                    selected.extend(pool[position:position + available].tolist())
                    positions[environment] += available
                    if positions[environment] == len(pool):
                        pools[environment] = generator.permutation(self.indices[environment])
                        positions[environment] = 0
                batch.extend(selected)
            generator.shuffle(batch)
            yield batch


def sample_key(city_id: str, period: str, cell_id: str) -> Tuple[str, str, str]:
    return str(city_id), str(period), str(cell_id)


def masked_edges(frame: pd.DataFrame, neighbors: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    index = {
        sample_key(row.city_id, row.period, row.cell_id): position
        for position, row in enumerate(frame.itertuples())
    }
    sources: List[int] = []
    targets: List[int] = []
    relevant_cities = set(frame["city_id"].astype(str))
    relevant_periods = set(frame["period"].astype(str))
    candidate = neighbors[
        neighbors["city_id"].astype(str).isin(relevant_cities)
        & neighbors["period"].astype(str).isin(relevant_periods)
    ]
    for row in candidate.itertuples():
        source = index.get(sample_key(row.city_id, row.period, row.cell_id))
        target = index.get(sample_key(row.city_id, row.period, row.neighbor_cell_id))
        if source is not None and target is not None:
            sources.append(source)
            targets.append(target)
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def neighbor_means(features: np.ndarray, frame: pd.DataFrame, neighbors: pd.DataFrame) -> np.ndarray:
    sources, targets = masked_edges(frame, neighbors)
    output = np.zeros_like(features, dtype=np.float32)
    counts = np.zeros(len(frame), dtype=np.float32)
    np.add.at(output, sources, features[targets])
    np.add.at(counts, sources, 1.0)
    missing = counts == 0
    counts[missing] = 1.0
    output /= counts[:, None]
    output[missing] = features[missing]
    return output


def fixed_neighborhood_indices(frame: pd.DataFrame, neighbors: pd.DataFrame) -> np.ndarray:
    """Return split-masked 3x3 neighborhood indices in row-major offset order.

    Missing or cross-split neighbors are represented by ``-1``. The center
    grid occupies slot 4 and is required to be present for every sample.
    """
    index = {
        sample_key(row.city_id, row.period, row.cell_id): position
        for position, row in enumerate(frame.itertuples())
    }
    output = np.full((len(frame), 9), -1, dtype=np.int64)
    relevant_cities = set(frame["city_id"].astype(str))
    relevant_periods = set(frame["period"].astype(str))
    candidate = neighbors[
        neighbors["city_id"].astype(str).isin(relevant_cities)
        & neighbors["period"].astype(str).isin(relevant_periods)
    ]
    required = {"offset_row", "offset_col"}
    if not required.issubset(candidate.columns):
        raise ValueError("Neighbor table must include offset_row and offset_col")
    for row in candidate.itertuples():
        source = index.get(sample_key(row.city_id, row.period, row.cell_id))
        target = index.get(sample_key(row.city_id, row.period, row.neighbor_cell_id))
        offset_row, offset_col = int(row.offset_row), int(row.offset_col)
        if source is None or target is None:
            continue
        if not (-1 <= offset_row <= 1 and -1 <= offset_col <= 1):
            raise ValueError(f"Invalid neighborhood offset: {(offset_row, offset_col)}")
        slot = (offset_row + 1) * 3 + (offset_col + 1)
        output[source, slot] = target
    expected_centers = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(output[:, 4], expected_centers):
        missing = int(np.count_nonzero(output[:, 4] != expected_centers))
        raise ValueError(f"Fixed neighborhoods are missing {missing} center nodes")
    return output


def normalized_adjacency(frame: pd.DataFrame, neighbors: pd.DataFrame, device: torch.device) -> torch.Tensor:
    sources, targets = masked_edges(frame, neighbors)
    if not len(sources):
        sources = targets = np.arange(len(frame), dtype=np.int64)
    degree = np.bincount(sources, minlength=len(frame)).astype(np.float32)
    degree[degree == 0] = 1.0
    values = 1.0 / np.sqrt(degree[sources] * degree[targets])
    indices = torch.as_tensor(np.vstack([sources, targets]), dtype=torch.long, device=device)
    values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values_tensor, (len(frame), len(frame)), device=device).coalesce()


class SparsePOIStore:
    def __init__(self, root: Path, cache_size: int = 4) -> None:
        self.root = Path(root)
        self.cache_size = cache_size
        self.cache: OrderedDict = OrderedDict()

    def _load(self, city_id: str, period: str) -> Dict:
        key = (str(city_id), str(period))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.root / str(city_id) / f"{period}.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["cell_lookup"] = {str(value): index for index, value in enumerate(arrays["cell_id"])}
        self.cache[key] = arrays
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return arrays

    def dense(self, city_id: str, period: str, cell_id: str) -> np.ndarray:
        arrays = self._load(city_id, period)
        cell_index = arrays["cell_lookup"].get(str(cell_id))
        if cell_index is None:
            raise KeyError(f"POI cell not found: {city_id}/{period}/{cell_id}")
        start = int(arrays["offsets"][cell_index])
        end = int(arrays["offsets"][cell_index + 1])
        result = np.zeros((17, 256, 256), dtype=np.float32)
        if end > start:
            channels = arrays["channel"][start:end].astype(np.int64)
            pixel_y = arrays["pixel_y"][start:end].astype(np.int64)
            pixel_x = arrays["pixel_x"][start:end].astype(np.int64)
            result[channels, pixel_y, pixel_x] = arrays["count"][start:end]
        return result


class OpenCarbonDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        remote: np.ndarray,
        environment: np.ndarray,
        neighborhoods: np.ndarray,
        poi_root: Path,
        poi_cache_size: int = 4,
        environment_mapping: Optional[Mapping[str, int]] = None,
        environment_key: str = "city_id",
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.remote = remote.astype(np.float32, copy=False)
        self.environment = environment.astype(np.float32, copy=False)
        self.neighborhoods = neighborhoods.astype(np.int64, copy=False)
        if self.neighborhoods.shape != (len(self.frame), 9):
            raise ValueError(
                f"Expected neighborhoods with shape {(len(self.frame), 9)}, "
                f"found {self.neighborhoods.shape}"
            )
        self.targets = self.frame["log1p_emission"].to_numpy(dtype=np.float32)
        self.environment_mapping = dict(environment_mapping or {})
        self.environment_key = str(environment_key)
        if self.environment_key not in self.frame:
            raise ValueError(f"Environment column is absent: {self.environment_key}")
        unknown = sorted(
            set(self.frame[self.environment_key].astype(str))
            - set(self.environment_mapping)
        )
        self.environment_ids = np.asarray([
            self.environment_mapping.get(value, -1)
            for value in self.frame[self.environment_key].astype(str)
        ], dtype=np.int64)
        self.unknown_environments = unknown
        self.poi_store = SparsePOIStore(poi_root, cache_size=poi_cache_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def collate(self, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        target_indices = np.asarray(indices, dtype=np.int64)
        neighborhoods = self.neighborhoods[target_indices]
        neighborhood_mask = neighborhoods >= 0
        unique_indices = np.unique(neighborhoods[neighborhood_mask])
        if not len(unique_indices):
            raise ValueError("OpenCarbon batch contains no valid neighborhood nodes")

        local_neighborhoods = np.zeros_like(neighborhoods)
        local_neighborhoods[neighborhood_mask] = np.searchsorted(
            unique_indices, neighborhoods[neighborhood_mask],
        )
        poi = []
        for index in unique_indices:
            row = self.frame.iloc[int(index)]
            poi.append(self.poi_store.dense(row["city_id"], str(row["period"]), row["cell_id"]))
        return {
            "poi": torch.from_numpy(np.stack(poi)),
            "remote": torch.from_numpy(self.remote[unique_indices]),
            "environment": torch.from_numpy(self.environment[unique_indices]),
            "neighborhood_indices": torch.from_numpy(local_neighborhoods),
            "neighborhood_mask": torch.from_numpy(neighborhood_mask),
            "target": torch.from_numpy(self.targets[target_indices]),
            "index": torch.from_numpy(target_indices),
            "environment_id": torch.from_numpy(self.environment_ids[target_indices]),
        }


class PrecomputedOpenCarbonDataset(Dataset):
    """OpenCarbon dataset backed by frozen POI representations."""

    def __init__(
        self,
        frame: pd.DataFrame,
        remote: np.ndarray,
        environment: np.ndarray,
        neighborhoods: np.ndarray,
        poi_embeddings: np.ndarray,
        environment_mapping: Optional[Mapping[str, int]] = None,
        environment_key: str = "city_id",
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.remote = remote.astype(np.float32, copy=False)
        self.environment = environment.astype(np.float32, copy=False)
        self.neighborhoods = neighborhoods.astype(np.int64, copy=False)
        self.poi_embeddings = poi_embeddings.astype(np.float32, copy=False)
        if self.neighborhoods.shape != (len(self.frame), 9):
            raise ValueError(
                f"Expected neighborhoods with shape {(len(self.frame), 9)}, "
                f"found {self.neighborhoods.shape}"
            )
        if len(self.poi_embeddings) != len(self.frame) or self.poi_embeddings.ndim != 2:
            raise ValueError("Precomputed POI embeddings must align one-to-one with the frame")
        self.targets = self.frame["log1p_emission"].to_numpy(dtype=np.float32)
        self.environment_mapping = dict(environment_mapping or {})
        self.environment_key = str(environment_key)
        if self.environment_key not in self.frame:
            raise ValueError(f"Environment column is absent: {self.environment_key}")
        self.environment_ids = np.asarray([
            self.environment_mapping.get(value, -1)
            for value in self.frame[self.environment_key].astype(str)
        ], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def collate(self, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        target_indices = np.asarray(indices, dtype=np.int64)
        neighborhoods = self.neighborhoods[target_indices]
        neighborhood_mask = neighborhoods >= 0
        unique_indices = np.unique(neighborhoods[neighborhood_mask])
        if not len(unique_indices):
            raise ValueError("OpenCarbon batch contains no valid neighborhood nodes")
        local_neighborhoods = np.zeros_like(neighborhoods)
        local_neighborhoods[neighborhood_mask] = np.searchsorted(
            unique_indices, neighborhoods[neighborhood_mask],
        )
        return {
            "poi_embedding": torch.from_numpy(self.poi_embeddings[unique_indices]),
            "remote": torch.from_numpy(self.remote[unique_indices]),
            "environment": torch.from_numpy(self.environment[unique_indices]),
            "neighborhood_indices": torch.from_numpy(local_neighborhoods),
            "neighborhood_mask": torch.from_numpy(neighborhood_mask),
            "target": torch.from_numpy(self.targets[target_indices]),
            "index": torch.from_numpy(target_indices),
            "environment_id": torch.from_numpy(self.environment_ids[target_indices]),
        }


class TabularOpenCarbonDataset(Dataset):
    """OpenCarbon dataset using the 17 panel-level POI aggregate features."""

    def __init__(self, frame, poi, remote, environment, neighborhoods,
                 environment_mapping=None, environment_key="city_id"):
        self.frame = frame.reset_index(drop=True)
        self.poi = poi.astype(np.float32, copy=False)
        self.remote = remote.astype(np.float32, copy=False)
        self.environment = environment.astype(np.float32, copy=False)
        self.neighborhoods = neighborhoods.astype(np.int64, copy=False)
        if self.poi.shape != (len(self.frame), 17):
            raise ValueError("Tabular POI features must have shape [rows, 17]")
        if self.neighborhoods.shape != (len(self.frame), 9):
            raise ValueError("Expected neighborhoods with shape [rows, 9]")
        self.targets = self.frame["log1p_emission"].to_numpy(dtype=np.float32)
        self.environment_mapping = dict(environment_mapping or {})
        self.environment_key = str(environment_key)
        self.environment_ids = np.asarray([
            self.environment_mapping.get(value, -1)
            for value in self.frame[self.environment_key].astype(str)
        ], dtype=np.int64)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return int(index)

    def collate(self, indices):
        target_indices = np.asarray(indices, dtype=np.int64)
        neighborhoods = self.neighborhoods[target_indices]
        mask = neighborhoods >= 0
        unique = np.unique(neighborhoods[mask])
        local = np.zeros_like(neighborhoods)
        local[mask] = np.searchsorted(unique, neighborhoods[mask])
        return {
            "poi": torch.from_numpy(self.poi[unique]),
            "remote": torch.from_numpy(self.remote[unique]),
            "environment": torch.from_numpy(self.environment[unique]),
            "neighborhood_indices": torch.from_numpy(local),
            "neighborhood_mask": torch.from_numpy(mask),
            "target": torch.from_numpy(self.targets[target_indices]),
            "index": torch.from_numpy(target_indices),
            "environment_id": torch.from_numpy(self.environment_ids[target_indices]),
        }
