from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .models import POIEncoder
from .utils import sha256_file, write_json


SCHEMA_VERSION = 1
INDEX_COLUMNS = ["city_id", "period", "cell_id", "embedding_row"]


def sample_key(city_id: object, period: object, cell_id: object) -> Tuple[str, str, str]:
    return str(city_id), str(period), str(cell_id)


def poi_tree_sha256(poi_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(poi_root).rglob("*.npz")):
        digest.update(str(path.relative_to(poi_root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def extract_poi_encoder_state(checkpoint: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("Checkpoint does not contain a model_state mapping")
    prefix = "poi_encoder."
    state = {
        str(key)[len(prefix):]: value
        for key, value in model_state.items()
        if str(key).startswith(prefix)
    }
    if not state:
        raise ValueError("Checkpoint does not contain poi_encoder weights")
    output_weight = state.get("output.weight")
    if not isinstance(output_weight, torch.Tensor) or output_weight.ndim != 2:
        raise ValueError("Checkpoint has an invalid POI output layer")
    return state


def expected_metadata(
    fold_id: str,
    checkpoint_path: Path,
    panel_path: Path,
    manifest_path: Path,
    poi_root: Path,
    representation_dim: int,
) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint_path).resolve()
    panel_path = Path(panel_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    poi_root = Path(poi_root).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "fold_id": str(fold_id),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "panel_path": str(panel_path),
        "panel_sha256": sha256_file(panel_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "poi_root": str(poi_root),
        "poi_tree_sha256": poi_tree_sha256(poi_root),
        "representation_dim": int(representation_dim),
        "dtype": "float32",
        "encoder_compute_dtype": "float32",
    }


def load_embedding_metadata(cache_dir: Path) -> Dict[str, object]:
    path = Path(cache_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"POI embedding metadata is absent: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"POI embedding metadata must be a mapping: {path}")
    return value


def validate_embedding_cache(
    cache_dir: Path,
    expected: Mapping[str, object],
) -> Dict[str, object]:
    cache_dir = Path(cache_dir)
    if (cache_dir / ".BUILDING").exists():
        raise RuntimeError(f"POI embedding cache is currently being built: {cache_dir}")
    metadata = load_embedding_metadata(cache_dir)
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"POI embedding cache {key}={metadata.get(key)!r} does not match "
                f"expected {expected_value!r}"
            )
    index_path = cache_dir / "index.parquet"
    embedding_path = cache_dir / "embeddings.npy"
    if not index_path.exists() or not embedding_path.exists():
        raise FileNotFoundError(f"POI embedding cache is incomplete: {cache_dir}")
    index = pd.read_parquet(index_path, columns=INDEX_COLUMNS)
    embeddings = np.load(embedding_path, mmap_mode="r")
    expected_shape = (
        int(metadata.get("samples", -1)),
        int(metadata["representation_dim"]),
    )
    if embeddings.shape != expected_shape or embeddings.dtype != np.float32:
        raise ValueError(
            f"Invalid POI embedding array: shape={embeddings.shape}, dtype={embeddings.dtype}; "
            f"expected shape={expected_shape}, dtype=float32"
        )
    if len(index) != expected_shape[0] or index.duplicated(INDEX_COLUMNS[:3]).any():
        raise ValueError("POI embedding index is incomplete or contains duplicate sample keys")
    if index["embedding_row"].to_numpy(dtype=np.int64).tolist() != list(range(len(index))):
        raise ValueError("POI embedding index rows are not contiguous")
    return metadata


class POIEmbeddingStore:
    def __init__(
        self,
        cache_dir: Path,
        expected: Mapping[str, object],
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.metadata = validate_embedding_cache(self.cache_dir, expected)
        self.index = pd.read_parquet(
            self.cache_dir / "index.parquet", columns=INDEX_COLUMNS,
        )
        self.embeddings = np.load(self.cache_dir / "embeddings.npy", mmap_mode="r")
        self.lookup = {
            sample_key(row.city_id, row.period, row.cell_id): int(row.embedding_row)
            for row in self.index.itertuples(index=False)
        }

    def rows_for_frame(self, frame: pd.DataFrame) -> np.ndarray:
        rows = np.asarray([
            self.lookup.get(sample_key(row.city_id, row.period, row.cell_id), -1)
            for row in frame.itertuples()
        ], dtype=np.int64)
        if np.any(rows < 0):
            raise KeyError(f"POI embedding cache misses {int(np.count_nonzero(rows < 0))} samples")
        return rows

    def take(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.embeddings[rows], dtype=np.float32)


def build_fold_embeddings(
    fold_id: str,
    panel_path: Path,
    manifest_path: Path,
    poi_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    representation_dim: int,
    device: torch.device,
    batch_size: int,
    force: bool = False,
    work_dir: Optional[Path] = None,
) -> Dict[str, object]:
    from .datasets import SparsePOIStore

    panel_path = Path(panel_path)
    manifest_path = Path(manifest_path)
    poi_root = Path(poi_root)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    expected = expected_metadata(
        fold_id, checkpoint_path, panel_path, manifest_path, poi_root, representation_dim,
    )
    if output_dir.exists() and not force:
        return validate_embedding_cache(output_dir, expected)

    panel = pd.read_parquet(panel_path, columns=["city_id", "period", "cell_id"])
    panel["period"] = panel["period"].astype(str)
    panel = panel.reset_index(drop=True)
    if panel.duplicated(["city_id", "period", "cell_id"]).any():
        raise ValueError("Panel contains duplicate POI embedding keys")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder = POIEncoder(17, representation_dim, use_se=True)
    encoder.load_state_dict(extract_poi_encoder_state(checkpoint), strict=True)
    encoder.to(device).eval()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = output_dir.parent / f".{output_dir.name}.lock"
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(f"POI embedding fold is already being built: {fold_id}") from error
    building_marker = output_dir / ".BUILDING"
    building_marker.parent.mkdir(parents=True, exist_ok=True)
    building_marker.write_text("building\n", encoding="utf-8")
    work_root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir()) / "carbon_poi_embeddings"
    work_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=work_root))
    try:
        embeddings = np.lib.format.open_memmap(
            temporary / "embeddings.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(panel), representation_dim),
        )
        store = SparsePOIStore(poi_root, cache_size=8)
        with torch.no_grad():
            for start in range(0, len(panel), int(batch_size)):
                stop = min(start + int(batch_size), len(panel))
                dense = np.stack([
                    store.dense(row.city_id, str(row.period), row.cell_id)
                    for row in panel.iloc[start:stop].itertuples()
                ])
                inputs = torch.from_numpy(dense).to(device, non_blocking=True)
                values = encoder(inputs)
                embeddings[start:stop] = values.float().cpu().numpy()
        embeddings.flush()
        index = panel.copy()
        index["embedding_row"] = np.arange(len(index), dtype=np.int64)
        index.to_parquet(temporary / "index.parquet", index=False)
        metadata = {
            **expected,
            "samples": int(len(index)),
            "cities": sorted(index["city_id"].astype(str).unique().tolist()),
            "periods": sorted(index["period"].astype(str).unique().tolist()),
        }
        write_json(temporary / "metadata.json", metadata)
        validate_embedding_cache(temporary, expected)
        # Publish into a remote temporary directory, then validate the remote copy
        # before replacing the previous cache.  This avoids exposing half-written
        # files on the Ceph-backed output directory.
        remote_temporary = Path(tempfile.mkdtemp(
            prefix=f".{output_dir.name}.publish.", dir=output_dir.parent,
        ))
        try:
            for filename in ("embeddings.npy", "index.parquet", "metadata.json"):
                source = temporary / filename
                destination = remote_temporary / filename
                with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
            validate_embedding_cache(remote_temporary, expected)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            os.replace(remote_temporary, output_dir)
        except BaseException:
            shutil.rmtree(remote_temporary, ignore_errors=True)
            raise
        building_marker.unlink(missing_ok=True)
        shutil.rmtree(lock_dir, ignore_errors=True)
        shutil.rmtree(temporary, ignore_errors=True)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        building_marker.unlink(missing_ok=True)
        shutil.rmtree(lock_dir, ignore_errors=True)
        raise
