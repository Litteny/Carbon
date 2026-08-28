from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from carbon_transfer.datasets import PrecomputedOpenCarbonDataset
from carbon_transfer.models import OpenCarbonModel
from carbon_transfer.poi_embeddings import (
    POIEmbeddingStore,
    extract_poi_encoder_state,
    validate_embedding_cache,
)
from carbon_transfer.training import (
    _open_carbon_poi_input_mode,
    _validate_precomputed_checkpoint,
)


def test_extract_poi_encoder_state_strips_prefix_and_rejects_missing_weights():
    checkpoint = {
        "model_state": {
            "poi_encoder.output.weight": torch.zeros(4, 3),
            "poi_encoder.output.bias": torch.zeros(4),
            "regressor.weight": torch.ones(1, 1),
        },
    }
    state = extract_poi_encoder_state(checkpoint)
    assert set(state) == {"output.weight", "output.bias"}
    with pytest.raises(ValueError, match="poi_encoder"):
        extract_poi_encoder_state({"model_state": {"regressor.weight": torch.ones(1)}})


def _write_cache(root: Path) -> dict:
    index = pd.DataFrame({
        "city_id": ["a", "b"],
        "period": ["202101", "202101"],
        "cell_id": ["x", "y"],
        "embedding_row": [0, 1],
    })
    index.to_parquet(root / "index.parquet", index=False)
    np.save(root / "embeddings.npy", np.arange(8, dtype=np.float32).reshape(2, 4))
    metadata = {
        "schema_version": 1,
        "fold_id": "target_a",
        "checkpoint_sha256": "checkpoint",
        "representation_dim": 4,
        "dtype": "float32",
        "samples": 2,
    }
    (root / "metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")
    return metadata


def test_embedding_store_maps_sample_keys_and_validates_provenance(tmp_path):
    metadata = _write_cache(tmp_path)
    store = POIEmbeddingStore(tmp_path, metadata)
    frame = pd.DataFrame({
        "city_id": ["b", "a"], "period": ["202101", "202101"], "cell_id": ["y", "x"],
    })
    assert store.rows_for_frame(frame).tolist() == [1, 0]
    assert store.take(np.asarray([1])).tolist() == [[4.0, 5.0, 6.0, 7.0]]
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_embedding_cache(tmp_path, {**metadata, "checkpoint_sha256": "changed"})


def test_precomputed_dataset_collate_returns_embeddings_without_poi_store():
    frame = pd.DataFrame({
        "city_id": ["a", "a"], "log1p_emission": [0.0, 1.0],
    })
    dataset = PrecomputedOpenCarbonDataset(
        frame,
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
        np.asarray([[0] * 9, [1] * 9], dtype=np.int64),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    batch = dataset.collate([0, 1])
    assert "poi" not in batch
    assert batch["poi_embedding"].shape == (2, 2)
    assert not hasattr(dataset, "poi_store")


def test_online_and_precomputed_forward_paths_are_identical():
    model = OpenCarbonModel(3, 2, representation_dim=8, dropout=0.0).eval()
    poi = torch.randn(2, 17, 256, 256)
    remote = torch.randn(2, 3)
    environment = torch.randn(2, 2)
    indices = torch.tensor([[0, 0, 0, 0, 0, 1, 0, 1, 0]])
    mask = torch.tensor([[False, False, False, False, True, True, False, True, False]])
    with torch.no_grad():
        online = model(poi, remote, environment, indices, mask)
        cached = model.forward_encoded(
            model.poi_encoder(poi), remote, environment, indices, mask,
        )
    for first, second in zip(online, cached):
        assert torch.equal(first, second)


def test_old_precomputed_checkpoint_is_rejected():
    metadata = {"checkpoint_sha256": "encoder", "fold_id": "target_a"}
    with pytest.raises(ValueError, match="poi_input_mode"):
        _validate_precomputed_checkpoint({}, metadata)
