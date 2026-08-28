import pytest

from carbon_transfer.config import validate_experiment_config
from carbon_transfer.training import _open_carbon_poi_input_mode


@pytest.mark.parametrize(
    "model_id",
    ["opencarbon_core", "opencarbon_monthly", "opencarbon_monthly_noviirs"],
)
def test_standard_open_carbon_models_can_use_precomputed_poi(model_id):
    assert _open_carbon_poi_input_mode(
        model_id, {"poi_input_mode": "precomputed"},
    ) == "precomputed"


def test_legacy_precomputed_models_remain_precomputed():
    assert _open_carbon_poi_input_mode(
        "opencarbon_monthly_vrex", {"poi_input_mode": "dense"},
    ) == "precomputed"
    with pytest.raises(ValueError, match="poi_input_mode"):
        _open_carbon_poi_input_mode(
            "opencarbon_monthly", {"poi_input_mode": "unknown"},
        )


def _config():
    return {
        "panel_file": "panel.parquet",
        "neighbors_file": "neighbors.parquet",
        "poi_dir": "poi",
        "split_dir": "splits",
        "artifact_dir": "outputs",
        "models": ["opencarbon_monthly"],
    }


def test_precomputed_config_requires_cache_and_checkpoint_template():
    config = _config()
    config["training"] = {"poi_input_mode": "precomputed"}
    with pytest.raises(ValueError, match="poi_embedding_dir"):
        validate_experiment_config(config)
    config.update({
        "poi_embedding_dir": "embeddings",
        "poi_embedding_checkpoint_template": "runs/{fold}/seed_{seed}/best.pt",
    })
    validated = validate_experiment_config(config)
    assert validated.training.poi_input_mode == "precomputed"


def test_dense_poi_mode_remains_default():
    assert validate_experiment_config(_config()).training.poi_input_mode == "dense"
