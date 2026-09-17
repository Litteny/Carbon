from pathlib import Path

from carbon_transfer.config import load_config
from carbon_transfer.constants import (
    STAGE1_B0_MODEL,
    STAGE1_M1_MODEL,
    STAGE1_FEATURE_COLUMNS,
    VIIRS_COLUMNS,
)
from carbon_transfer.models.registry import list_model_ids
from carbon_transfer.within_city_grid import _split_grid_ids


ROOT = Path(__file__).resolve().parents[1]


def test_chicago_singapore_supplement_config_is_isolated():
    config = load_config(
        ROOT / "configs/experiments/stage1_within_city_grid_chicago_singapore.yaml"
    )

    assert config["cities"] == ["chicago", "singapore"]
    assert config["split_experiment"] == config["experiment"]
    assert config["artifact_dir"].endswith(
        "stage1_within_city_grid_chicago_singapore_2021_2023/runs"
    )
    assert config["report_dir"].endswith(
        "stage1_within_city_grid_chicago_singapore_2021_2023/reports"
    )


def test_stage1_models_are_registered_and_use_fixed_feature_schema():
    assert STAGE1_B0_MODEL in set(list_model_ids())
    assert STAGE1_M1_MODEL in set(list_model_ids())
    assert STAGE1_FEATURE_COLUMNS[:len(VIIRS_COLUMNS)] == VIIRS_COLUMNS


def test_within_city_grid_assignment_is_deterministic_and_disjoint():
    cells = [f"cell_{index:02d}" for index in range(20)]
    first = _split_grid_ids(cells, [0.7, 0.2, 0.1], seed=2026)
    second = _split_grid_ids(cells, [0.7, 0.2, 0.1], seed=2026)

    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(cells)


def test_within_city_grid_assignment_rejects_invalid_ratios():
    cells = ["a", "b", "c"]
    try:
        _split_grid_ids(cells, [0.7, 0.2, 0.2], seed=1)
    except ValueError as error:
        assert "sum to 1" in str(error)
    else:
        raise AssertionError("invalid split ratios were accepted")
