from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

from carbon_transfer.config import load_config
from carbon_transfer.experiments import discover_tasks
from carbon_transfer.splits import _build_fixed_cross_city_split


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "cross_city_chicago_to_tokyo"
FOLD = "chicago_train_nyc_singapore_validation_tokyo_test"


def _definition():
    return {
        "experiment": EXPERIMENT,
        "fold_id": FOLD,
        "train_cities": ["chicago"],
        "validation_cities": ["nyc", "singapore"],
        "test_cities": ["tokyo"],
    }


def _panel():
    rows = []
    for city in ("chicago", "nyc", "singapore", "tokyo"):
        rows.append({
            "city_id": city,
            "cell_id": f"{city}_cell",
            "period": "202201",
            "admin_id": f"{city}_admin",
            "admin_name": city,
        })
    return pd.DataFrame(rows)


def test_fixed_cross_city_assigns_complete_cities(tmp_path):
    entry = _build_fixed_cross_city_split(_panel(), tmp_path, _definition())
    manifest = pd.read_parquet(entry["file"])
    assignments = manifest.set_index("city_id")["split"].to_dict()
    assert assignments == {
        "chicago": "train",
        "nyc": "validation",
        "singapore": "validation",
        "tokyo": "test",
    }
    assert entry["city_roles"] == {
        "train": ["chicago"],
        "validation": ["nyc", "singapore"],
        "test": ["tokyo"],
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("test_cities", ["missing"], "absent from the panel"),
        ("test_cities", ["chicago"], "must be disjoint"),
        ("validation_cities", ["nyc"], "leaves cities unassigned"),
        ("train_cities", [], "must not be empty"),
    ],
)
def test_fixed_cross_city_rejects_invalid_city_roles(tmp_path, key, value, message):
    definition = _definition()
    definition[key] = value
    with pytest.raises(ValueError, match=message):
        _build_fixed_cross_city_split(_panel(), tmp_path, definition)


def test_real_fixed_cross_city_manifest_has_expected_complete_city_roles():
    path = ROOT / "data/splits" / EXPERIMENT / f"{FOLD}.parquet"
    frame = pd.read_parquet(path)
    assert frame.groupby("split").size().to_dict() == {
        "test": 32580,
        "train": 33840,
        "validation": 76572,
    }
    assert frame.groupby("city_id")["split"].unique().map(list).to_dict() == {
        "chicago": ["train"],
        "nyc": ["validation"],
        "singapore": ["validation"],
        "tokyo": ["test"],
    }
    assert not frame.duplicated(["city_id", "cell_id", "period"]).any()
    assert frame.groupby(["city_id", "cell_id"])["split"].nunique().max() == 1


@pytest.mark.parametrize(
    ("config_name", "models"),
    [
        (
            "cross_city_chicago_to_tokyo_opencarbon.yaml",
            {"opencarbon_core", "opencarbon_monthly", "opencarbon_monthly_noviirs"},
        ),
        (
            "cross_city_chicago_to_tokyo_precomputed.yaml",
            {"opencarbon_core", "opencarbon_monthly", "opencarbon_monthly_noviirs"},
        ),
        (
            "cross_city_chicago_to_tokyo_vrex.yaml",
            {"opencarbon_monthly_precomputed", "opencarbon_monthly_vrex"},
        ),
    ],
)
def test_fixed_cross_city_configs_discover_one_fold(config_name, models):
    config = load_config(ROOT / "configs/experiments" / config_name)
    tasks = discover_tasks(config)
    assert {task.experiment for task in tasks} == {EXPERIMENT}
    assert {task.fold for task in tasks} == {FOLD}
    assert {task.model_id for task in tasks} == models
    assert len(tasks) == len(models)


def test_fixed_cross_city_vrex_uses_shared_month_environments_and_valid_batch():
    config = load_config(
        ROOT / "configs/experiments/cross_city_chicago_to_tokyo_vrex.yaml"
    )
    assert config["training"]["environment_key"] == "period"
    assert config["training"]["batch_size"] >= 36
    assert config["training"]["learning_rate"] == 0.0003
    assert config["training"]["record_train_r2"] is True
    tasks = discover_tasks(config)
    assert {task.model_id for task in tasks} == {
        "opencarbon_monthly_precomputed",
        "opencarbon_monthly_vrex",
    }


def test_fixed_cross_city_script_dry_run(capsys):
    path = ROOT / "scripts/run_cross_city_chicago_to_tokyo.py"
    spec = spec_from_file_location("run_cross_city_chicago_to_tokyo", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--models", "opencarbon_monthly", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "train=chicago validation=nyc,singapore test=tokyo" in output
    assert f"fold={FOLD}" in output
    assert "tasks=1" in output


def test_fixed_cross_city_script_rejects_non_opencarbon_model():
    path = ROOT / "scripts/run_cross_city_chicago_to_tokyo.py"
    spec = spec_from_file_location("run_cross_city_chicago_to_tokyo_invalid", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.main(["--models", "bpnn", "--dry-run"])


def test_multisource_tokyo_manifests_have_three_roles_and_fixed_test_city():
    split_root = ROOT / "data/splits/cross_city_multisource_tokyo"
    manifests = sorted(split_root.glob("*.parquet"))
    assert {path.stem for path in manifests} == {
        "chicago_nyc_train_singapore_validation_tokyo_test",
        "chicago_singapore_train_nyc_validation_tokyo_test",
        "nyc_singapore_train_chicago_validation_tokyo_test",
    }
    expected = {
        "chicago_nyc_train_singapore_validation_tokyo_test": {"train": 77076, "validation": 33336, "test": 32580},
        "chicago_singapore_train_nyc_validation_tokyo_test": {"train": 67176, "validation": 43236, "test": 32580},
        "nyc_singapore_train_chicago_validation_tokyo_test": {"train": 76572, "validation": 33840, "test": 32580},
    }
    for path in manifests:
        frame = pd.read_parquet(path)
        assert frame.groupby("split").size().to_dict() == expected[path.stem]
        assert set(frame.loc[frame["city_id"] == "tokyo", "split"]) == {"test"}
        assert not frame.duplicated(["city_id", "cell_id", "period"]).any()


@pytest.mark.parametrize(
    ("config_name", "expected_models", "expected_tasks"),
    [
        (
            "cross_city_multisource_tokyo_opencarbon.yaml",
            {"opencarbon_monthly"},
            3,
        ),
        (
            "cross_city_multisource_tokyo_vrex.yaml",
            {"opencarbon_monthly_precomputed", "opencarbon_monthly_vrex"},
            6,
        ),
    ],
)
def test_multisource_tokyo_configs_discover_expected_tasks(
    config_name, expected_models, expected_tasks,
):
    config = load_config(ROOT / "configs/experiments" / config_name)
    tasks = discover_tasks(config)
    assert len(tasks) == expected_tasks
    assert {task.model_id for task in tasks} == expected_models
    assert {task.experiment for task in tasks} == {"cross_city_multisource_tokyo"}


def test_multisource_dense_config_disables_train_r2_recording():
    config = load_config(
        ROOT / "configs/experiments/cross_city_multisource_tokyo_opencarbon.yaml"
    )
    assert config["training"]["record_train_r2"] is False
