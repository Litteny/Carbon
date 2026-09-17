from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from carbon_transfer.experiments import ExperimentTask


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_name", "fold", "city"),
    [
        ("run_stage1_cross_city", "target_nyc", "nyc"),
        ("run_stage1_cross_region", "chicago_02", "chicago"),
        ("run_single_month_cross_region", "202208_singapore", "singapore"),
        (
            "run_single_month_cross_region_multiseed",
            "202208_tokyo_split_seed_43",
            "tokyo",
        ),
        (
            "run_annual_cross_region",
            "2022_nyc_split_seed_44",
            "nyc",
        ),
        (
            "run_three_year_cross_region",
            "2021-2023_tokyo_split_seed_43",
            "tokyo",
        ),
        (
            "run_stage1_within_city_grid",
            "2021-2023_tokyo_grid_fixed",
            "tokyo",
        ),
    ],
)
def test_experiment_scripts_map_folds_to_cities(script_name, fold, city):
    module = load_script(script_name)
    task = ExperimentTask("experiment", "protocol", fold, "bpnn", 42, Path("manifest"))
    assert module._task_city(task) == city


@pytest.mark.parametrize(
    "script_name",
    [
        "run_stage1_cross_city",
        "run_stage1_cross_region",
        "run_single_month_cross_region",
        "run_single_month_cross_region_multiseed",
        "run_annual_cross_region",
        "run_three_year_cross_region",
        "run_stage1_within_city_grid",
    ],
)
def test_experiment_scripts_accept_common_training_overrides(script_name):
    module = load_script(script_name)
    args = module.build_parser().parse_args([
        "--cities", "new_york",
        "--models", "opencarbon_monthly",
        "--seeds", "42", "123",
        "--epochs", "80",
        "--patience", "12",
        "--batch-size", "256",
        "--device", "cuda",
        "--run-name", "trial_01",
        "--dry-run",
    ])
    assert module._selected_cities(args.cities) == {"nyc"}
    assert args.epochs == 80
    assert args.patience == 12
    assert args.batch_size == 256
    assert args.run_name == "trial_01"


def test_stage1_cross_city_dry_run_filters_alias_model_and_seed(capsys):
    module = load_script("run_stage1_cross_city")
    assert module.main([
        "--cities", "new_york",
        "--models", "opencarbon_monthly",
        "--seeds", "42",
        "--dry-run",
    ]) == 0
    output = capsys.readouterr().out
    assert "city=nyc" in output
    assert "fold=target_nyc" in output
    assert "model=opencarbon_monthly seed=42" in output
    assert "tasks=1" in output


def test_training_counts_must_be_positive():
    module = load_script("run_stage1_cross_city")
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--epochs", "0"])
