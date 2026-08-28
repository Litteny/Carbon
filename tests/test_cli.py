from pathlib import Path

import pytest

from carbon_transfer.cli import build_parser, main
from carbon_transfer.config import load_config
from carbon_transfer.experiments import discover_tasks
from carbon_transfer.models.registry import list_model_ids


ROOT = Path(__file__).resolve().parents[1]


def test_models_list_command_outputs_stable_model_ids(capsys):
    assert main(["models", "list"]) == 0
    assert capsys.readouterr().out.strip().splitlines() == list(list_model_ids())


def test_carbon_cli_only_contains_data_manifest_model_and_validation_commands():
    parser = build_parser()
    help_text = parser.format_help()
    assert "{models,data,splits,validate}" in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])


def test_stage1_cross_city_config_only_discovers_cross_city_tasks():
    config = load_config(ROOT / "configs/experiments/stage1_cross_city.yaml")
    tasks = discover_tasks(config)
    assert tasks
    assert {task.experiment for task in tasks} == {"cross_city"}
    assert {task.protocol for task in tasks} == {"stage1"}


def test_task_discovery_can_filter_explicit_experiment():
    config = load_config(ROOT / "configs/experiments/stage1_cross_city.yaml")
    tasks = discover_tasks(
        config,
        models=["opencarbon_monthly"],
        seeds=[42],
        experiments=["cross_city"],
    )
    assert len(tasks) == 4
    assert {task.fold for task in tasks} == {
        "target_chicago", "target_nyc", "target_singapore", "target_tokyo",
    }


def test_unknown_model_has_clear_error():
    config = load_config(ROOT / "configs/experiments/stage1_cross_city.yaml")
    with pytest.raises(ValueError, match="Unknown model"):
        discover_tasks(config, models=["unknown"])


def test_precomputed_open_carbon_config_discovers_standard_models():
    config = load_config(
        ROOT / "configs/experiments/stage1_cross_city_precomputed.yaml"
    )
    tasks = discover_tasks(config)
    assert len(tasks) == 12
    assert {task.model_id for task in tasks} == {
        "opencarbon_core", "opencarbon_monthly",
        "opencarbon_monthly_noviirs",
    }


def test_vrex_config_discovers_only_cross_city_erm_and_vrex_tasks():
    config = load_config(ROOT / "configs/experiments/stage1_cross_city_vrex.yaml")
    tasks = discover_tasks(config)
    assert len(tasks) == 8
    assert {task.experiment for task in tasks} == {"cross_city"}
    assert {task.model_id for task in tasks} == {
        "opencarbon_monthly_precomputed", "opencarbon_monthly_vrex",
    }


def test_smoke_vrex_config_discovers_four_cross_city_tasks():
    config = load_config(ROOT / "configs/experiments/smoke_vrex.yaml")
    tasks = discover_tasks(config)
    assert len(tasks) == 4
    assert {task.model_id for task in tasks} == {"opencarbon_monthly_vrex"}
