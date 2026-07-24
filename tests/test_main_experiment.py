from pathlib import Path

import pytest

from main_experiment import DEFAULT_MODEL, build_parser, discover_folds, prepare_tasks, resolve_fold, resolve_models


ROOT = Path(__file__).resolve().parents[1]


def test_default_model_and_city_resolution():
    args = build_parser().parse_args(["--city", "chicago"])
    assert args.model == DEFAULT_MODEL
    config, fold, models, seed = prepare_tasks(args)
    assert config["training"]["learning_rate"] == 0.001
    assert fold == "target_chicago"
    assert models == ["opencarbon_monthly"]
    assert seed == int(config["seed"])


def test_fold_resolution_accepts_cross_region_fold():
    folds = discover_folds(ROOT / "data" / "splits")
    assert resolve_fold(None, "chicago_02", folds) == "chicago_02"


def test_models_support_repeated_and_comma_separated_values():
    models = resolve_models(
        "opencarbon_monthly",
        ["bpnn,carbongcn", "bpnn", "opencarbon_monthly_noviirs"],
    )
    assert models == ["bpnn", "carbongcn", "opencarbon_monthly_noviirs"]


@pytest.mark.parametrize(
    "city, fold, message",
    [
        ("beijing", None, "Unknown city 'beijing'"),
        (None, "missing_fold", "Unknown fold 'missing_fold'"),
    ],
)
def test_unknown_targets_have_clear_errors(city, fold, message):
    folds = discover_folds(ROOT / "data" / "splits")
    with pytest.raises(ValueError, match=message):
        resolve_fold(city, fold, folds)


def test_empty_and_unknown_model_lists_have_clear_errors():
    with pytest.raises(ValueError, match="at least one model"):
        resolve_models(DEFAULT_MODEL, [",", " "])
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_models(DEFAULT_MODEL, ["unknown"])


def test_city_and_fold_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--city", "chicago", "--fold", "target_chicago"])
