import pandas as pd

from carbon_transfer.r2_diagnostics import (
    _stage_summary,
    calculate_r2_diagnostics,
    diagnose_r2_stages,
    grouped_r2_diagnostics,
)


def _frame(split, observed, predicted, city="a", period="202201"):
    return pd.DataFrame({
        "split": split,
        "city_id": city,
        "period": period,
        "cell_id": [f"c{i}" for i in range(len(observed))],
        "y_log": observed,
        "pred_log": predicted,
    })


def test_diagnostics_identify_test_as_first_negative_stage():
    predictions = pd.concat([
        _frame("train", [0, 1, 2], [0, 1, 2]),
        _frame("validation", [0, 1, 2], [0.1, 1.0, 1.9]),
        _frame("test", [0, 1, 2], [3, 3, 3]),
    ], ignore_index=True)
    diagnosis = diagnose_r2_stages(_stage_summary(predictions))
    assert diagnosis["pooled_first_negative_split"] == "test"
    assert diagnosis["monthly_mean_first_negative_split"] == "test"


def test_debiased_r2_detects_pure_mean_shift():
    frame = _frame("test", [0, 1, 2, 3], [10, 11, 12, 13])
    result = calculate_r2_diagnostics(frame)
    assert result["r2"] < 0
    assert result["debiased_r2"] == 1.0


def test_monthly_debiasing_identifies_mean_shift_dominated_test_failure():
    predictions = pd.concat([
        _frame("train", [0, 1, 2], [0, 1, 2]),
        _frame("validation", [0, 1, 2], [0, 1, 2]),
        _frame("test", [0, 1, 2], [10, 11, 12], period="202201"),
        _frame("test", [0, 1, 2], [-10, -9, -8], period="202202"),
    ], ignore_index=True)
    diagnosis = diagnose_r2_stages(_stage_summary(predictions))
    assert diagnosis["monthly_mean_first_negative_split"] == "test"
    assert diagnosis["test_monthly_bias_diagnosis"] == (
        "monthly_mean_level_shift_dominant"
    )


def test_constant_targets_are_undefined_not_negative():
    result = calculate_r2_diagnostics(
        _frame("test", [1, 1, 1], [0, 1, 2]),
    )
    assert result["r2"] is None
    assert result["r2_status"].startswith("undefined")


def test_grouped_diagnostics_preserve_city_and_month_keys():
    predictions = pd.concat([
        _frame("train", [0, 1, 2], [0, 1, 2], city="a", period="202201"),
        _frame("train", [0, 1, 2], [0, 1, 2], city="b", period="202202"),
    ], ignore_index=True)
    grouped = grouped_r2_diagnostics(
        predictions, ["split", "city_id", "period"],
    )
    assert list(grouped[["city_id", "period"]].itertuples(index=False, name=None)) == [
        ("a", "202201"), ("b", "202202"),
    ]
