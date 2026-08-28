import json
from pathlib import Path

import pandas as pd

from carbon_transfer.experiments.aggregation_comparison import compare_annual_aggregations


def _write_run(path: Path, offset: float) -> None:
    path.mkdir(parents=True)
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")
    monthly = pd.DataFrame({
        "period": ["202201", "202202"],
        "log_r2": [0.1 + offset, 0.2 + offset],
        "log_mae": [0.5 - offset, 0.4 - offset],
        "log_rmse": [0.6 - offset, 0.5 - offset],
        "log_spearman": [0.3 + offset, 0.4 + offset],
        "tc_mae": [10.0 - offset, 9.0 - offset],
        "tc_rmse": [12.0 - offset, 11.0 - offset],
    })
    monthly.to_csv(path / "metrics_monthly.csv", index=False)
    summary = {"metrics": {
        metric: {"mean": float(monthly[metric].mean())}
        for metric in ("log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse")
    }}
    (path / "metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame({
        "admin_id": ["a"],
        **{f"{metric}_mean": [float(monthly[metric].mean())] for metric in summary["metrics"]},
    }).to_csv(path / "metrics_by_admin_annual.csv", index=False)


def test_comparison_uses_positive_improvement_for_both_metric_directions(tmp_path):
    baseline, candidate, report = tmp_path / "baseline", tmp_path / "candidate", tmp_path / "report"
    _write_run(baseline, 0.0)
    _write_run(candidate, 0.1)
    result = compare_annual_aggregations(baseline, candidate, report)
    assert result["periods_compared"] == 2
    assert result["admins_compared"] == 1
    assert all(entry["improvement"] > 0 for entry in result["metrics"].values())
    assert all(entry["winner"] == "mean_mlp_gate" for entry in result["metrics"].values())
    assert (report / "aggregation_comparison_monthly.csv").exists()
    assert (report / "aggregation_comparison_admin.csv").exists()
