from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from carbon_transfer.utils import write_json


METRICS = ("log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse")
HIGHER_IS_BETTER = {"log_r2", "log_spearman"}


def _require_run(run_dir: Path) -> None:
    required = (
        "COMPLETE", "metrics_summary.json", "metrics_monthly.csv",
        "metrics_by_admin_annual.csv",
    )
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete comparison run {run_dir}: missing {', '.join(missing)}")


def _improvement(metric: str, baseline: pd.Series, candidate: pd.Series) -> pd.Series:
    if metric in HIGHER_IS_BETTER:
        return candidate - baseline
    return baseline - candidate


def compare_annual_aggregations(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    report_dir: Path,
) -> Dict:
    """Compare one annual run while treating positive improvement as better."""
    baseline_run_dir = Path(baseline_run_dir)
    candidate_run_dir = Path(candidate_run_dir)
    report_dir = Path(report_dir)
    _require_run(baseline_run_dir)
    _require_run(candidate_run_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(baseline_run_dir / "metrics_monthly.csv").merge(
        pd.read_csv(candidate_run_dir / "metrics_monthly.csv"),
        on="period", suffixes=("_spatial_attention", "_mean_mlp_gate"),
        validate="one_to_one",
    )
    for metric in METRICS:
        monthly[f"{metric}_improvement"] = _improvement(
            metric,
            monthly[f"{metric}_spatial_attention"],
            monthly[f"{metric}_mean_mlp_gate"],
        )
    monthly.to_csv(report_dir / "aggregation_comparison_monthly.csv", index=False)

    with (baseline_run_dir / "metrics_summary.json").open(encoding="utf-8") as handle:
        baseline_summary = json.load(handle)
    with (candidate_run_dir / "metrics_summary.json").open(encoding="utf-8") as handle:
        candidate_summary = json.load(handle)
    overall_rows = []
    for metric in METRICS:
        baseline = float(baseline_summary["metrics"][metric]["mean"])
        candidate = float(candidate_summary["metrics"][metric]["mean"])
        improvement = candidate - baseline if metric in HIGHER_IS_BETTER else baseline - candidate
        overall_rows.append({
            "metric": metric,
            "spatial_attention": baseline,
            "mean_mlp_gate": candidate,
            "improvement": improvement,
            "winner": "mean_mlp_gate" if improvement > 0 else (
                "spatial_attention" if improvement < 0 else "tie"
            ),
        })
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(report_dir / "aggregation_comparison_overall.csv", index=False)

    admin = pd.read_csv(baseline_run_dir / "metrics_by_admin_annual.csv").merge(
        pd.read_csv(candidate_run_dir / "metrics_by_admin_annual.csv"),
        on="admin_id", suffixes=("_spatial_attention", "_mean_mlp_gate"),
        validate="one_to_one",
    )
    for metric in METRICS:
        column = f"{metric}_mean"
        admin[f"{metric}_improvement"] = _improvement(
            metric,
            admin[f"{column}_spatial_attention"],
            admin[f"{column}_mean_mlp_gate"],
        )
    admin.to_csv(report_dir / "aggregation_comparison_admin.csv", index=False)

    result = {
        "baseline_aggregation": "spatial_attention",
        "candidate_aggregation": "mean_mlp_gate",
        "baseline_run_dir": str(baseline_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "periods_compared": int(len(monthly)),
        "admins_compared": int(len(admin)),
        "metrics": {
            row["metric"]: {
                "spatial_attention": float(row["spatial_attention"]),
                "mean_mlp_gate": float(row["mean_mlp_gate"]),
                "improvement": float(row["improvement"]),
                "winner": row["winner"],
            }
            for row in overall_rows
        },
    }
    write_json(report_dir / "aggregation_comparison_summary.json", result)
    return result
