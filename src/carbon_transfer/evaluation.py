from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .metrics import calculate_metrics
from .utils import write_json


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No completed runs._"
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number").columns:
        formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in formatted.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def evaluate_run(run_dir: Path) -> Dict:
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    monthly, summary = calculate_metrics(predictions)
    monthly.to_csv(run_dir / "metrics_monthly.csv", index=False)
    write_json(run_dir / "metrics_summary.json", summary)
    return summary


def aggregate_runs(artifact_dir: Path, report_dir: Path) -> Dict:
    records = []
    for metric_file in sorted(artifact_dir.rglob("metrics_monthly.csv")):
        monthly = pd.read_csv(metric_file)
        predictions = pd.read_parquet(metric_file.parent / "predictions.parquet", columns=[
            "experiment", "fold_id", "model", "seed", "city_id", "cell_id",
        ])
        identity = predictions.iloc[0][["experiment", "fold_id", "model", "seed"]].to_dict()
        record = dict(identity)
        record["test_samples"] = len(predictions)
        record["test_grids"] = int(predictions["cell_id"].nunique())
        record["test_cities"] = ",".join(sorted(predictions["city_id"].unique()))
        for column in ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]:
            record[f"{column}_mean"] = monthly[column].mean(skipna=True)
            record[f"{column}_std_months"] = monthly[column].std(skipna=True, ddof=1)
        records.append(record)
    results = pd.DataFrame(records)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "stage1_results.csv", index=False)
    if results.empty:
        summary = {"completed_runs": 0, "single_seed_preliminary": True}
    else:
        cross_city = results[results["experiment"] == "cross_city"]
        cross_city_macro = (
            cross_city.groupby("model")[["log_r2_mean", "log_spearman_mean", "log_mae_mean", "log_rmse_mean"]]
            .mean().reset_index().to_dict("records")
        )
        cross_region = results[results["experiment"] == "cross_region"].copy()
        cross_region_macro = (
            cross_region.groupby("model")[["log_r2_mean", "log_spearman_mean", "log_mae_mean", "log_rmse_mean"]]
            .mean().reset_index().to_dict("records")
        )
        weighted_records = []
        for model, group in cross_region.groupby("model"):
            weights = group["test_grids"].to_numpy(dtype=float)
            record = {"model": model}
            for metric in ["log_r2_mean", "log_spearman_mean", "log_mae_mean", "log_rmse_mean"]:
                values = group[metric].to_numpy(dtype=float)
                valid = pd.notna(values)
                record[metric] = float((values[valid] * weights[valid]).sum() / weights[valid].sum()) if valid.any() else None
            weighted_records.append(record)

        comparison_keys = ["experiment", "fold_id", "seed"]
        monthly = results[results["model"] == "opencarbon_monthly"]
        no_viirs = results[results["model"] == "opencarbon_monthly_noviirs"]
        sensitivity = monthly.merge(no_viirs, on=comparison_keys, suffixes=("_monthly", "_no_viirs"))
        sensitivity_records = []
        for row in sensitivity.itertuples():
            sensitivity_records.append({
                "experiment": row.experiment, "fold_id": row.fold_id, "seed": int(row.seed),
                "log_r2_delta_monthly_minus_no_viirs": row.log_r2_mean_monthly - row.log_r2_mean_no_viirs,
                "log_spearman_delta_monthly_minus_no_viirs": row.log_spearman_mean_monthly - row.log_spearman_mean_no_viirs,
                "log_mae_delta_monthly_minus_no_viirs": row.log_mae_mean_monthly - row.log_mae_mean_no_viirs,
            })
        summary = {
            "completed_runs": len(results), "single_seed_preliminary": True,
            "cross_city_macro": cross_city_macro,
            "cross_region_city_macro": cross_region_macro,
            "cross_region_grid_weighted": weighted_records,
            "viirs_sensitivity": sensitivity_records,
        }
    write_json(report_dir / "stage1_summary.json", summary)
    lines = [
        "# Stage 1 experiment report", "",
        "> Single-seed (`42`) preliminary results; no across-seed variance is reported.", "",
        f"Completed runs: **{len(results)} / 48**", "",
    ]
    if not results.empty:
        display = results[[
            "experiment", "fold_id", "model", "test_grids", "log_r2_mean",
            "log_spearman_mean", "log_mae_mean", "tc_mae_mean",
        ]].copy()
        lines.append(_markdown_table(display))
        lines.extend(["", "## Cross-city macro average", ""])
        lines.append(_markdown_table(pd.DataFrame(summary["cross_city_macro"])))
        lines.extend(["", "## Cross-region grid-weighted pilot average", ""])
        lines.append(_markdown_table(pd.DataFrame(summary["cross_region_grid_weighted"])))
        if summary["viirs_sensitivity"]:
            lines.extend(["", "## VIIRS sensitivity", ""])
            lines.append(_markdown_table(pd.DataFrame(summary["viirs_sensitivity"])))
    (report_dir / "stage1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def aggregate_single_month_runs(artifact_dir: Path, report_dir: Path, expected_runs: int = 4) -> Dict:
    """Aggregate one-month regional-transfer runs without month-variance fields."""
    metric_names = ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]
    records = []
    admin_frames = []
    for metric_file in sorted(artifact_dir.rglob("metrics_monthly.csv")):
        predictions = pd.read_parquet(metric_file.parent / "predictions.parquet")
        if predictions.empty or predictions["experiment"].iloc[0] != "single_month_cross_region":
            continue
        metrics = pd.read_csv(metric_file).iloc[0]
        record = predictions.iloc[0][["experiment", "fold_id", "model", "seed", "city_id"]].to_dict()
        record.update({"test_samples": len(predictions), "test_grids": int(predictions["cell_id"].nunique())})
        record.update({name: metrics[name] for name in metric_names})
        records.append(record)
        admin_file = metric_file.parent / "metrics_by_admin.csv"
        if admin_file.exists():
            admin_frames.append(pd.read_csv(admin_file).assign(
                city_id=record["city_id"], fold_id=record["fold_id"],
                model=record["model"], seed=record["seed"],
            ))
    results = pd.DataFrame(records)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "single_month_results.csv", index=False)
    if admin_frames:
        pd.concat(admin_frames, ignore_index=True).to_csv(
            report_dir / "single_month_admin_results.csv", index=False,
        )

    macro, weighted = {}, {}
    if not results.empty:
        weights = results["test_grids"].to_numpy(dtype=float)
        for name in metric_names:
            values = results[name].to_numpy(dtype=float)
            valid = pd.notna(values)
            macro[name] = float(pd.Series(values).mean(skipna=True))
            weighted[name] = (
                float((values[valid] * weights[valid]).sum() / weights[valid].sum())
                if valid.any() else None
            )
    summary = {
        "experiment": "single_month_cross_region", "period": "202208",
        "completed_runs": int(len(results)), "expected_runs": int(expected_runs),
        "single_seed_preliminary": True, "city_macro": macro, "grid_weighted": weighted,
    }
    write_json(report_dir / "single_month_summary.json", summary)
    lines = [
        "# 2022-08 single-month cross-region experiment", "",
        "> Single-seed (`42`) preliminary results; no across-month or across-seed variance is reported.", "",
        f"Completed runs: **{len(results)} / {expected_runs}**", "",
    ]
    if not results.empty:
        lines.extend([_markdown_table(results), "", "## City macro average", "",
                      _markdown_table(pd.DataFrame([macro])), "",
                      "## Test-grid weighted average", "",
                      _markdown_table(pd.DataFrame([weighted]))])
    (report_dir / "single_month_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def aggregate_single_month_multiseed_runs(
    artifact_dir: Path, report_dir: Path, expected_runs: int = 12,
) -> Dict:
    """Aggregate per-city mean/std across joint split and training seeds."""
    metric_names = ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]
    records = []
    admin_frames = []
    for metric_file in sorted(artifact_dir.rglob("metrics_monthly.csv")):
        predictions = pd.read_parquet(metric_file.parent / "predictions.parquet")
        if predictions.empty or predictions["experiment"].iloc[0] != "single_month_cross_region_multiseed":
            continue
        metrics = pd.read_csv(metric_file).iloc[0]
        record = predictions.iloc[0][["experiment", "fold_id", "model", "seed", "city_id"]].to_dict()
        record.update({"test_samples": len(predictions), "test_grids": predictions["cell_id"].nunique()})
        record.update({name: metrics[name] for name in metric_names})
        records.append(record)
        admin_file = metric_file.parent / "metrics_by_admin.csv"
        if admin_file.exists():
            admin_frames.append(pd.read_csv(admin_file).assign(
                city_id=record["city_id"], fold_id=record["fold_id"],
                model=record["model"], seed=record["seed"],
            ))
    results = pd.DataFrame(records)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "multiseed_results.csv", index=False)
    if admin_frames:
        pd.concat(admin_frames, ignore_index=True).to_csv(
            report_dir / "multiseed_admin_results.csv", index=False,
        )

    city_rows = []
    if not results.empty:
        for city_id, group in results.groupby("city_id", sort=True):
            row = {"city_id": city_id, "runs": len(group), "test_grids_mean": group["test_grids"].mean()}
            for metric in metric_names:
                row[f"{metric}_mean"] = group[metric].mean(skipna=True)
                row[f"{metric}_std_seeds"] = group[metric].std(skipna=True, ddof=1)
            city_rows.append(row)
    city_summary = pd.DataFrame(city_rows)
    city_summary.to_csv(report_dir / "multiseed_city_summary.csv", index=False)

    seed_macro_rows = []
    if not results.empty:
        for seed, group in results.groupby("seed", sort=True):
            row = {"seed": int(seed)}
            for metric in metric_names:
                row[metric] = group[metric].mean(skipna=True)
            seed_macro_rows.append(row)
    seed_macro = pd.DataFrame(seed_macro_rows)
    seed_macro.to_csv(report_dir / "multiseed_seed_macro.csv", index=False)
    overall = {}
    for metric in metric_names:
        values = seed_macro[metric] if not seed_macro.empty else pd.Series(dtype=float)
        overall[metric] = {
            "mean": float(values.mean()) if not values.empty else None,
            "std_across_seeds": float(values.std(ddof=1)) if len(values) > 1 else None,
        }
    summary = {
        "experiment": "single_month_cross_region_multiseed", "period": "202208",
        "completed_runs": len(results), "expected_runs": expected_runs,
        "seeds": sorted(results["seed"].astype(int).unique().tolist()) if not results.empty else [],
        "overall_city_macro": overall,
    }
    write_json(report_dir / "multiseed_summary.json", summary)
    lines = [
        "# 2022-08 single-month cross-region multi-seed experiment", "",
        f"Completed runs: **{len(results)} / {expected_runs}**", "",
    ]
    if not results.empty:
        lines.extend(["## Per-city mean and standard deviation", "", _markdown_table(city_summary), "",
                      "## Four-city macro average by seed", "", _markdown_table(seed_macro)])
    (report_dir / "multiseed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
