from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import calculate_metrics, calculate_pooled_metrics
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
    if (
        not predictions.empty
        and str(predictions["experiment"].iloc[0]).startswith("stage1_within_city_grid")
    ):
        calculate_pooled_metrics(predictions).to_csv(
            run_dir / "metrics_pooled.csv", index=False,
        )
    write_json(run_dir / "metrics_summary.json", summary)
    return summary


def aggregate_runs(
    artifact_dir: Path,
    report_dir: Path,
    experiments: Optional[Sequence[str]] = None,
) -> Dict:
    selected_experiments = {str(value) for value in experiments} if experiments else None
    records = []
    for metric_file in sorted(artifact_dir.rglob("metrics_monthly.csv")):
        monthly = pd.read_csv(metric_file)
        predictions = pd.read_parquet(metric_file.parent / "predictions.parquet", columns=[
            "experiment", "fold_id", "model", "seed", "city_id", "cell_id",
        ])
        identity = predictions.iloc[0][["experiment", "fold_id", "model", "seed"]].to_dict()
        if selected_experiments and str(identity["experiment"]) not in selected_experiments:
            continue
        record = dict(identity)
        record["test_samples"] = len(predictions)
        record["test_grids"] = int(predictions["cell_id"].nunique())
        record["test_cities"] = ",".join(sorted(predictions["city_id"].unique()))
        for column in ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]:
            record[f"{column}_mean"] = monthly[column].mean(skipna=True)
            record[f"{column}_std_months"] = monthly[column].std(skipna=True, ddof=1)
        environment_file = metric_file.parent / "environment_metrics.csv"
        if environment_file.exists():
            environment_metrics = pd.read_csv(environment_file)
            if not environment_metrics.empty:
                for column in (
                    "environment_macro_mae",
                    "worst_environment_mae",
                    "environment_mae_std",
                ):
                    record[column] = environment_metrics[column].iloc[0]
        records.append(record)
    results = pd.DataFrame(records)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "stage1_results.csv", index=False)
    if results.empty:
        summary = {"completed_runs": 0, "single_seed_preliminary": True}
    else:
        cross_city = results[
            results["experiment"].astype(str).str.startswith("cross_city")
        ]
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
            "environment_robustness": results.loc[
                results["environment_macro_mae"].notna(),
                [
                    "experiment", "fold_id", "model", "seed",
                    "environment_macro_mae", "worst_environment_mae",
                    "environment_mae_std",
                ],
            ].to_dict("records") if "environment_macro_mae" in results else [],
        }
    write_json(report_dir / "stage1_summary.json", summary)
    lines = [
        "# Stage 1 experiment report", "",
        "> Single-seed (`42`) preliminary results; no across-seed variance is reported.", "",
        f"Completed runs: **{len(results)}**", "",
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
        if summary["environment_robustness"]:
            lines.extend(["", "## Source-environment robustness", ""])
            lines.append(_markdown_table(pd.DataFrame(summary["environment_robustness"])))
    (report_dir / "stage1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def aggregate_stage1_within_city_grid_runs(
    artifact_dir: Path, report_dir: Path, expected_runs: int = 0,
) -> Dict:
    """Aggregate B0/M1 paired results for the within-city grid protocol."""
    metric_names = ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]
    records = []
    prediction_frames = []
    for prediction_file in sorted(artifact_dir.rglob("predictions.parquet")):
        predictions = pd.read_parquet(prediction_file)
        if predictions.empty:
            continue
        experiment = str(predictions["experiment"].iloc[0])
        if not experiment.startswith("stage1_within_city_grid"):
            continue
        metrics_file = prediction_file.parent / "metrics_monthly.csv"
        if not metrics_file.exists():
            continue
        monthly = pd.read_csv(metrics_file)
        identity = predictions.iloc[0][[
            "experiment", "fold_id", "model", "seed", "city_id",
        ]].to_dict()
        record = dict(identity)
        record.update({
            "test_samples": int(len(predictions)),
            "test_grids": int(predictions["cell_id"].nunique()),
            "months": int(predictions["period"].nunique()),
        })
        if "neighborhood_aggregation" in predictions:
            record["neighborhood_aggregation"] = str(
                predictions["neighborhood_aggregation"].iloc[0],
            )
        for metric in metric_names:
            record[f"{metric}_mean"] = monthly[metric].mean(skipna=True)
            record[f"{metric}_std_months"] = monthly[metric].std(skipna=True, ddof=1)
        pooled_file = prediction_file.parent / "metrics_pooled.csv"
        if pooled_file.exists():
            pooled = pd.read_csv(pooled_file).iloc[0]
            for metric in metric_names:
                record[f"{metric}_pooled"] = pooled[metric]
        records.append(record)
        prediction_frames.append(predictions)

    results = pd.DataFrame(records)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "stage1_runs.csv", index=False)

    paired = pd.DataFrame()
    if prediction_frames:
        all_predictions = pd.concat(prediction_frames, ignore_index=True)
        key = ["experiment", "fold_id", "city_id", "cell_id", "period", "seed"]
        if "neighborhood_aggregation" in all_predictions:
            key.append("neighborhood_aggregation")
        base = all_predictions[all_predictions["model"] == "opencarbon_stage1_b0"].copy()
        candidate = all_predictions[all_predictions["model"] == "opencarbon_stage1_m1"].copy()
        base["b0_abs_log_error"] = (base["pred_log"] - base["y_log"]).abs()
        candidate["m1_abs_log_error"] = (candidate["pred_log"] - candidate["y_log"]).abs()
        paired = base[key + ["b0_abs_log_error"]].merge(
            candidate[key + ["m1_abs_log_error"]], on=key, how="inner", validate="one_to_one",
        )
        paired["delta_log_mae"] = paired["m1_abs_log_error"] - paired["b0_abs_log_error"]
        paired.to_csv(report_dir / "stage1_paired_samples.csv", index=False)
        group_keys = ["experiment", "fold_id", "city_id", "seed"]
        if "neighborhood_aggregation" in paired:
            group_keys.append("neighborhood_aggregation")
        paired_summary = paired.groupby(group_keys, sort=True).agg(
            paired_samples=("delta_log_mae", "size"),
            b0_log_mae=("b0_abs_log_error", "mean"),
            m1_log_mae=("m1_abs_log_error", "mean"),
            delta_log_mae=("delta_log_mae", "mean"),
            m1_improved_fraction=("delta_log_mae", lambda values: float((values < 0).mean())),
        ).reset_index()
        grid_deltas = paired.groupby(group_keys + ["cell_id"], sort=True)[
            "delta_log_mae"
        ].mean().reset_index()
        bootstrap = []
        generator = np.random.default_rng(2026)
        for group_identity, group in grid_deltas.groupby(group_keys, sort=True):
            values = group["delta_log_mae"].to_numpy(dtype=float)
            if len(values) < 2:
                low = high = float(values[0]) if len(values) else np.nan
            else:
                indices = generator.integers(0, len(values), size=(1000, len(values)))
                means = values[indices].mean(axis=1)
                low, high = np.quantile(means, [0.025, 0.975]).tolist()
            identity_values = (
                group_identity if isinstance(group_identity, tuple) else (group_identity,)
            )
            identity = dict(zip(group_keys, identity_values))
            identity.update({
                "grid_count": int(len(values)),
                "delta_log_mae_ci_low": float(low),
                "delta_log_mae_ci_high": float(high),
            })
            bootstrap.append(identity)
        paired_summary = paired_summary.merge(
            pd.DataFrame(bootstrap), on=group_keys, how="left", validate="one_to_one",
        )
    else:
        paired_summary = pd.DataFrame()
        pd.DataFrame().to_csv(report_dir / "stage1_paired_samples.csv", index=False)
    paired_summary.to_csv(report_dir / "stage1_paired_comparison.csv", index=False)

    seed_summary = pd.DataFrame()
    city_summary = pd.DataFrame()
    city_grid_weighted = pd.DataFrame()
    seed_city_macro = pd.DataFrame()
    if not paired_summary.empty:
        group_keys = ["city_id", "seed"]
        if "neighborhood_aggregation" in paired_summary:
            group_keys.append("neighborhood_aggregation")
        seed_summary = paired_summary.groupby(group_keys, sort=True)[[
            "b0_log_mae", "m1_log_mae", "delta_log_mae", "m1_improved_fraction",
        ]].mean().reset_index()
        city_keys = ["city_id"]
        if "neighborhood_aggregation" in paired_summary:
            city_keys.append("neighborhood_aggregation")
        city_summary = seed_summary.groupby(city_keys, sort=True)[[
            "b0_log_mae", "m1_log_mae", "delta_log_mae", "m1_improved_fraction",
        ]].mean().reset_index()
        weighted_rows = []
        for identity, group in paired_summary.groupby(city_keys, sort=True):
            weights = group["grid_count"].to_numpy(dtype=float)
            row = dict(zip(city_keys, identity if isinstance(identity, tuple) else (identity,)))
            for metric in ("b0_log_mae", "m1_log_mae", "delta_log_mae", "m1_improved_fraction"):
                values = group[metric].to_numpy(dtype=float)
                row[metric] = float(np.average(values, weights=weights))
            weighted_rows.append(row)
        city_grid_weighted = pd.DataFrame(weighted_rows)
        macro_keys = ["seed"]
        if "neighborhood_aggregation" in paired_summary:
            macro_keys.append("neighborhood_aggregation")
        seed_city_macro = seed_summary.groupby(macro_keys, sort=True)[[
            "b0_log_mae", "m1_log_mae", "delta_log_mae", "m1_improved_fraction",
        ]].mean().reset_index()
    seed_summary.to_csv(report_dir / "stage1_seed_summary.csv", index=False)
    city_summary.to_csv(report_dir / "stage1_city_summary.csv", index=False)
    city_grid_weighted.to_csv(report_dir / "stage1_city_grid_weighted.csv", index=False)
    seed_city_macro.to_csv(report_dir / "stage1_seed_city_macro.csv", index=False)
    summary = {
        "experiment": "stage1_within_city_grid_2021_2023",
        "expected_runs": int(expected_runs),
        "completed_runs": int(len(results)),
        "primary_metric": "paired_log_mae_delta_m1_minus_b0",
        "paired_samples": int(len(paired)),
        "neighborhood_aggregations": sorted(
            results["neighborhood_aggregation"].dropna().unique().tolist()
        ) if "neighborhood_aggregation" in results else [],
    }
    write_json(report_dir / "stage1_summary.json", summary)
    lines = [
        "# Stage 1 within-city grid experiment", "",
        f"Completed runs: **{len(results)} / {expected_runs}**", "",
        "Primary metric: paired Log MAE delta (`M1 - B0`; negative means improvement).", "",
    ]
    if not results.empty:
        lines.extend(["## Run metrics", "", _markdown_table(results), ""])
    if not paired_summary.empty:
        lines.extend(["## Paired comparison", "", _markdown_table(paired_summary), ""])
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


def aggregate_annual_cross_region_runs(
    artifact_dir: Path, report_dir: Path, expected_runs: int = 12,
) -> Dict:
    """Aggregate 12-month regional transfer runs across cities and seeds."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metric_names = ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]
    records = []
    diagnostics = []
    admin_monthly_frames = []
    admin_annual_frames = []
    for metric_file in sorted(artifact_dir.rglob("metrics_monthly.csv")):
        predictions = pd.read_parquet(metric_file.parent / "predictions.parquet")
        if predictions.empty or not str(predictions["experiment"].iloc[0]).startswith("annual_cross_region"):
            continue
        monthly = pd.read_csv(metric_file)
        identity = predictions.iloc[0][["experiment", "fold_id", "model", "seed", "city_id"]].to_dict()
        record = dict(identity)
        record.update({
            "test_samples": int(len(predictions)),
            "test_grids": int(predictions["cell_id"].nunique()),
            "test_admins": int(predictions["admin_id"].nunique()),
            "months": int(predictions["period"].nunique()),
        })
        for metric in metric_names:
            record[f"{metric}_mean"] = monthly[metric].mean(skipna=True)
            record[f"{metric}_std_months"] = monthly[metric].std(skipna=True, ddof=1)
            record[f"{metric}_valid_months"] = int(monthly[metric].notna().sum())
        records.append(record)

        zero = predictions[predictions["y_tc"] == 0]
        positive = predictions[predictions["y_tc"] > 0]
        threshold = predictions["y_tc"].quantile(0.99)
        trimmed = predictions[predictions["y_tc"] <= threshold]
        monthly_biases = []
        range_ratios = []
        debiased_r2 = []
        for _, group in predictions.groupby("period", sort=True):
            error = group["pred_log"].to_numpy() - group["y_log"].to_numpy()
            monthly_biases.append(float(error.mean()))
            true_std = float(group["y_log"].std(ddof=1))
            range_ratios.append(float(group["pred_log"].std(ddof=1) / true_std) if true_std > 0 else np.nan)
            adjusted = group["pred_log"].to_numpy() - error.mean()
            debiased_r2.append(r2_score(group["y_log"], adjusted) if group["y_log"].nunique() > 1 else np.nan)
        diagnostic = dict(identity)
        diagnostic.update({
            "zero_rate": float((predictions["y_tc"] == 0).mean()),
            "zero_log_mae": mean_absolute_error(zero["y_log"], zero["pred_log"]) if len(zero) else None,
            "positive_log_mae": mean_absolute_error(positive["y_log"], positive["pred_log"]) if len(positive) else None,
            "trimmed_tc_mae": mean_absolute_error(trimmed["y_tc"], trimmed["pred_tc"]) if len(trimmed) else None,
            "trimmed_tc_rmse": mean_squared_error(trimmed["y_tc"], trimmed["pred_tc"]) ** 0.5 if len(trimmed) else None,
            "monthly_log_bias_mean": float(np.nanmean(monthly_biases)),
            "dynamic_range_ratio_mean": float(np.nanmean(range_ratios)),
            "debiased_log_r2_mean": float(np.nanmean(debiased_r2)),
        })
        diagnostics.append(diagnostic)
        for filename, destination in (
            ("metrics_by_admin_monthly.csv", admin_monthly_frames),
            ("metrics_by_admin_annual.csv", admin_annual_frames),
        ):
            path = metric_file.parent / filename
            if path.exists():
                destination.append(pd.read_csv(path).assign(
                    city_id=identity["city_id"], fold_id=identity["fold_id"],
                    model=identity["model"], seed=identity["seed"],
                ))

    results = pd.DataFrame(records)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "annual_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(report_dir / "annual_diagnostics.csv", index=False)
    if admin_monthly_frames:
        pd.concat(admin_monthly_frames, ignore_index=True).to_csv(
            report_dir / "annual_admin_monthly.csv", index=False,
        )
    if admin_annual_frames:
        pd.concat(admin_annual_frames, ignore_index=True).to_csv(
            report_dir / "annual_admin_summary.csv", index=False,
        )

    city_rows = []
    if not results.empty:
        for city_id, group in results.groupby("city_id", sort=True):
            row = {"city_id": city_id, "runs": len(group), "test_grids_mean": group["test_grids"].mean()}
            for metric in metric_names:
                values = group[f"{metric}_mean"]
                row[f"{metric}_mean"] = values.mean(skipna=True)
                row[f"{metric}_std_seeds"] = values.std(skipna=True, ddof=1)
            city_rows.append(row)
    city_summary = pd.DataFrame(city_rows)
    city_summary.to_csv(report_dir / "annual_city_summary.csv", index=False)

    seed_rows = []
    if not results.empty:
        for seed, group in results.groupby("seed", sort=True):
            row = {"seed": int(seed)}
            for metric in metric_names:
                row[metric] = group[f"{metric}_mean"].mean(skipna=True)
            seed_rows.append(row)
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(report_dir / "annual_seed_macro.csv", index=False)
    weighted_rows = []
    if not results.empty:
        for seed, group in results.groupby("seed", sort=True):
            row = {"seed": int(seed)}
            weights = group["test_grids"].to_numpy(dtype=float)
            for metric in metric_names:
                values = group[f"{metric}_mean"].to_numpy(dtype=float)
                valid = pd.notna(values)
                row[metric] = (
                    float((values[valid] * weights[valid]).sum() / weights[valid].sum())
                    if valid.any() else None
                )
            weighted_rows.append(row)
    weighted_summary = pd.DataFrame(weighted_rows)
    weighted_summary.to_csv(report_dir / "annual_seed_grid_weighted.csv", index=False)
    overall = {}
    for metric in metric_names:
        values = seed_summary[metric] if not seed_summary.empty else pd.Series(dtype=float)
        overall[metric] = {
            "mean": float(values.mean()) if not values.empty else None,
            "std_across_seeds": float(values.std(ddof=1)) if len(values) > 1 else None,
        }
    summary = {
        "experiment": "annual_cross_region_2022",
        "year": "2022",
        "completed_runs": int(len(results)),
        "expected_runs": int(expected_runs),
        "seeds": sorted(results["seed"].astype(int).unique().tolist()) if not results.empty else [],
        "overall_city_macro": overall,
    }
    write_json(report_dir / "annual_summary.json", summary)
    lines = [
        "# 2022 annual cross-region experiment", "",
        f"Completed runs: **{len(results)} / {expected_runs}**", "",
    ]
    if not results.empty:
        lines.extend(["## Per-city seed summary", "", _markdown_table(city_summary), "",
                      "## Four-city macro average by seed", "", _markdown_table(seed_summary)])
    (report_dir / "annual_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary



def aggregate_three_year_cross_region_runs(
    artifact_dir: Path, report_dir: Path, expected_runs: int = 12,
) -> Dict:
    """Aggregate 36-month region-transfer runs across cities and seeds."""
    metric_names = ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]
    records = []
    yearly_frames = []
    admin_monthly_frames = []
    admin_summary_frames = []
    experiment_name = "three_year_cross_region_2021_2023"
    for metric_file in sorted(artifact_dir.rglob("metrics_monthly.csv")):
        run_dir = metric_file.parent
        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        if predictions.empty or str(predictions["experiment"].iloc[0]) != experiment_name:
            continue
        monthly = pd.read_csv(metric_file)
        identity = predictions.iloc[0][
            ["experiment", "fold_id", "model", "seed", "city_id"]
        ].to_dict()
        record = dict(identity)
        record.update({
            "test_samples": int(len(predictions)),
            "test_grids": int(predictions["cell_id"].nunique()),
            "test_admins": int(predictions["admin_id"].nunique()),
            "months": int(predictions["period"].nunique()),
            "years": int(predictions["period"].astype(str).str[:4].nunique()),
        })
        for metric in metric_names:
            record[f"{metric}_mean"] = monthly[metric].mean(skipna=True)
            record[f"{metric}_std_months"] = monthly[metric].std(skipna=True, ddof=1)
            record[f"{metric}_valid_months"] = int(monthly[metric].notna().sum())
        records.append(record)

        yearly_path = run_dir / "metrics_yearly.csv"
        if yearly_path.exists():
            yearly_frames.append(pd.read_csv(yearly_path).assign(
                city_id=identity["city_id"], fold_id=identity["fold_id"],
                model=identity["model"], seed=identity["seed"],
                test_grids=int(predictions["cell_id"].nunique()),
            ))
        for filename, destination in (
            ("metrics_by_admin_monthly.csv", admin_monthly_frames),
            ("metrics_by_admin_three_year.csv", admin_summary_frames),
        ):
            source = run_dir / filename
            if source.exists():
                destination.append(pd.read_csv(source).assign(
                    city_id=identity["city_id"], fold_id=identity["fold_id"],
                    model=identity["model"], seed=identity["seed"],
                ))

    results = pd.DataFrame(records)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "three_year_results.csv", index=False)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly.to_csv(report_dir / "three_year_yearly.csv", index=False)
    (
        year_city_summary,
        year_seed_macro,
        year_seed_weighted,
        year_summary,
    ) = _summarize_three_year_yearly(yearly, metric_names)
    year_city_summary.to_csv(
        report_dir / "three_year_year_city_summary.csv", index=False,
    )
    year_seed_macro.to_csv(
        report_dir / "three_year_year_seed_macro.csv", index=False,
    )
    year_seed_weighted.to_csv(
        report_dir / "three_year_year_seed_grid_weighted.csv", index=False,
    )
    year_summary.to_csv(
        report_dir / "three_year_year_summary.csv", index=False,
    )
    if admin_monthly_frames:
        pd.concat(admin_monthly_frames, ignore_index=True).to_csv(
            report_dir / "three_year_admin_monthly.csv", index=False,
        )
    if admin_summary_frames:
        pd.concat(admin_summary_frames, ignore_index=True).to_csv(
            report_dir / "three_year_admin_summary.csv", index=False,
        )

    city_rows = []
    if not results.empty:
        for city_id, group in results.groupby("city_id", sort=True):
            row = {
                "city_id": city_id,
                "runs": len(group),
                "test_grids_mean": group["test_grids"].mean(),
            }
            for metric in metric_names:
                values = group[f"{metric}_mean"]
                row[f"{metric}_mean"] = values.mean(skipna=True)
                row[f"{metric}_std_seeds"] = values.std(skipna=True, ddof=1)
            city_rows.append(row)
    city_summary = pd.DataFrame(city_rows)
    city_summary.to_csv(report_dir / "three_year_city_summary.csv", index=False)

    seed_rows = []
    weighted_rows = []
    if not results.empty:
        for seed, group in results.groupby("seed", sort=True):
            macro = {"seed": int(seed)}
            weighted = {"seed": int(seed)}
            weights = group["test_grids"].to_numpy(dtype=float)
            for metric in metric_names:
                values = group[f"{metric}_mean"].to_numpy(dtype=float)
                valid = pd.notna(values)
                macro[metric] = float(pd.Series(values).mean(skipna=True))
                weighted[metric] = (
                    float((values[valid] * weights[valid]).sum() / weights[valid].sum())
                    if valid.any() else None
                )
            seed_rows.append(macro)
            weighted_rows.append(weighted)
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(report_dir / "three_year_seed_macro.csv", index=False)
    pd.DataFrame(weighted_rows).to_csv(
        report_dir / "three_year_seed_grid_weighted.csv", index=False,
    )

    overall = {}
    for metric in metric_names:
        values = seed_summary[metric] if not seed_summary.empty else pd.Series(dtype=float)
        overall[metric] = {
            "mean": float(values.mean()) if not values.empty else None,
            "std_across_seeds": float(values.std(ddof=1)) if len(values) > 1 else None,
        }
    summary = {
        "experiment": experiment_name,
        "years": ["2021", "2022", "2023"],
        "months": 36,
        "completed_runs": int(len(results)),
        "expected_runs": int(expected_runs),
        "seeds": (
            sorted(results["seed"].astype(int).unique().tolist())
            if not results.empty else []
        ),
        "overall_city_macro": overall,
    }
    write_json(report_dir / "three_year_summary.json", summary)
    yearly_summary = {
        "experiment": experiment_name,
        "metric_scope": "monthly_metrics_averaged_within_each_year",
        "city_aggregation": "equal_weight_macro_average",
        "years": {
            str(row["year"]): {
                metric: {
                    "mean": float(row[f"{metric}_mean"]),
                    "std_across_seeds": float(row[f"{metric}_std_seeds"]),
                }
                for metric in metric_names
            }
            for row in year_summary.to_dict("records")
        },
    }
    write_json(report_dir / "three_year_year_summary.json", yearly_summary)
    lines = [
        "# 2021-2023 three-year cross-region experiment", "",
        f"Completed runs: **{len(results)} / {expected_runs}**", "",
    ]
    if not results.empty:
        lines.extend([
            "## Per-city mean and standard deviation", "", _markdown_table(city_summary), "",
            "## Four-city macro average by seed", "", _markdown_table(seed_summary), "",
            "## Per-year overall summary", "", _markdown_table(year_summary), "",
            "## Per-year and city summary", "", _markdown_table(year_city_summary), "",
            "## Per-year and seed macro average", "", _markdown_table(year_seed_macro), "",
            "## Per-year run metrics", "", _markdown_table(yearly),
        ])
    (report_dir / "three_year_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    return summary


def _summarize_three_year_yearly(
    yearly: pd.DataFrame,
    metric_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize per-run yearly metrics by city, seed, and year."""
    if yearly.empty:
        empty = pd.DataFrame()
        return empty, empty.copy(), empty.copy(), empty.copy()
    required = {"year", "city_id", "seed", "test_grids"}
    required.update(f"{metric}_mean" for metric in metric_names)
    missing = sorted(required - set(yearly.columns))
    if missing:
        raise ValueError(f"Three-year yearly metrics missing columns: {', '.join(missing)}")

    year_city_rows = []
    for (year, city_id), group in yearly.groupby(["year", "city_id"], sort=True):
        row = {
            "year": str(year),
            "city_id": city_id,
            "runs": int(len(group)),
            "test_grids_mean": float(group["test_grids"].mean()),
        }
        for metric in metric_names:
            values = group[f"{metric}_mean"]
            row[f"{metric}_mean"] = values.mean(skipna=True)
            row[f"{metric}_std_seeds"] = values.std(skipna=True, ddof=1)
        year_city_rows.append(row)
    year_city_summary = pd.DataFrame(year_city_rows)

    macro_rows = []
    weighted_rows = []
    for (year, seed), group in yearly.groupby(["year", "seed"], sort=True):
        macro = {"year": str(year), "seed": int(seed), "cities": int(len(group))}
        weighted = dict(macro)
        weights = group["test_grids"].to_numpy(dtype=float)
        for metric in metric_names:
            values = group[f"{metric}_mean"].to_numpy(dtype=float)
            valid = pd.notna(values)
            macro[metric] = float(pd.Series(values).mean(skipna=True))
            weighted[metric] = (
                float((values[valid] * weights[valid]).sum() / weights[valid].sum())
                if valid.any() else None
            )
        macro_rows.append(macro)
        weighted_rows.append(weighted)
    year_seed_macro = pd.DataFrame(macro_rows)
    year_seed_weighted = pd.DataFrame(weighted_rows)

    year_rows = []
    for year, group in year_seed_macro.groupby("year", sort=True):
        row = {"year": str(year), "seeds": int(len(group))}
        for metric in metric_names:
            values = group[metric]
            row[f"{metric}_mean"] = values.mean(skipna=True)
            row[f"{metric}_std_seeds"] = values.std(skipna=True, ddof=1)
        year_rows.append(row)
    return (
        year_city_summary,
        year_seed_macro,
        year_seed_weighted,
        pd.DataFrame(year_rows),
    )
