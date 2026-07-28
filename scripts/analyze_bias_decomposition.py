#!/usr/bin/env python
"""Decompose log-space prediction error into per-month level shift and residual.

For every run's predictions.parquet, per period:

    MSE_m = bias_m^2 + Var(e_m),  e = pred_log - y_log,  bias_m = mean(e_m)

which splits the (often negative) monthly R^2 into a calibration term and a
debiased R^2 that measures spatial-pattern skill after removing the city-month
level shift:

    R2_m          = 1 - MSE_m   / Var(y_m)
    R2_debiased_m = 1 - Var(e_m)/ Var(y_m)
    bias_share_m  = bias_m^2 / MSE_m

Also reports the fold-level total-emission ratio sum(pred_tc)/sum(y_tc), a
direct measure of magnitude calibration. Requires run artifacts; point
--artifact-dir at the full stage1 artifacts directory once synced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_SAMPLES = 5


def decompose_run(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for period, group in predictions.groupby("period", sort=True):
        y = group["y_log"].to_numpy(dtype=float)
        pred = group["pred_log"].to_numpy(dtype=float)
        if len(group) < MIN_SAMPLES or np.unique(y).size < 2:
            continue
        error = pred - y
        bias = float(error.mean())
        mse = float((error ** 2).mean())
        var_error = float(error.var())
        var_y = float(y.var())
        rows.append({
            "period": str(period),
            "n": len(group),
            "bias": bias,
            "rmse": mse ** 0.5,
            "r2": 1 - mse / var_y,
            "r2_debiased": 1 - var_error / var_y,
            "bias_share_of_mse": (bias ** 2) / mse if mse > 0 else np.nan,
        })
    monthly = pd.DataFrame(rows)
    y_tc_total = float(predictions["y_tc"].sum())
    summary = {
        "months": len(monthly),
        "r2_mean": float(monthly["r2"].mean()),
        "r2_debiased_mean": float(monthly["r2_debiased"].mean()),
        "r2_gain_from_debias": float((monthly["r2_debiased"] - monthly["r2"]).mean()),
        "bias_mean": float(monthly["bias"].mean()),
        "bias_abs_mean": float(monthly["bias"].abs().mean()),
        "bias_share_of_mse_mean": float(monthly["bias_share_of_mse"].mean()),
        "tc_total_ratio": float(predictions["pred_tc"].sum() / y_tc_total) if y_tc_total > 0 else np.nan,
    }
    return monthly, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-dir", default="artifacts", help="Root of run artifacts")
    parser.add_argument("--output-dir", default="reports/stage1/diagnostics")
    args = parser.parse_args()

    artifact_dir = ROOT / args.artifact_dir
    prediction_files = sorted(artifact_dir.rglob("predictions.parquet"))
    if not prediction_files:
        parser.error(f"No predictions.parquet found under {artifact_dir}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, monthly_frames = [], []
    for path in prediction_files:
        predictions = pd.read_parquet(path)
        context = {
            column: predictions[column].iloc[0]
            for column in ("experiment", "fold_id", "model", "seed")
        }
        monthly, summary = decompose_run(predictions)
        if monthly.empty:
            print(f"skipped (too few samples per month): {path}")
            continue
        summaries.append({**context, **summary})
        monthly_frames.append(monthly.assign(**context))

    summary_frame = pd.DataFrame(summaries).sort_values(["experiment", "fold_id", "model"])
    monthly_frame = pd.concat(monthly_frames, ignore_index=True)
    suffix = "" if args.artifact_dir == "artifacts" else f"_{Path(args.artifact_dir).name}"
    summary_path = out_dir / f"bias_decomposition{suffix}.csv"
    summary_frame.to_csv(summary_path, index=False)
    monthly_frame.to_csv(out_dir / f"bias_decomposition_monthly{suffix}.csv", index=False)

    display = summary_frame[[
        "experiment", "fold_id", "model", "r2_mean", "r2_debiased_mean",
        "bias_mean", "bias_share_of_mse_mean", "tc_total_ratio",
    ]]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(display.to_string(index=False))
    print(f"\nOutputs written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
