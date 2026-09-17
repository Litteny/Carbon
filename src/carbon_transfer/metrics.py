from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prediction_frame(metadata: pd.DataFrame, predictions: np.ndarray, context: Dict) -> pd.DataFrame:
    result = metadata[["city_id", "admin_id", "period", "cell_id", "log1p_emission", "emission_tc"]].copy()
    result = result.rename(columns={"log1p_emission": "y_log", "emission_tc": "y_tc"})
    result["pred_log"] = np.asarray(predictions, dtype=float)
    result["pred_tc"] = np.maximum(0.0, np.expm1(result["pred_log"].to_numpy()))
    for column in ("experiment", "fold_id", "model", "seed"):
        result.insert(0, column, context[column])
    return result


def calculate_metrics(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    for period, group in predictions.groupby("period", sort=True):
        y_log = group["y_log"].to_numpy(dtype=float)
        pred_log = group["pred_log"].to_numpy(dtype=float)
        y_tc = group["y_tc"].to_numpy(dtype=float)
        pred_tc = group["pred_tc"].to_numpy(dtype=float)
        valid_count = int(np.isfinite(y_log).sum())
        log_r2 = r2_score(y_log, pred_log) if valid_count >= 5 and np.unique(y_log).size > 1 else np.nan
        correlation = spearmanr(y_log, pred_log).statistic if valid_count >= 2 else np.nan
        rows.append({
            "period": str(period), "n": len(group), "log_r2": log_r2,
            "log_mae": mean_absolute_error(y_log, pred_log),
            "log_rmse": mean_squared_error(y_log, pred_log) ** 0.5,
            "log_spearman": correlation,
            "tc_mae": mean_absolute_error(y_tc, pred_tc),
            "tc_rmse": mean_squared_error(y_tc, pred_tc) ** 0.5,
            "r2_status": "ok" if np.isfinite(log_r2) else "undefined_less_than_5_or_constant",
        })
    monthly = pd.DataFrame(rows)
    metric_columns = ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]
    summary = {
        "single_seed_preliminary": True,
        "periods": int(monthly["period"].nunique()),
        "samples": int(len(predictions)),
        "metrics": {
            column: {
                "mean": float(monthly[column].mean(skipna=True)),
                "std_across_months": (
                    float(monthly[column].std(skipna=True, ddof=1))
                    if monthly[column].notna().sum() > 1 else None
                ),
                "valid_months": int(monthly[column].notna().sum()),
            }
            for column in metric_columns
        },
    }
    return monthly, summary


def calculate_pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate one pooled metric row across all grid-month test samples."""
    y_log = predictions["y_log"].to_numpy(dtype=float)
    pred_log = predictions["pred_log"].to_numpy(dtype=float)
    y_tc = predictions["y_tc"].to_numpy(dtype=float)
    pred_tc = predictions["pred_tc"].to_numpy(dtype=float)
    valid = np.isfinite(y_log) & np.isfinite(pred_log) & np.isfinite(y_tc) & np.isfinite(pred_tc)
    y_log, pred_log = y_log[valid], pred_log[valid]
    y_tc, pred_tc = y_tc[valid], pred_tc[valid]
    log_r2 = r2_score(y_log, pred_log) if len(y_log) >= 2 and np.unique(y_log).size > 1 else np.nan
    correlation = spearmanr(y_log, pred_log).statistic if len(y_log) >= 2 else np.nan
    return pd.DataFrame([{
        "scope": "pooled",
        "n": int(len(y_log)),
        "log_r2": log_r2,
        "log_mae": mean_absolute_error(y_log, pred_log) if len(y_log) else np.nan,
        "log_rmse": mean_squared_error(y_log, pred_log) ** 0.5 if len(y_log) else np.nan,
        "log_spearman": correlation,
        "tc_mae": mean_absolute_error(y_tc, pred_tc) if len(y_tc) else np.nan,
        "tc_rmse": mean_squared_error(y_tc, pred_tc) ** 0.5 if len(y_tc) else np.nan,
        "r2_status": "ok" if np.isfinite(log_r2) else "undefined_less_than_2_or_constant",
    }])


def calculate_yearly_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly metrics independently for each calendar year."""
    metric_columns = ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]
    frame = monthly.copy()
    frame["year"] = frame["period"].astype(str).str[:4]
    rows = []
    for year, group in frame.groupby("year", sort=True):
        row = {"year": str(year), "months": int(group["period"].nunique()), "n": int(group["n"].sum())}
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean(skipna=True)
            row[f"{metric}_std_months"] = group[metric].std(skipna=True, ddof=1)
            row[f"{metric}_valid_months"] = int(group[metric].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_admin_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate single-month metrics independently for each test region."""
    rows = []
    for admin_id, group in predictions.groupby("admin_id", sort=True, dropna=False):
        metrics, _ = calculate_metrics(group)
        row = metrics.iloc[0].to_dict()
        row["admin_id"] = admin_id
        rows.append(row)
    columns = [
        "admin_id", "period", "n", "log_r2", "log_mae", "log_rmse",
        "log_spearman", "tc_mae", "tc_rmse", "r2_status",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def calculate_annual_admin_metrics(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-admin monthly metrics and their annual summaries."""
    metric_columns = ["log_r2", "log_mae", "log_rmse", "log_spearman", "tc_mae", "tc_rmse"]
    monthly_frames = []
    annual_rows = []
    for admin_id, group in predictions.groupby("admin_id", sort=True, dropna=False):
        monthly, _ = calculate_metrics(group)
        monthly.insert(0, "admin_id", admin_id)
        monthly_frames.append(monthly)
        row = {
            "admin_id": admin_id,
            "months": int(monthly["period"].nunique()),
            "samples": int(len(group)),
            "grids": int(group["cell_id"].nunique()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = monthly[metric].mean(skipna=True)
            row[f"{metric}_std_months"] = monthly[metric].std(skipna=True, ddof=1)
            row[f"{metric}_valid_months"] = int(monthly[metric].notna().sum())
        annual_rows.append(row)
    monthly_result = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    return monthly_result, pd.DataFrame(annual_rows)
