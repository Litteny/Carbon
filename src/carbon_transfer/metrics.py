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
