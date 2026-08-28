from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def _alternating_demean(
    values: pd.DataFrame,
    entity: pd.Series,
    time: pd.Series,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> np.ndarray:
    work = values.astype(float).copy()
    for _ in range(max_iterations):
        previous = work.to_numpy(copy=True)
        work -= work.groupby(entity, observed=True).transform("mean")
        work -= work.groupby(time, observed=True).transform("mean")
        if float(np.max(np.abs(work.to_numpy() - previous))) < tolerance:
            break
    return work.to_numpy(dtype=float)


def fit_fixed_effects(
    frame: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: Sequence[str] = (),
    *,
    entity_column: str = "entity_id",
    time_column: str = "time_id",
    cluster_column: str = "entity_id",
) -> Dict:
    """OLS after absorbing entity and time effects, with entity-clustered SE."""
    columns = list(dict.fromkeys([
        outcome, treatment, *controls, entity_column, time_column, cluster_column,
    ]))
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    regressors = [treatment, *controls]
    if len(data) <= len(regressors) + 1:
        return {"identified": False, "reason": "insufficient_complete_rows", "n_obs": len(data)}

    transformed = _alternating_demean(
        data[[outcome, *regressors]], data[entity_column], data[time_column],
    )
    y = transformed[:, 0]
    x = transformed[:, 1:]
    treatment_variance = float(np.var(x[:, 0]))
    if treatment_variance < 1e-14:
        return {
            "identified": False,
            "reason": "no_within_fixed_effect_variation",
            "n_obs": len(data),
        }
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        return {"identified": False, "reason": "collinear_design", "n_obs": len(data)}

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    clusters = data[cluster_column].astype(str).to_numpy()
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for cluster in np.unique(clusters):
        selected = clusters == cluster
        score = x[selected].T @ residual[selected]
        meat += np.outer(score, score)
    cluster_count = int(np.unique(clusters).size)
    correction = 1.0
    if cluster_count > 1 and len(data) > x.shape[1]:
        correction = cluster_count / (cluster_count - 1) * (len(data) - 1) / (len(data) - x.shape[1])
    covariance = correction * xtx_inv @ meat @ xtx_inv
    standard_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    coefficient = float(beta[0])
    z_value = coefficient / standard_error if standard_error > 0 else np.nan
    p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return {
        "identified": True,
        "coefficient": coefficient,
        "standard_error": standard_error,
        "ci_low": coefficient - 1.96 * standard_error,
        "ci_high": coefficient + 1.96 * standard_error,
        "p_value": p_value,
        "n_obs": int(len(data)),
        "n_entities": int(data[entity_column].nunique()),
        "n_clusters": cluster_count,
        "n_controls": len(controls),
        "residual_rmse": float(np.sqrt(np.mean(residual ** 2))),
    }
