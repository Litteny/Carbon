from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from carbon_transfer.causal.fixed_effects import _alternating_demean


def _partial_r2(y: np.ndarray, treatment: np.ndarray, controls: np.ndarray) -> float:
    if controls.shape[1]:
        base_beta = np.linalg.lstsq(controls, y, rcond=None)[0]
        base_residual = y - controls @ base_beta
    else:
        base_residual = y
    full = np.column_stack([treatment, controls])
    full_beta = np.linalg.lstsq(full, y, rcond=None)[0]
    full_residual = y - full @ full_beta
    base_sse = float(base_residual @ base_residual)
    return max(0.0, (base_sse - float(full_residual @ full_residual)) / max(base_sse, 1e-15))


def partial_r2_audit(
    panel: pd.DataFrame,
    feature: str,
    outcome: str,
    controls: Sequence[str],
    *,
    bootstrap_iterations: int = 20,
    seed: int = 42,
    relevance_threshold: float = 1e-4,
) -> tuple[Dict, List[Dict]]:
    columns = list(dict.fromkeys(["city_id", "cell_id", "period", outcome, feature, *controls]))
    data = panel[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    data["entity_id"] = data["city_id"].astype(str) + "::" + data["cell_id"].astype(str)
    data["time_id"] = data["city_id"].astype(str) + "::" + data["period"].astype(str)
    transformed = _alternating_demean(
        data[[outcome, feature, *controls]], data["entity_id"], data["time_id"],
    )
    y = transformed[:, 0]
    treatment = transformed[:, 1]
    control_values = transformed[:, 2:]
    overall = _partial_r2(y, treatment, control_values)

    city_rows = []
    city_values = data["city_id"].astype(str).to_numpy()
    for city in sorted(np.unique(city_values)):
        selected = city_values == city
        city_rows.append({
            "feature": feature,
            "city_id": city,
            "partial_r2": _partial_r2(y[selected], treatment[selected], control_values[selected]),
        })

    rng = np.random.default_rng(seed)
    periods = data["period"].astype(str).to_numpy()
    unique_periods = np.unique(periods)
    indices = {period: np.flatnonzero(periods == period) for period in unique_periods}
    bootstrap = []
    for _ in range(bootstrap_iterations):
        sampled = rng.choice(unique_periods, size=len(unique_periods), replace=True)
        selected = np.concatenate([indices[period] for period in sampled])
        bootstrap.append(_partial_r2(y[selected], treatment[selected], control_values[selected]))
    city_information = np.asarray([row["partial_r2"] for row in city_rows])
    return {
        "feature": feature,
        "partial_r2": overall,
        "partial_r2_ci_low": float(np.quantile(bootstrap, 0.025)) if bootstrap else np.nan,
        "partial_r2_ci_high": float(np.quantile(bootstrap, 0.975)) if bootstrap else np.nan,
        "bootstrap_relevance_frequency": float(np.mean(np.asarray(bootstrap) >= relevance_threshold)) if bootstrap else np.nan,
        "cities_above_relevance_threshold": int(np.sum(city_information >= relevance_threshold)),
        "city_partial_r2_cv": float(np.std(city_information) / max(np.mean(city_information), 1e-15)),
        "information_rows": len(data),
        "controls": list(controls),
    }, city_rows
