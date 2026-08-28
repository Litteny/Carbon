from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def distribution_audit(panel: pd.DataFrame, feature: str) -> tuple[Dict, List[Dict]]:
    rows = []
    cities = sorted(panel["city_id"].astype(str).unique())
    for target_city in cities:
        target = panel.loc[panel["city_id"].astype(str) == target_city, feature].dropna().to_numpy(dtype=float)
        source = panel.loc[panel["city_id"].astype(str) != target_city, feature].dropna().to_numpy(dtype=float)
        if not len(target) or not len(source):
            continue
        source_sd = float(np.std(source, ddof=1))
        lower, upper = np.quantile(source, [0.05, 0.95])
        overlap = float(np.mean((target >= lower) & (target <= upper)))
        normalized_wasserstein = float(wasserstein_distance(source, target) / max(source_sd, 1e-12))
        standardized_mean_difference = float(
            abs(np.mean(target) - np.mean(source))
            / max(np.sqrt((np.var(target) + np.var(source)) / 2.0), 1e-12)
        )
        rows.append({
            "feature": feature,
            "target_city": target_city,
            "support_overlap": overlap,
            "normalized_wasserstein": normalized_wasserstein,
            "standardized_mean_difference": standardized_mean_difference,
            "target_below_source_p05": float(np.mean(target < lower)),
            "target_above_source_p95": float(np.mean(target > upper)),
        })
    return {
        "feature": feature,
        "mean_support_overlap": float(np.mean([row["support_overlap"] for row in rows])),
        "worst_support_overlap": float(np.min([row["support_overlap"] for row in rows])),
        "mean_normalized_wasserstein": float(np.mean([row["normalized_wasserstein"] for row in rows])),
        "worst_normalized_wasserstein": float(np.max([row["normalized_wasserstein"] for row in rows])),
    }, rows
