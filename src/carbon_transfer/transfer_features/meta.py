from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import numpy as np
from scipy.stats import chi2


def _pool(rows: Iterable[Mapping]) -> Dict:
    selected = [
        row for row in rows
        if row.get("identified")
        and np.isfinite(float(row.get("effect_per_within_sd", np.nan)))
        and float(row.get("effect_se_per_within_sd", 0.0)) > 0
    ]
    if not selected:
        return {"identified": False, "reason": "no_valid_city_effects"}
    effects = np.asarray([float(row["effect_per_within_sd"]) for row in selected])
    standard_errors = np.asarray([float(row["effect_se_per_within_sd"]) for row in selected])
    weights = 1.0 / standard_errors ** 2
    effect = float(np.sum(weights * effects) / np.sum(weights))
    standard_error = float(np.sqrt(1.0 / np.sum(weights)))
    q_value = float(np.sum(weights * (effects - effect) ** 2))
    degrees_freedom = len(selected) - 1
    i2 = float(max(0.0, (q_value - degrees_freedom) / q_value)) if q_value > 0 else 0.0
    positive = float(np.mean(effects > 0))
    sign_consistency = max(positive, 1.0 - positive)
    return {
        "identified": True,
        "effect": effect,
        "standard_error": standard_error,
        "ci_low": effect - 1.96 * standard_error,
        "ci_high": effect + 1.96 * standard_error,
        "q": q_value,
        "q_p_value": float(chi2.sf(q_value, degrees_freedom)) if degrees_freedom else np.nan,
        "i2": i2,
        "cities": len(selected),
        "sign_consistency": sign_consistency,
    }


def summarize_city_effects(feature: str, city_rows: List[Mapping]) -> tuple[Dict, List[Dict]]:
    overall = _pool(city_rows)
    if not overall.get("identified"):
        return {"feature": feature, **overall}, []
    loco_rows = []
    for target in sorted(str(row["city_id"]) for row in city_rows):
        target_row = next(row for row in city_rows if str(row["city_id"]) == target)
        source = _pool(row for row in city_rows if str(row["city_id"]) != target)
        if not source.get("identified") or not target_row.get("identified"):
            continue
        target_effect = float(target_row["effect_per_within_sd"])
        source_effect = float(source["effect"])
        loco_rows.append({
            "feature": feature,
            "target_city": target,
            "source_effect": source_effect,
            "source_se": source["standard_error"],
            "source_ci_low": source["ci_low"],
            "source_ci_high": source["ci_high"],
            "target_effect": target_effect,
            "target_se": float(target_row["effect_se_per_within_sd"]),
            "sign_agreement": bool(np.sign(source_effect) == np.sign(target_effect)),
            "target_in_source_ci": bool(source["ci_low"] <= target_effect <= source["ci_high"]),
            "absolute_effect_difference": abs(source_effect - target_effect),
            "source_sign_consistency": source["sign_consistency"],
        })
    source_effects = np.asarray([row["source_effect"] for row in loco_rows], dtype=float)
    overall.update({
        "feature": feature,
        "loco_sign_agreement": float(np.mean([row["sign_agreement"] for row in loco_rows])) if loco_rows else np.nan,
        "loco_ci_coverage": float(np.mean([row["target_in_source_ci"] for row in loco_rows])) if loco_rows else np.nan,
        "loco_source_effect_sd": float(np.std(source_effects, ddof=1)) if len(source_effects) > 1 else 0.0,
    })
    return overall, loco_rows
