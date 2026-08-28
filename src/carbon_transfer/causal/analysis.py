from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from carbon_transfer.causal.fixed_effects import fit_fixed_effects
from carbon_transfer.causal.panel import (
    add_shifted_feature, select_controls, validate_panel, within_standard_deviation,
)
from carbon_transfer.causal.registry import FeatureSpec, build_feature_registry
from carbon_transfer.utils import write_json


def _model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["city_id"] = result["city_id"].astype(str)
    result["cell_id"] = result["cell_id"].astype(str)
    result["period"] = result["period"].astype(str)
    result["entity_id"] = result["city_id"] + "::" + result["cell_id"]
    result["time_id"] = result["city_id"] + "::" + result["period"]
    return result


def _scaled_estimate(estimate: Dict, treatment_sd: float) -> Dict:
    result = dict(estimate)
    result["treatment_within_sd"] = treatment_sd
    for source, target in (
        ("coefficient", "effect_per_within_sd"),
        ("standard_error", "effect_se_per_within_sd"),
        ("ci_low", "effect_ci_low_per_within_sd"),
        ("ci_high", "effect_ci_high_per_within_sd"),
    ):
        if source in result:
            result[target] = float(result[source]) * treatment_sd
    return result


def _city_estimates(
    frame: pd.DataFrame,
    outcome: str,
    controls: Sequence[str],
    treatment_sd: float,
) -> List[Dict]:
    rows = []
    for city_id, city in frame.groupby("city_id", sort=True, observed=True):
        estimate = fit_fixed_effects(city, outcome, "treatment", controls)
        estimate.update({"city_id": str(city_id)})
        rows.append(_scaled_estimate(estimate, treatment_sd))
    return rows


def _permutation_placebo(
    frame: pd.DataFrame,
    outcome: str,
    controls: Sequence[str],
    treatment_sd: float,
    iterations: int,
    seed: int,
) -> Dict:
    if iterations <= 0:
        return {"iterations": 0, "status": "not_run"}
    rng = np.random.default_rng(seed)
    values = frame["treatment"].to_numpy(copy=True)
    entities = frame["entity_id"].to_numpy()
    positions = [np.flatnonzero(entities == value) for value in np.unique(entities)]
    effects = []
    for _ in range(iterations):
        permuted = values.copy()
        for selected in positions:
            permuted[selected] = rng.permutation(permuted[selected])
        placebo = frame.copy()
        placebo["treatment"] = permuted
        estimate = fit_fixed_effects(placebo, outcome, "treatment", controls)
        if estimate.get("identified"):
            effects.append(float(estimate["coefficient"]) * treatment_sd)
    if not effects:
        return {"iterations": iterations, "successful": 0, "status": "not_identified"}
    return {
        "iterations": iterations,
        "successful": len(effects),
        "status": "complete",
        "abs_effect_95pct": float(np.quantile(np.abs(effects), 0.95)),
        "mean_effect": float(np.mean(effects)),
        "effects": effects,
    }


def _evidence_grade(
    spec: FeatureSpec,
    lag: int,
    estimate: Mapping,
    city_estimates: Sequence[Mapping],
    lead_significant: Optional[bool],
    label_sources: Sequence[str],
    placebo_failed: bool = False,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    if not estimate.get("identified"):
        return "not_identified", [str(estimate.get("reason", "unknown"))]
    if spec.role in {"measurement_quality", "control_or_measurement"}:
        return "not_causal", ["feature is a measurement/control variable"]
    if spec.family == "viirs" and (
        not label_sources or "viirs" in {value.lower() for value in label_sources}
    ):
        return "D", ["VIIRS label provenance is unknown or circular"]
    if lag < 0:
        return "negative_control", ["future exposure is a temporal negative control"]

    p_value = float(estimate.get("p_value", 1.0))
    identified_cities = [row for row in city_estimates if row.get("identified")]
    significant_cities = [row for row in identified_cities if float(row.get("p_value", 1.0)) < 0.05]
    if significant_cities:
        signs = np.sign([float(row["coefficient"]) for row in significant_cities])
        direction_consistency = float(max(np.mean(signs > 0), np.mean(signs < 0)))
    else:
        direction_consistency = 0.0

    grade = "C" if p_value < 0.05 else "D"
    reasons.append("observational fixed-effects estimate; unmeasured time-varying confounding remains")
    if p_value < 0.05 and direction_consistency >= 0.75 and len(significant_cities) >= 2:
        grade = "C+"
        reasons.append("effect direction is consistent in at least two significant city estimates")
    if spec.role == "outcome_proxy":
        grade = "D"
        reasons.append("feature is an outcome/activity proxy")
    if spec.family == "poi" and grade == "C+":
        grade = "C"
        reasons.append("POI openings/closures are not externally randomized")
    if lead_significant:
        grade = "D"
        reasons.append("future-exposure negative control is significant")
    if placebo_failed:
        grade = "D"
        reasons.append("observed effect does not exceed the permutation-placebo 95% threshold")
    return grade, reasons


def analyze_feature(
    panel: pd.DataFrame,
    spec: FeatureSpec,
    *,
    outcome: str,
    lag: int,
    controls: Sequence[str],
    placebo_iterations: int = 0,
    seed: int = 42,
) -> Dict:
    shifted = add_shifted_feature(_model_frame(panel), spec.name, lag)
    treatment_sd = within_standard_deviation(shifted["treatment"], shifted["entity_id"])
    estimate = _scaled_estimate(
        fit_fixed_effects(shifted, outcome, "treatment", controls), treatment_sd,
    )
    cities = _city_estimates(shifted, outcome, controls, treatment_sd)
    placebo = _permutation_placebo(
        shifted.dropna(subset=["treatment"]), outcome, controls, treatment_sd,
        placebo_iterations, seed,
    )
    return {
        "feature": spec.name,
        "family": spec.family,
        "role": spec.role,
        "lag": lag,
        "temporal_role": "future_negative_control" if lag < 0 else (
            "contemporaneous" if lag == 0 else "lagged_exposure"
        ),
        "intervention": spec.intervention,
        "controls": list(controls),
        "estimate": estimate,
        "city_estimates": cities,
        "placebo": placebo,
        "caveat": spec.caveat,
    }


def _summary_row(
    result: Mapping,
    label_sources: Sequence[str],
    lead_significant: Optional[bool],
) -> Dict:
    spec = FeatureSpec(
        name=str(result["feature"]), family=str(result["family"]), role=str(result["role"]),
        estimable=True, intervention=str(result["intervention"]),
        default_lags=(int(result["lag"]),), caveat=str(result["caveat"]),
    )
    estimate = result["estimate"]
    placebo = result.get("placebo", {})
    observed_effect = estimate.get("effect_per_within_sd")
    placebo_failed = bool(
        placebo.get("status") == "complete"
        and observed_effect is not None
        and abs(float(observed_effect)) <= float(placebo["abs_effect_95pct"])
    )
    grade, reasons = _evidence_grade(
        spec, int(result["lag"]), estimate, result["city_estimates"],
        lead_significant, label_sources, placebo_failed,
    )
    city_rows = [row for row in result["city_estimates"] if row.get("identified")]
    effect = estimate.get("effect_per_within_sd")
    direction = "not_identified"
    if effect is not None:
        direction = "increase" if effect > 0 else ("decrease" if effect < 0 else "zero")
    return {
        "feature": result["feature"], "family": result["family"], "role": result["role"],
        "lag": result["lag"], "temporal_role": result["temporal_role"],
        "identified": bool(estimate.get("identified")), "effect_direction": direction,
        "effect_per_within_sd": effect,
        "effect_ci_low_per_within_sd": estimate.get("effect_ci_low_per_within_sd"),
        "effect_ci_high_per_within_sd": estimate.get("effect_ci_high_per_within_sd"),
        "p_value": estimate.get("p_value"), "n_obs": estimate.get("n_obs"),
        "cities_identified": len(city_rows),
        "cities_significant": sum(float(row.get("p_value", 1.0)) < 0.05 for row in city_rows),
        "future_negative_control_significant": lead_significant,
        "permutation_placebo_failed": placebo_failed,
        "evidence_grade": grade, "evidence_reasons": "; ".join(reasons),
        "caveat": result["caveat"],
    }


def _write_report(summary: pd.DataFrame, registry: Sequence[FeatureSpec], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "feature_evidence_table.csv", index=False)
    ranking = summary[
        (summary["lag"] >= 0) & summary["identified"]
    ].copy()
    grade_order = {"C+": 0, "C": 1, "D": 2, "not_identified": 3}
    ranking["grade_order"] = ranking["evidence_grade"].map(grade_order).fillna(9)
    ranking = ranking.sort_values(["grade_order", "p_value", "feature", "lag"])
    ranking.drop(columns=["grade_order"]).to_csv(
        report_dir / "causal_candidate_ranking.csv", index=False,
    )
    registry_frame = pd.DataFrame([spec.to_dict() for spec in registry])
    registry_frame.to_csv(report_dir / "feature_registry.csv", index=False)

    top = ranking.head(20)
    lines = [
        "# Feature causal-candidate analysis", "",
        "> These are observational panel diagnostics, not proof of intervention effects.", "",
        "## Identification design", "",
        "Grid fixed effects absorb time-invariant local differences. City-by-period fixed effects absorb city-wide monthly shocks. Standard errors are clustered by grid. A negative lag uses a future feature as a temporal negative control.", "",
        "## Highest-ranked estimates", "",
    ]
    if top.empty:
        lines.append("No feature-lag estimate was identified.")
    else:
        lines.extend(["| Feature | Lag | Effect per within SD | 95% CI | p | Grade |", "|---|---:|---:|---:|---:|---|"])
        for row in top.itertuples():
            effect = "NA" if pd.isna(row.effect_per_within_sd) else f"{row.effect_per_within_sd:.4g}"
            interval = "NA" if pd.isna(row.effect_ci_low_per_within_sd) else f"[{row.effect_ci_low_per_within_sd:.4g}, {row.effect_ci_high_per_within_sd:.4g}]"
            p_value = "NA" if pd.isna(row.p_value) else f"{row.p_value:.3g}"
            lines.append(f"| {row.feature} | {row.lag} | {effect} | {interval} | {p_value} | {row.evidence_grade} |")
    lines.extend([
        "", "## Interpretation limits", "",
        "Grades C/C+ mean a feature survives this observational fixed-effects audit. Grade D means predictive association only, failed negative control, or proxy/circularity risk. A/B grades require a separately configured natural experiment, policy event, valid instrument, or comparable design and are intentionally unavailable here.", "",
    ])
    (report_dir / "causal_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_feature_causal_analysis(
    config: Mapping,
    *,
    features: Optional[Iterable[str]] = None,
    lags: Optional[Sequence[int]] = None,
    force: bool = False,
) -> Dict:
    panel_path = Path(str(config["panel_file"]))
    panel = pd.read_parquet(panel_path)
    outcome = str(config.get("outcome", "log1p_emission"))
    validate_panel(panel, outcome)
    configured_lags = tuple(int(value) for value in (lags or config.get("lags", (0, 1, 3, 6, -1))))
    registry = build_feature_registry(
        panel.columns, lags=configured_lags, overrides=config.get("feature_overrides"),
    )
    selected = set(features or ())
    unknown = sorted(selected - {spec.name for spec in registry})
    if unknown:
        raise ValueError(f"Unknown causal features: {', '.join(unknown)}")
    specs = [spec for spec in registry if spec.estimable and (not selected or spec.name in selected)]
    if selected and not specs:
        raise ValueError(
            "Selected features are controls/measurement variables and have no causal estimand"
        )
    artifact_dir = Path(str(config["artifact_dir"]))
    report_dir = Path(str(config["report_dir"]))
    controls_config = dict(config.get("controls", {}))
    placebo_iterations = int(config.get("placebo_iterations", 0))
    seed = int(config.get("seed", 42))
    label_sources = tuple(str(value) for value in config.get("label_feature_sources", ()))

    results: List[Dict] = []
    for spec in specs:
        controls = select_controls(spec.family, spec.name, panel.columns, controls_config)
        for lag in configured_lags:
            run_dir = artifact_dir / spec.name / f"lag_{lag:+d}"
            complete = run_dir / "COMPLETE"
            result_file = run_dir / "estimate.json"
            if complete.exists() and not force:
                if not result_file.exists():
                    raise FileNotFoundError(f"Completed causal run is missing estimate: {run_dir}")
                with result_file.open(encoding="utf-8") as handle:
                    results.append(json.load(handle))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            result = analyze_feature(
                panel, spec, outcome=outcome, lag=lag, controls=controls,
                placebo_iterations=placebo_iterations if lag >= 0 else 0,
                seed=seed + lag,
            )
            write_json(result_file, result)
            pd.DataFrame(result["city_estimates"]).to_csv(run_dir / "city_effects.csv", index=False)
            complete.write_text("complete\n", encoding="utf-8")
            results.append(result)

    lead_flags = {}
    for result in results:
        if int(result["lag"]) < 0:
            estimate = result["estimate"]
            lead_flags[result["feature"]] = bool(
                estimate.get("identified") and float(estimate.get("p_value", 1.0)) < 0.05
            )
    summary = pd.DataFrame([
        _summary_row(result, label_sources, lead_flags.get(result["feature"]))
        for result in results
    ])
    _write_report(summary, registry, report_dir)
    metadata = {
        "panel_file": str(panel_path), "outcome": outcome,
        "features_analyzed": len(specs), "estimates": len(results),
        "lags": list(configured_lags), "label_feature_sources": list(label_sources),
        "artifact_dir": str(artifact_dir), "report_dir": str(report_dir),
        "claim_boundary": "observational causal-candidate evidence only",
    }
    write_json(report_dir / "analysis_metadata.json", metadata)
    return metadata


def describe_analysis(
    config: Mapping,
    features: Optional[Iterable[str]] = None,
    lags: Optional[Sequence[int]] = None,
) -> Dict:
    panel = pd.read_parquet(Path(str(config["panel_file"])))
    outcome = str(config.get("outcome", "log1p_emission"))
    validate_panel(panel, outcome)
    configured_lags = tuple(int(value) for value in (lags or config.get("lags", (0, 1, 3, 6, -1))))
    registry = build_feature_registry(
        panel.columns, lags=configured_lags, overrides=config.get("feature_overrides"),
    )
    selected = set(features or ())
    unknown = sorted(selected - {spec.name for spec in registry})
    if unknown:
        raise ValueError(f"Unknown causal features: {', '.join(unknown)}")
    specs = [spec for spec in registry if spec.estimable and (not selected or spec.name in selected)]
    if selected and not specs:
        raise ValueError(
            "Selected features are controls/measurement variables and have no causal estimand"
        )
    return {
        "rows": len(panel), "cities": int(panel["city_id"].nunique()),
        "periods": int(panel["period"].nunique()), "features": [spec.name for spec in specs],
        "lags": list(configured_lags), "tasks": len(specs) * len(configured_lags),
    }
