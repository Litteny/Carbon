from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from carbon_transfer.causal.panel import select_controls, validate_panel
from carbon_transfer.causal.registry import FeatureSpec, build_feature_registry
from carbon_transfer.transfer_features.distribution import distribution_audit
from carbon_transfer.transfer_features.domain import domain_classification
from carbon_transfer.transfer_features.information import partial_r2_audit
from carbon_transfer.transfer_features.meta import summarize_city_effects
from carbon_transfer.utils import write_json


def _rank_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    return values.rank(method="average", pct=True, ascending=higher_is_better)


def _risk_score(spec: FeatureSpec, temporal_risk: bool) -> tuple[float, str]:
    risks = []
    score = 0.0
    if spec.role == "outcome_proxy":
        score += 0.7
        risks.append("outcome/activity proxy")
    if spec.family == "poi":
        score += 0.25
        risks.append("POI measurement/update risk")
    if spec.family == "modis" and "reflectance" in spec.name:
        score += 0.2
        risks.append("mixed reflectance mechanism")
    if temporal_risk:
        score += 0.15
        risks.append("temporal confounding signal")
    return min(score, 1.0), "; ".join(risks) if risks else "none"


def _load_city_effects(run_root: Path, feature: str) -> tuple[List[Dict], bool]:
    current = run_root / feature / "lag_+0" / "estimate.json"
    future = run_root / feature / "lag_-1" / "estimate.json"
    if not current.exists():
        raise FileNotFoundError(f"Missing contemporaneous city effects: {current}")
    with current.open(encoding="utf-8") as handle:
        city_rows = json.load(handle)["city_estimates"]
    temporal_risk = False
    if future.exists():
        with future.open(encoding="utf-8") as handle:
            estimate = json.load(handle)["estimate"]
        temporal_risk = bool(
            estimate.get("identified") and float(estimate.get("p_value", 1.0)) < 0.05
        )
    return city_rows, temporal_risk


def _write_report(ranking: pd.DataFrame, report_dir: Path) -> None:
    top = ranking.head(20)
    lines = [
        "# Cross-city feature transfer screening", "",
        "> This is a model-independent statistical screen, not a causal proof or target-model ablation.", "",
        "## Scoring", "",
        "The total score combines relation stability (35%), distribution/domain transferability (30%), independent information (25%), and measurement-risk penalty (10%). All source metrics remain in the CSV report.", "",
        "## Ranking", "",
        "| Rank | Feature | Tier | Score | I² | LOCO sign | Support | Domain accuracy | Partial R² | Risk |", 
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in top.itertuples():
        lines.append(
            f"| {row.transfer_rank} | {row.feature} | {row.screening_tier} | {row.transfer_score:.3f} | "
            f"{row.i2:.1%} | {row.loco_sign_agreement:.1%} | {row.mean_support_overlap:.1%} | "
            f"{row.domain_balanced_accuracy:.1%} | {row.partial_r2:.3g} | {row.risk_flags} |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "High-ranked features combine similar city-level relationships, source-target support, weak city-identification signal, and stable conditional information. A high score means statistically promising for transfer; only leave-one-city-out prediction or ablation can establish actual target-city performance gain.", "",
    ])
    (report_dir / "feature_transfer_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_feature_transfer_screening(
    config: Mapping,
    *,
    features: Optional[Iterable[str]] = None,
) -> Dict:
    panel_path = Path(str(config["panel_file"]))
    causal_run_dir = Path(str(config["causal_run_dir"]))
    report_dir = Path(str(config["report_dir"]))
    panel = pd.read_parquet(panel_path)
    outcome = str(config.get("outcome", "log1p_emission"))
    validate_panel(panel, outcome)
    registry = build_feature_registry(panel.columns, overrides=config.get("feature_overrides"))
    selected = set(features or ())
    unknown = sorted(selected - {spec.name for spec in registry})
    if unknown:
        raise ValueError(f"Unknown transfer-screening features: {', '.join(unknown)}")
    specs = [spec for spec in registry if spec.estimable and (not selected or spec.name in selected)]
    if not specs:
        raise ValueError("No estimable features selected for transfer screening")

    controls_config = dict(config.get("controls", {}))
    bootstrap_iterations = int(config.get("bootstrap_iterations", 20))
    domain_folds = int(config.get("domain_folds", 5))
    max_domain_samples = int(config.get("max_domain_samples_per_city", 5000))
    relevance_threshold = float(config.get("partial_r2_relevance_threshold", 1e-4))
    seed = int(config.get("seed", 42))

    summary_rows = []
    loco_rows: List[Dict] = []
    distribution_rows: List[Dict] = []
    city_information_rows: List[Dict] = []
    for position, spec in enumerate(specs):
        city_effects, temporal_risk = _load_city_effects(causal_run_dir, spec.name)
        meta, feature_loco = summarize_city_effects(spec.name, city_effects)
        distribution, feature_distribution = distribution_audit(panel, spec.name)
        domain = domain_classification(
            panel, spec.name, folds=domain_folds,
            max_samples_per_city=max_domain_samples, seed=seed,
        )
        controls = select_controls(spec.family, spec.name, panel.columns, controls_config)
        information, feature_city_information = partial_r2_audit(
            panel, spec.name, outcome, controls,
            bootstrap_iterations=bootstrap_iterations, seed=seed + position,
            relevance_threshold=relevance_threshold,
        )
        risk_score, risk_flags = _risk_score(spec, temporal_risk)
        summary_rows.append({
            "feature": spec.name, "family": spec.family, "role": spec.role,
            **meta, **distribution, **domain, **information,
            "temporal_confounding_risk": temporal_risk,
            "risk_score": risk_score, "risk_flags": risk_flags,
        })
        loco_rows.extend(feature_loco)
        distribution_rows.extend(feature_distribution)
        city_information_rows.extend(feature_city_information)

    ranking = pd.DataFrame(summary_rows)
    ranking["relation_stability_score"] = (
        0.30 * ranking["sign_consistency"]
        + 0.30 * ranking["loco_sign_agreement"]
        + 0.25 * (1.0 - ranking["i2"].clip(0.0, 1.0))
        + 0.15 * ranking["loco_ci_coverage"]
    )
    ranking["distribution_transfer_score"] = (
        0.40 * ranking["mean_support_overlap"]
        + 0.20 * ranking["worst_support_overlap"]
        + 0.20 * np.exp(-ranking["mean_normalized_wasserstein"])
        + 0.20 * ranking["domain_invariance_score"]
    )
    ranking["information_score"] = (
        0.50 * _rank_score(ranking["partial_r2"], higher_is_better=True)
        + 0.25 * ranking["bootstrap_relevance_frequency"]
        + 0.25 * ranking["cities_above_relevance_threshold"] / panel["city_id"].nunique()
    )
    ranking["transfer_score"] = (
        0.35 * ranking["relation_stability_score"]
        + 0.30 * ranking["distribution_transfer_score"]
        + 0.25 * ranking["information_score"]
        - 0.10 * ranking["risk_score"]
    ).clip(0.0, 1.0)
    ranking["heterogeneity_level"] = pd.cut(
        ranking["i2"], bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["low", "moderate", "high", "very_high"],
    ).astype(str)
    priority = (
        (ranking["relation_stability_score"] >= 0.55)
        & (ranking["distribution_transfer_score"] >= 0.60)
        & (ranking["information_score"] >= 0.50)
        & (ranking["loco_sign_agreement"] >= 0.75)
        & (ranking["risk_score"] < 0.70)
    )
    conditional = (
        (ranking["relation_stability_score"] >= 0.45)
        & (ranking["distribution_transfer_score"] >= 0.50)
        & (ranking["information_score"] >= 0.20)
    )
    ranking["screening_tier"] = np.select(
        [priority, conditional],
        ["priority_validation", "conditional_validation"],
        default="low_priority",
    )
    ranking = ranking.sort_values(
        ["transfer_score", "relation_stability_score", "partial_r2"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.insert(0, "transfer_rank", np.arange(1, len(ranking) + 1))

    report_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(report_dir / "feature_transfer_ranking.csv", index=False)
    pd.DataFrame(loco_rows).to_csv(report_dir / "loco_stability.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(report_dir / "distribution_by_target_city.csv", index=False)
    pd.DataFrame(city_information_rows).to_csv(report_dir / "partial_r2_by_city.csv", index=False)
    _write_report(ranking, report_dir)
    metadata = {
        "panel_file": str(panel_path), "causal_run_dir": str(causal_run_dir),
        "report_dir": str(report_dir), "features_screened": len(specs),
        "bootstrap_iterations": bootstrap_iterations,
        "claim_boundary": "model-independent statistical transfer screen",
    }
    write_json(report_dir / "screening_metadata.json", metadata)
    return metadata


def describe_screening(config: Mapping, features: Optional[Iterable[str]] = None) -> Dict:
    panel = pd.read_parquet(Path(str(config["panel_file"])))
    registry = build_feature_registry(panel.columns, overrides=config.get("feature_overrides"))
    selected = set(features or ())
    unknown = sorted(selected - {spec.name for spec in registry})
    if unknown:
        raise ValueError(f"Unknown transfer-screening features: {', '.join(unknown)}")
    specs = [spec for spec in registry if spec.estimable and (not selected or spec.name in selected)]
    return {
        "rows": len(panel), "cities": int(panel["city_id"].nunique()),
        "periods": int(panel["period"].nunique()), "features": [spec.name for spec in specs],
        "bootstrap_iterations": int(config.get("bootstrap_iterations", 20)),
    }
