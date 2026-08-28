from __future__ import annotations

import numpy as np
import pandas as pd

from carbon_transfer.transfer_features.distribution import distribution_audit
from carbon_transfer.transfer_features.domain import domain_classification
from carbon_transfer.transfer_features.information import partial_r2_audit
from carbon_transfer.transfer_features.meta import summarize_city_effects


def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for city_index, city in enumerate(("a", "b", "c", "d")):
        for cell in range(20):
            entity = rng.normal()
            for month in range(12):
                stable = rng.normal()
                city_specific = city_index * 4.0 + rng.normal(scale=0.2)
                control = rng.normal()
                outcome = entity + month * 0.1 + 1.2 * stable + 0.3 * control + rng.normal(scale=0.1)
                rows.append({
                    "city_id": city, "cell_id": f"{city}_{cell}",
                    "period": f"2022{month + 1:02d}", "log1p_emission": outcome,
                    "stable": stable, "city_specific": city_specific, "control": control,
                })
    return pd.DataFrame(rows)


def test_meta_analysis_reports_stable_loco_direction():
    city_rows = [
        {"city_id": city, "identified": True, "effect_per_within_sd": effect,
         "effect_se_per_within_sd": 0.1}
        for city, effect in zip(("a", "b", "c", "d"), (1.0, 1.1, 0.9, 1.05))
    ]
    summary, loco = summarize_city_effects("stable", city_rows)
    assert summary["i2"] < 0.25
    assert summary["loco_sign_agreement"] == 1.0
    assert len(loco) == 4


def test_distribution_and_domain_detect_city_specific_feature():
    panel = synthetic_panel()
    stable, _ = distribution_audit(panel, "stable")
    shifted, _ = distribution_audit(panel, "city_specific")
    assert stable["mean_support_overlap"] > shifted["mean_support_overlap"]
    stable_domain = domain_classification(panel, "stable", max_samples_per_city=1000)
    shifted_domain = domain_classification(panel, "city_specific", max_samples_per_city=1000)
    assert shifted_domain["domain_balanced_accuracy"] > 0.9
    assert stable_domain["domain_balanced_accuracy"] < 0.4


def test_partial_r2_detects_independent_information():
    panel = synthetic_panel()
    stable, cities = partial_r2_audit(
        panel, "stable", "log1p_emission", ["control"], bootstrap_iterations=5,
    )
    shifted, _ = partial_r2_audit(
        panel, "city_specific", "log1p_emission", ["control"], bootstrap_iterations=5,
    )
    assert stable["partial_r2"] > 0.8
    assert stable["partial_r2"] > shifted["partial_r2"]
    assert stable["bootstrap_relevance_frequency"] == 1.0
    assert len(cities) == 4
