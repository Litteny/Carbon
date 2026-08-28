from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carbon_transfer.causal.analysis import analyze_feature, run_feature_causal_analysis
from carbon_transfer.causal.panel import add_shifted_feature
from carbon_transfer.causal.registry import FeatureSpec, build_feature_registry


ROOT = Path(__file__).resolve().parents[1]


def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for city_index, city in enumerate(("a", "b")):
        for cell_index in range(8):
            entity_effect = rng.normal()
            for month in range(12):
                weather = rng.normal()
                outcome = entity_effect + 0.2 * month + 1.5 * weather + rng.normal(scale=0.05)
                rows.append({
                    "city_id": city, "cell_id": str(cell_index),
                    "period": f"2022{month + 1:02d}", "log1p_emission": outcome,
                    "temperature_2m_mean_c": weather,
                    "ntl_radiance_mean": outcome + rng.normal(scale=0.01),
                    "modis_ndvi_mean_pixel_count": 100,
                    "month_01": int(month == 0),
                })
    return pd.DataFrame(rows)


def test_registry_excludes_measurement_and_calendar_features():
    specs = build_feature_registry(synthetic_panel().columns)
    assert len(specs) == len({spec.name for spec in specs})
    registry = {spec.name: spec for spec in specs}
    assert registry["temperature_2m_mean_c"].role == "external_exposure"
    assert registry["temperature_2m_mean_c"].estimable
    assert not registry["modis_ndvi_mean_pixel_count"].estimable
    assert not registry["month_01"].estimable


def test_shifted_feature_respects_city_and_grid_boundaries():
    panel = synthetic_panel()
    shifted = add_shifted_feature(panel, "temperature_2m_mean_c", 1)
    first = shifted.groupby(["city_id", "cell_id"], sort=False).head(1)
    assert first["treatment"].isna().all()


def test_fixed_effect_analysis_recovers_positive_effect():
    panel = synthetic_panel()
    spec = FeatureSpec(
        "temperature_2m_mean_c", "weather", "external_exposure", True,
        "one within-panel standard-deviation increase", (0,), "test",
    )
    result = analyze_feature(
        panel, spec, outcome="log1p_emission", lag=0, controls=(),
    )
    estimate = result["estimate"]
    assert estimate["identified"]
    assert 1.35 < estimate["coefficient"] < 1.65
    assert estimate["p_value"] < 0.001


def test_analysis_writes_report_and_downgrades_viirs(tmp_path):
    panel_path = tmp_path / "panel.parquet"
    synthetic_panel().to_parquet(panel_path, index=False)
    config = {
        "panel_file": str(panel_path), "outcome": "log1p_emission",
        "artifact_dir": str(tmp_path / "runs"), "report_dir": str(tmp_path / "reports"),
        "lags": [0, -1], "controls": {"default": []},
        "label_feature_sources": [], "placebo_iterations": 0,
    }
    metadata = run_feature_causal_analysis(
        config, features=["ntl_radiance_mean"],
    )
    assert metadata["estimates"] == 2
    evidence = pd.read_csv(tmp_path / "reports" / "feature_evidence_table.csv")
    contemporaneous = evidence[evidence["lag"] == 0].iloc[0]
    assert contemporaneous["evidence_grade"] == "D"
    assert (tmp_path / "reports" / "causal_analysis_report.md").exists()


def test_analysis_rejects_measurement_only_selection(tmp_path):
    panel_path = tmp_path / "panel.parquet"
    synthetic_panel().to_parquet(panel_path, index=False)
    config = {
        "panel_file": str(panel_path), "outcome": "log1p_emission",
        "artifact_dir": str(tmp_path / "runs"), "report_dir": str(tmp_path / "reports"),
    }
    with pytest.raises(ValueError, match="no causal estimand"):
        run_feature_causal_analysis(
            config, features=["modis_ndvi_mean_pixel_count"],
        )


def test_causal_script_dry_run_lists_tasks(capsys):
    path = ROOT / "scripts" / "run_feature_causal_analysis.py"
    spec = spec_from_file_location("run_feature_causal_analysis", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([
        "--features", "temperature_2m_mean_c", "--lags", "0", "-1", "--dry-run",
    ]) == 0
    output = capsys.readouterr().out
    assert "feature=temperature_2m_mean_c" in output
    assert "tasks=2" in output
