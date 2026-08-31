from pathlib import Path

import pandas as pd

from carbon_transfer.config import load_config
from carbon_transfer.evaluation import _summarize_three_year_yearly
from carbon_transfer.experiments import discover_tasks
from carbon_transfer.metrics import calculate_yearly_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_three_year_task_discovery_and_seed_filtering():
    config = load_config(ROOT / "configs/experiments/three_year_cross_region_2021_2023.yaml")
    tasks = discover_tasks(config)
    assert len(tasks) == 12
    assert {task.protocol for task in tasks} == {"three_year_cross_region"}
    assert {task.model_id for task in tasks} == {"opencarbon_monthly"}
    filtered = discover_tasks(config, seeds=[43])
    assert len(filtered) == 4
    assert {task.seed for task in filtered} == {43}


def test_real_three_year_manifests_are_complete_and_region_disjoint():
    root = ROOT / "data/splits/three_year_cross_region_2021_2023"
    manifests = sorted(root.glob("*.parquet"))
    if not manifests:
        return
    assert len(manifests) == 12
    expected = {f"{year}{month:02d}" for year in (2021, 2022, 2023) for month in range(1, 13)}
    test_admins = {}
    for path in manifests:
        frame = pd.read_parquet(path)
        assert set(frame["period"].astype(str)) == expected
        assert frame.groupby(["city_id", "cell_id"])["period"].nunique().eq(36).all()
        assert frame.groupby("admin_id")["split"].nunique().max() == 1
        assert frame.groupby(["city_id", "cell_id"])["split"].nunique().max() == 1
        grids = frame.drop_duplicates(["city_id", "cell_id"])
        ratios = grids["split"].value_counts(normalize=True)
        assert abs(ratios["train"] - 0.7) <= 0.02
        assert abs(ratios["validation"] - 0.2) <= 0.02
        assert abs(ratios["test"] - 0.1) <= 0.02
        prefix, seed = path.stem.rsplit("_split_seed_", 1)
        city = prefix.split("_", 1)[1]
        test_admins[(city, int(seed))] = frozenset(
            frame.loc[frame["split"] == "test", "admin_id"].astype(str)
        )
    for city in ("chicago", "nyc", "singapore", "tokyo"):
        assert len({test_admins[(city, seed)] for seed in (42, 43, 44)}) == 3


def test_yearly_metrics_preserve_three_years_and_month_counts():
    monthly = pd.DataFrame({
        "period": [f"{year}{month:02d}" for year in (2021, 2022, 2023) for month in range(1, 13)],
        "n": [10] * 36,
        "log_r2": [0.1] * 36,
        "log_mae": [0.2] * 36,
        "log_rmse": [0.3] * 36,
        "log_spearman": [0.4] * 36,
        "tc_mae": [1.0] * 36,
        "tc_rmse": [2.0] * 36,
    })
    yearly = calculate_yearly_metrics(monthly)
    assert yearly["year"].tolist() == ["2021", "2022", "2023"]
    assert yearly["months"].eq(12).all()
    assert yearly["n"].eq(120).all()


def test_yearly_protocol_summary_uses_city_macro_then_seed_statistics():
    rows = []
    for year in ("2021", "2022", "2023"):
        for seed in (42, 43, 44):
            rows.extend([
                {
                    "year": year, "city_id": "small", "seed": seed,
                    "test_grids": 10, "log_mae_mean": float(seed - 41),
                },
                {
                    "year": year, "city_id": "large", "seed": seed,
                    "test_grids": 30, "log_mae_mean": float(seed - 39),
                },
            ])
    city, macro, weighted, summary = _summarize_three_year_yearly(
        pd.DataFrame(rows), ["log_mae"],
    )
    assert len(city) == 6
    assert len(macro) == len(weighted) == 9
    assert len(summary) == 3
    seed_42 = macro[(macro["year"] == "2021") & (macro["seed"] == 42)].iloc[0]
    weighted_42 = weighted[
        (weighted["year"] == "2021") & (weighted["seed"] == 42)
    ].iloc[0]
    assert seed_42["log_mae"] == 2.0
    assert weighted_42["log_mae"] == 2.5
    assert summary.set_index("year").loc["2021", "log_mae_mean"] == 3.0
