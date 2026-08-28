from pathlib import Path

import numpy as np
import pandas as pd

from carbon_transfer.config import load_config
from carbon_transfer.datasets import OpenCarbonDataset
from carbon_transfer.experiments import discover_tasks
from carbon_transfer.metrics import calculate_annual_admin_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_annual_task_discovery_and_seed_filtering():
    config = load_config(ROOT / "configs/experiments/annual_cross_region_2022.yaml")
    tasks = discover_tasks(config)
    assert len(tasks) == 12
    assert {task.protocol for task in tasks} == {"annual_cross_region"}
    assert {task.model_id for task in tasks} == {"opencarbon_monthly"}
    filtered = discover_tasks(config, seeds=[43])
    assert len(filtered) == 4
    assert {task.seed for task in filtered} == {43}
    assert all(task.fold.endswith("_split_seed_43") for task in filtered)


def test_real_annual_manifests_are_complete_and_region_disjoint():
    root = ROOT / "data/splits/annual_cross_region_2022"
    manifests = sorted(root.glob("*.parquet"))
    if not manifests:
        return
    assert len(manifests) == 12
    test_admins = {}
    expected_periods = {f"2022{month:02d}" for month in range(1, 13)}
    for path in manifests:
        frame = pd.read_parquet(path)
        assert set(frame["period"].astype(str)) == expected_periods
        assert frame.groupby(["city_id", "cell_id"])["period"].nunique().eq(12).all()
        assert frame.groupby("admin_id")["split"].nunique().max() == 1
        assert frame.groupby(["city_id", "cell_id"])["split"].nunique().max() == 1
        grid_frame = frame.drop_duplicates(["city_id", "cell_id"])
        ratios = grid_frame["split"].value_counts(normalize=True)
        assert abs(ratios["train"] - 0.7) <= 0.02
        assert abs(ratios["validation"] - 0.2) <= 0.02
        assert abs(ratios["test"] - 0.1) <= 0.02
        prefix, seed = path.stem.rsplit("_split_seed_", 1)
        city = prefix.removeprefix("2022_")
        test_admins[(city, int(seed))] = frozenset(
            frame.loc[frame["split"] == "test", "admin_id"].astype(str)
        )
    for city in ("chicago", "nyc", "singapore", "tokyo"):
        assert len({test_admins[(city, seed)] for seed in (42, 43, 44)}) == 3


def test_annual_admin_metrics_preserve_all_months():
    rows = []
    for admin_id in ("a", "b"):
        for month in range(1, 13):
            for cell in range(5):
                value = float(month + cell)
                rows.append({
                    "admin_id": admin_id,
                    "cell_id": f"{admin_id}_{cell}",
                    "period": f"2022{month:02d}",
                    "y_log": value,
                    "pred_log": value + 0.1,
                    "y_tc": value,
                    "pred_tc": value + 0.1,
                })
    monthly, annual = calculate_annual_admin_metrics(pd.DataFrame(rows))
    assert len(monthly) == 24
    assert monthly.groupby("admin_id")["period"].nunique().eq(12).all()
    assert annual.set_index("admin_id")["months"].eq(12).all()
    assert annual.set_index("admin_id")["log_r2_valid_months"].eq(12).all()


def test_annual_open_carbon_dataset_uses_configurable_poi_cache():
    frame = pd.DataFrame({
        "city_id": ["x"], "period": ["202201"], "cell_id": ["a"],
        "log1p_emission": [0.0],
    })
    dataset = OpenCarbonDataset(
        frame,
        remote=np.zeros((1, 2), dtype=np.float32),
        environment=np.zeros((1, 3), dtype=np.float32),
        neighborhoods=np.asarray([[-1, -1, -1, -1, 0, -1, -1, -1, -1]]),
        poi_root=ROOT / "data/features/poi_sparse",
        poi_cache_size=12,
    )
    assert dataset.poi_store.cache_size == 12
