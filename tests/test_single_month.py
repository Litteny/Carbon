from pathlib import Path

import pandas as pd

from carbon_transfer.metrics import calculate_admin_metrics, calculate_metrics
from carbon_transfer.single_month import balanced_admin_allocation


ROOT = Path(__file__).resolve().parents[1]


def test_balanced_admin_allocation_is_deterministic_and_disjoint():
    counts = {"a": 40, "b": 30, "c": 15, "d": 10, "e": 5}
    first, totals = balanced_admin_allocation(counts, "city", 42, [0.7, 0.2, 0.1])
    second, _ = balanced_admin_allocation(counts, "city", 42, [0.7, 0.2, 0.1])
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    assert sum(totals.values()) == sum(counts.values())
    assert all(abs(totals[name] / 100 - target) <= 0.1 for name, target in {
        "train": 0.7, "validation": 0.2, "test": 0.1,
    }.items())


def test_single_month_metrics_have_no_month_variance_and_admin_boundaries():
    predictions = pd.DataFrame({
        "admin_id": ["a"] * 5 + ["b"] * 4,
        "period": ["202208"] * 9,
        "y_log": list(range(9)), "pred_log": list(range(9)),
        "y_tc": list(range(9)), "pred_tc": list(range(9)),
    })
    _, summary = calculate_metrics(predictions)
    admins = calculate_admin_metrics(predictions)
    assert summary["metrics"]["log_mae"]["std_across_months"] is None
    assert admins.set_index("admin_id").loc["a", "r2_status"] == "ok"
    assert admins.set_index("admin_id").loc["b", "r2_status"].startswith("undefined")


def test_real_single_month_manifests_are_region_disjoint():
    manifests = sorted((ROOT / "data/splits/single_month_cross_region").glob("*.parquet"))
    if not manifests:
        return
    assert len(manifests) == 4
    for path in manifests:
        frame = pd.read_parquet(path)
        assert set(frame["period"].astype(str)) == {"202208"}
        assert set(frame["split"]) == {"train", "validation", "test"}
        assert frame.groupby("admin_id")["split"].nunique().max() == 1
        ratios = frame["split"].value_counts(normalize=True)
        assert abs(ratios["train"] - 0.7) <= 0.02
        assert abs(ratios["validation"] - 0.2) <= 0.02
        assert abs(ratios["test"] - 0.1) <= 0.02


def test_real_multiseed_manifests_are_diverse_and_region_disjoint():
    root = ROOT / "data/splits/single_month_cross_region_multiseed"
    manifests = sorted(root.glob("*.parquet"))
    if not manifests:
        return
    assert len(manifests) == 12
    test_admins = {}
    for path in manifests:
        frame = pd.read_parquet(path)
        assert set(frame["period"].astype(str)) == {"202208"}
        assert frame.groupby("admin_id")["split"].nunique().max() == 1
        ratios = frame["split"].value_counts(normalize=True)
        assert abs(ratios["train"] - 0.7) <= 0.02
        assert abs(ratios["validation"] - 0.2) <= 0.02
        assert abs(ratios["test"] - 0.1) <= 0.02
        prefix, seed = path.stem.rsplit("_split_seed_", 1)
        city = prefix.removeprefix("202208_")
        test_admins[(city, int(seed))] = frozenset(
            frame.loc[frame["split"] == "test", "admin_id"].astype(str)
        )
    for city in ("chicago", "nyc", "singapore", "tokyo"):
        assert len({test_admins[(city, seed)] for seed in (42, 43, 44)}) == 3
