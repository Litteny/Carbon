from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from .config import project_path
from .single_month import balanced_admin_allocation
from .utils import write_json


def build_three_year_region_splits(config: Dict, write_manifest: Callable) -> Dict:
    """Build region-disjoint manifests spanning every month in three years."""
    panel = pd.read_parquet(project_path(config["panel_file"]), columns=[
        "city_id", "cell_id", "period", "admin_id", "admin_name", "eligible_cross_region",
    ])
    panel["period"] = panel["period"].astype(str)
    panel["admin_id"] = panel["admin_id"].astype("string")
    years = [str(value) for value in config["years"]]
    if len(years) != 3 or len(set(years)) != 3:
        raise ValueError("years must contain exactly three distinct years")
    years = sorted(years)
    expected_periods = [f"{year}{month:02d}" for year in years for month in range(1, 13)]
    cities = [str(city) for city in config["cities"]]
    seeds = [int(seed) for seed in config.get("split_seeds", [42, 43, 44])]
    ratios = [float(value) for value in config.get("split_ratios", [0.7, 0.2, 0.1])]
    experiment = str(config.get("experiment", f"three_year_cross_region_{years[0]}_{years[-1]}"))
    output_root = project_path(config["split_dir"])
    year_label = f"{years[0]}-{years[-1]}"
    audit = {
        "experiment": experiment,
        "years": years,
        "expected_periods": expected_periods,
        "split_seeds": seeds,
        "split_ratios": ratios,
        "folds": [],
    }

    for seed in seeds:
        for city_id in cities:
            all_city = panel[
                (panel["city_id"] == city_id) & panel["period"].isin(expected_periods)
            ].copy()
            eligible = all_city[
                all_city["eligible_cross_region"].eq(True) & all_city["admin_id"].notna()
            ].copy()
            if eligible.empty:
                raise ValueError(f"No eligible three-year samples for {city_id}/{year_label}")
            admin_stability = eligible.groupby("cell_id")["admin_id"].nunique()
            if int((admin_stability > 1).sum()):
                raise ValueError(f"Administrative regions change across years for {city_id}")
            periods_per_grid = eligible.groupby("cell_id")["period"].agg(lambda values: set(values))
            complete_grids = periods_per_grid[periods_per_grid == set(expected_periods)].index
            excluded_incomplete = int(eligible["cell_id"].nunique() - len(complete_grids))
            eligible = eligible[eligible["cell_id"].isin(complete_grids)].copy()
            if eligible.empty:
                raise ValueError(f"No complete 36-month grids for {city_id}/{year_label}")
            admin_counts = eligible.groupby("admin_id")["cell_id"].nunique().astype(int).to_dict()
            assignments, actual = balanced_admin_allocation(
                admin_counts, f"{experiment}:{city_id}", seed, ratios, seed_diversity=True,
            )
            eligible["split"] = eligible["admin_id"].map(assignments)
            if eligible["split"].isna().any():
                raise ValueError(f"Unassigned three-year region in {city_id}/{year_label}")
            fold_id = f"{year_label}_{city_id}_split_seed_{seed}"
            entry = write_manifest(
                eligible,
                output_root / experiment / f"{fold_id}.parquet",
                experiment,
                fold_id,
            )
            total_grids = int(eligible["cell_id"].nunique())
            entry.update({
                "city_id": city_id,
                "years": years,
                "split_seed": seed,
                "periods": expected_periods,
                "complete_grids": total_grids,
                "excluded_incomplete_grids": excluded_incomplete,
                "excluded_ineligible_rows": int(len(all_city) - len(panel[
                    (panel["city_id"] == city_id)
                    & panel["period"].isin(expected_periods)
                    & panel["eligible_cross_region"].eq(True)
                    & panel["admin_id"].notna()
                ])),
                "target_ratios": dict(zip(("train", "validation", "test"), ratios)),
                "actual_ratios": {name: count / total_grids for name, count in actual.items()},
                "admins": {
                    name: sorted(admin for admin, split in assignments.items() if split == name)
                    for name in ("train", "validation", "test")
                },
            })
            audit["folds"].append(entry)
    write_json(output_root / f"{experiment}_audit.json", audit)
    return audit
