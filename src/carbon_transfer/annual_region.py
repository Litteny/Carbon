from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from .config import project_path
from .single_month import balanced_admin_allocation
from .utils import write_json


def build_annual_region_splits(config: Dict, write_manifest: Callable) -> Dict:
    panel = pd.read_parquet(project_path(config["panel_file"]), columns=[
        "city_id", "cell_id", "period", "admin_id", "admin_name", "eligible_cross_region",
    ])
    panel["period"] = panel["period"].astype(str)
    panel["admin_id"] = panel["admin_id"].astype("string")
    year = str(config["year"])
    expected_periods = [f"{year}{month:02d}" for month in range(1, 13)]
    cities = [str(city) for city in config["cities"]]
    seeds = [int(seed) for seed in config.get("split_seeds", [42, 43, 44])]
    ratios = [float(value) for value in config.get("split_ratios", [0.7, 0.2, 0.1])]
    experiment = str(config.get("experiment", f"annual_cross_region_{year}"))
    output_root = project_path(config["split_dir"])
    audit = {
        "experiment": experiment,
        "year": year,
        "expected_periods": expected_periods,
        "split_seeds": seeds,
        "split_ratios": ratios,
        "folds": [],
    }

    for seed in seeds:
        for city_id in cities:
            city = panel[
                (panel["city_id"] == city_id)
                & panel["period"].isin(expected_periods)
                & panel["eligible_cross_region"].eq(True)
                & panel["admin_id"].notna()
            ].copy()
            if city.empty:
                raise ValueError(f"No eligible annual samples for {city_id}/{year}")
            periods_per_grid = city.groupby("cell_id")["period"].agg(lambda values: set(values))
            complete_grids = periods_per_grid[periods_per_grid == set(expected_periods)].index
            excluded_incomplete = int(city["cell_id"].nunique() - len(complete_grids))
            city = city[city["cell_id"].isin(complete_grids)].copy()
            if city.empty:
                raise ValueError(f"No complete 12-month grids for {city_id}/{year}")
            admin_counts = city.groupby("admin_id")["cell_id"].nunique().astype(int).to_dict()
            assignments, actual = balanced_admin_allocation(
                admin_counts,
                f"{experiment}:{city_id}",
                seed,
                ratios,
                seed_diversity=True,
            )
            city["split"] = city["admin_id"].map(assignments)
            if city["split"].isna().any():
                raise ValueError(f"Unassigned annual administrative region in {city_id}/{year}")
            fold_id = f"{year}_{city_id}_split_seed_{seed}"
            entry = write_manifest(
                city,
                output_root / experiment / f"{fold_id}.parquet",
                experiment,
                fold_id,
            )
            total_grids = int(city["cell_id"].nunique())
            entry.update({
                "city_id": city_id,
                "year": year,
                "split_seed": seed,
                "periods": expected_periods,
                "complete_grids": total_grids,
                "excluded_incomplete_grids": excluded_incomplete,
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
