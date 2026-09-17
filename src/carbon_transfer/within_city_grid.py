from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd

from .config import project_path
from .utils import write_json


def _split_grid_ids(
    cell_ids: Sequence[str], ratios: Sequence[float], seed: int,
) -> Dict[str, list[str]]:
    values = sorted({str(value) for value in cell_ids})
    if len(ratios) != 3 or any(float(value) <= 0 for value in ratios):
        raise ValueError("split_ratios must contain three positive values")
    if not np.isclose(float(sum(ratios)), 1.0):
        raise ValueError("split_ratios must sum to 1")
    if len(values) < 3:
        raise ValueError("Within-city grid split requires at least three complete grids")

    shuffled = np.asarray(values, dtype=object)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    train_count = max(1, int(np.floor(len(shuffled) * float(ratios[0]))))
    validation_count = max(1, int(np.floor(len(shuffled) * float(ratios[1]))))
    if train_count + validation_count >= len(shuffled):
        validation_count = max(1, len(shuffled) - train_count - 1)
        train_count = len(shuffled) - validation_count - 1
    return {
        "train": shuffled[:train_count].tolist(),
        "validation": shuffled[train_count:train_count + validation_count].tolist(),
        "test": shuffled[train_count + validation_count:].tolist(),
    }


def build_within_city_grid_splits(
    config: Dict, write_manifest: Callable,
) -> Dict:
    """Build deterministic same-month spatial grid holdouts for each city."""
    panel = pd.read_parquet(project_path(config["panel_file"]), columns=[
        "city_id", "cell_id", "period", "admin_id", "admin_name",
    ])
    panel["city_id"] = panel["city_id"].astype(str)
    panel["cell_id"] = panel["cell_id"].astype(str)
    panel["period"] = panel["period"].astype(str)
    panel["admin_id"] = panel["admin_id"].astype("string")
    years = sorted(str(value) for value in config.get("years", ["2021", "2022", "2023"]))
    if len(years) != 3 or len(set(years)) != 3:
        raise ValueError("Stage1 within-city grid split requires exactly three distinct years")
    periods = [f"{year}{month:02d}" for year in years for month in range(1, 13)]
    cities = [str(value) for value in config["cities"]]
    split_seed = int(config.get("split_seed", 2026))
    ratios = [float(value) for value in config.get("split_ratios", [0.7, 0.2, 0.1])]
    experiment = str(config.get("experiment", "stage1_within_city_grid_2021_2023"))
    output_root = project_path(config["split_dir"])
    year_label = f"{years[0]}-{years[-1]}"
    audit = {
        "experiment": experiment,
        "years": years,
        "expected_periods": periods,
        "cities": cities,
        "split_seed": split_seed,
        "split_ratios": ratios,
        "folds": [],
    }

    for city_id in cities:
        city = panel[
            (panel["city_id"] == city_id) & panel["period"].isin(periods)
        ].copy()
        if city.empty:
            raise ValueError(f"No rows found for {city_id}/{year_label}")
        periods_per_grid = city.groupby("cell_id")["period"].agg(lambda values: set(values))
        complete_grids = periods_per_grid[
            periods_per_grid == set(periods)
        ].index.astype(str).tolist()
        if not complete_grids:
            raise ValueError(f"No complete 36-month grids for {city_id}/{year_label}")
        assignments = _split_grid_ids(complete_grids, ratios, split_seed)
        assignment_map = {
            cell_id: split
            for split, cell_ids in assignments.items()
            for cell_id in cell_ids
        }
        manifest = city[city["cell_id"].isin(complete_grids)].copy()
        manifest["split"] = manifest["cell_id"].map(assignment_map)
        if manifest["split"].isna().any():
            raise ValueError(f"Unassigned grid in {city_id}/{year_label}")
        fold_id = f"{year_label}_{city_id}_grid_fixed"
        entry = write_manifest(
            manifest,
            output_root / experiment / f"{fold_id}.parquet",
            experiment,
            fold_id,
        )
        entry.update({
            "city_id": city_id,
            "years": years,
            "split_seed": split_seed,
            "periods": periods,
            "complete_grids": len(complete_grids),
            "excluded_incomplete_grids": int(city["cell_id"].nunique() - len(complete_grids)),
            "target_ratios": dict(zip(("train", "validation", "test"), ratios)),
            "actual_ratios": {
                split: len(cell_ids) / len(complete_grids)
                for split, cell_ids in assignments.items()
            },
            "grid_ids": assignments,
        })
        audit["folds"].append(entry)

    write_json(output_root / f"{experiment}_audit.json", audit)
    return audit
