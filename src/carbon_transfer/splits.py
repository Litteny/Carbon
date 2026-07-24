from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .config import project_path
from .utils import sha256_file, write_json


SPLIT_COLUMNS = ["city_id", "cell_id", "period", "admin_id", "admin_name", "split"]


def _stable_validation_admins(admin_ids: Iterable[str], city_id: str, seed: int, fraction: float) -> List[str]:
    values = sorted({str(value) for value in admin_ids if pd.notna(value)})
    if not values:
        return []
    count = max(1, int(math.ceil(len(values) * fraction)))
    ranked = sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{city_id}:{value}".encode("utf-8")).hexdigest(),
    )
    return ranked[:count]


def _write_manifest(frame: pd.DataFrame, path: Path, experiment: str, fold_id: str) -> Dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame[SPLIT_COLUMNS].sort_values(["split", "city_id", "period", "cell_id"]).reset_index(drop=True)
    if frame.duplicated(["city_id", "cell_id", "period"]).any():
        raise ValueError(f"Duplicate samples in split {fold_id}")
    consistency = frame.groupby(["city_id", "cell_id"])["split"].nunique()
    if int((consistency > 1).sum()):
        raise ValueError(f"Grid-month leakage detected in split {fold_id}")
    frame.to_parquet(path, index=False)
    counts = frame.groupby("split").size().to_dict()
    grid_counts = frame.groupby("split")["cell_id"].nunique().to_dict()
    return {
        "experiment": experiment,
        "fold_id": fold_id,
        "file": str(path),
        "sha256": sha256_file(path),
        "row_counts": {str(key): int(value) for key, value in counts.items()},
        "grid_counts": {str(key): int(value) for key, value in grid_counts.items()},
    }


def build_splits(config: Dict) -> Dict:
    panel_path = project_path(config["panel_file"])
    panel = pd.read_parquet(panel_path, columns=["city_id", "cell_id", "period", "admin_id", "admin_name"])
    panel["period"] = panel["period"].astype(str)
    panel["admin_id"] = panel["admin_id"].astype("string")
    output_root = project_path(config["split_dir"])
    seed = int(config.get("split_seed", 2026))
    fraction = float(config.get("validation_fraction", 0.2))
    audit = {"split_seed": seed, "validation_fraction": fraction, "folds": []}

    for target_city in config["cross_city_targets"]:
        manifest = panel.copy()
        manifest["split"] = "train"
        manifest.loc[manifest["city_id"] == target_city, "split"] = "test"
        validation_admins = {}
        for source_city in sorted(set(panel["city_id"]) - {target_city}):
            city_admins = panel.loc[
                (panel["city_id"] == source_city) & panel["admin_id"].notna(), "admin_id"
            ].unique()
            selected = _stable_validation_admins(city_admins, source_city, seed, fraction)
            validation_admins[source_city] = selected
            mask = (manifest["city_id"] == source_city) & manifest["admin_id"].isin(selected)
            manifest.loc[mask, "split"] = "validation"
        fold_id = f"target_{target_city}"
        entry = _write_manifest(
            manifest, output_root / "cross_city" / f"{fold_id}.parquet", "cross_city", fold_id,
        )
        entry["validation_admins"] = validation_admins
        audit["folds"].append(entry)

    for city_id, test_admin in config["cross_region_pilots"].items():
        city = panel[(panel["city_id"] == city_id) & panel["admin_id"].notna()].copy()
        if test_admin not in set(city["admin_id"].astype(str)):
            raise ValueError(f"Pilot admin {city_id}/{test_admin} is absent from the panel")
        remaining = city.loc[city["admin_id"] != test_admin, "admin_id"].unique()
        validation_admins = _stable_validation_admins(remaining, city_id, seed, fraction)
        city["split"] = "train"
        city.loc[city["admin_id"].isin(validation_admins), "split"] = "validation"
        city.loc[city["admin_id"] == test_admin, "split"] = "test"
        fold_id = f"{city_id}_{test_admin}"
        entry = _write_manifest(
            city, output_root / "cross_region" / f"{fold_id}.parquet", "cross_region", fold_id,
        )
        entry["test_admin"] = test_admin
        entry["validation_admins"] = validation_admins
        audit["folds"].append(entry)

    write_json(output_root / "split_audit.json", audit)
    return audit
