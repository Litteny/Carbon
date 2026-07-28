from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .config import project_path
from .utils import write_json


def _stable_admin_order(admin_counts: Mapping[str, int], city_id: str, seed: int) -> List[str]:
    return sorted(
        admin_counts,
        key=lambda admin_id: (
            -int(admin_counts[admin_id]),
            hashlib.sha256(f"{seed}:{city_id}:{admin_id}".encode("utf-8")).hexdigest(),
        ),
    )


def _allocation_error(
    assignments: Mapping[str, str], admin_counts: Mapping[str, int], targets: Mapping[str, float],
) -> float:
    totals = {name: 0 for name in targets}
    for admin_id, split in assignments.items():
        totals[split] += int(admin_counts[admin_id])
    return float(sum(abs(totals[name] - targets[name]) for name in targets))


def balanced_admin_allocation(
    admin_counts: Mapping[str, int], city_id: str, seed: int, ratios: Sequence[float],
    seed_diversity: bool = False,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Allocate whole administrative regions near target grid-count ratios."""
    names = ("train", "validation", "test")
    if len(ratios) != 3 or any(float(value) <= 0 for value in ratios):
        raise ValueError("split_ratios must contain three positive values")
    ratio_total = float(sum(ratios))
    normalized = {name: float(value) / ratio_total for name, value in zip(names, ratios)}
    if len(admin_counts) < 3:
        raise ValueError(f"{city_id} needs at least three eligible administrative regions")
    total = int(sum(admin_counts.values()))
    targets = {name: normalized[name] * total for name in names}
    order = (
        sorted(
            admin_counts,
            key=lambda admin_id: hashlib.sha256(
                f"{seed}:{city_id}:{admin_id}".encode("utf-8")
            ).hexdigest(),
        )
        if seed_diversity else _stable_admin_order(admin_counts, city_id, seed)
    )
    assignments: Dict[str, str] = {}
    totals = {name: 0 for name in names}
    for index, admin_id in enumerate(order):
        if index < len(names):
            split = names[index]
        else:
            split = min(
                names,
                key=lambda name: (
                    (totals[name] + int(admin_counts[admin_id]) - targets[name]) ** 2
                    - (totals[name] - targets[name]) ** 2,
                    names.index(name),
                ),
            )
        assignments[admin_id] = split
        totals[split] += int(admin_counts[admin_id])

    while True:
        baseline = _allocation_error(assignments, admin_counts, targets)
        best_error = baseline
        best_operation = None
        split_counts = pd.Series(assignments).value_counts().to_dict()
        for admin_id in order:
            source = assignments[admin_id]
            if split_counts.get(source, 0) <= 1:
                continue
            for destination in names:
                if destination == source:
                    continue
                candidate = dict(assignments)
                candidate[admin_id] = destination
                error = _allocation_error(candidate, admin_counts, targets)
                operation = ("move", admin_id, destination)
                if error < best_error - 1e-9 or (
                    abs(error - best_error) <= 1e-9
                    and best_operation is not None and operation < best_operation
                ):
                    best_error, best_operation = error, operation
        for left_index, left in enumerate(order):
            for right in order[left_index + 1:]:
                if assignments[left] == assignments[right]:
                    continue
                candidate = dict(assignments)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                error = _allocation_error(candidate, admin_counts, targets)
                operation = ("swap", left, right)
                if error < best_error - 1e-9 or (
                    abs(error - best_error) <= 1e-9
                    and best_operation is not None and operation < best_operation
                ):
                    best_error, best_operation = error, operation
        if best_operation is None or best_error >= baseline - 1e-9:
            break
        if best_operation[0] == "move":
            _, admin_id, destination = best_operation
            assignments[admin_id] = destination
        else:
            _, left, right = best_operation
            assignments[left], assignments[right] = assignments[right], assignments[left]

    actual = {
        name: int(sum(admin_counts[admin] for admin, split in assignments.items() if split == name))
        for name in names
    }
    return assignments, actual


def build_single_month_splits(config: Dict, write_manifest: Callable) -> Dict:
    panel = pd.read_parquet(project_path(config["panel_file"]), columns=[
        "city_id", "cell_id", "period", "admin_id", "admin_name", "eligible_cross_region",
    ])
    panel["period"] = panel["period"].astype(str)
    panel["admin_id"] = panel["admin_id"].astype("string")
    period = str(config["period"])
    cities = [str(city) for city in config["cities"]]
    seeds = [int(value) for value in config.get("split_seeds", [config.get("split_seed", 42)])]
    ratios = [float(value) for value in config.get("split_ratios", [0.7, 0.2, 0.1])]
    output_root = project_path(config["split_dir"])
    experiment = str(config.get("experiment", "single_month_cross_region"))
    seed_diversity = bool(config.get("seed_diversity", False))
    audit = {
        "experiment": experiment, "period": period,
        "split_seeds": seeds, "split_ratios": ratios, "folds": [],
    }
    for seed in seeds:
        for city_id in cities:
            all_city = panel[(panel["city_id"] == city_id) & (panel["period"] == period)].copy()
            eligible = all_city[
                all_city["eligible_cross_region"].eq(True) & all_city["admin_id"].notna()
            ].copy()
            if eligible.empty:
                raise ValueError(f"No eligible samples for {city_id}/{period}")
            counts = eligible.groupby("admin_id").size().astype(int).to_dict()
            assignments, actual = balanced_admin_allocation(
                counts, city_id, seed, ratios, seed_diversity=seed_diversity,
            )
            eligible["split"] = eligible["admin_id"].map(assignments)
            if eligible["split"].isna().any():
                raise ValueError(f"Unassigned administrative region in {city_id}/{period}")
            fold_id = (
                f"{period}_{city_id}" if len(seeds) == 1
                else f"{period}_{city_id}_split_seed_{seed}"
            )
            entry = write_manifest(
                eligible, output_root / experiment / f"{fold_id}.parquet",
                experiment, fold_id,
            )
            total = len(eligible)
            entry.update({
                "city_id": city_id, "period": period, "split_seed": seed,
                "excluded_samples": int(len(all_city) - total),
                "target_ratios": dict(zip(("train", "validation", "test"), ratios)),
                "actual_ratios": {name: count / total for name, count in actual.items()},
                "admins": {
                    name: sorted(admin for admin, split in assignments.items() if split == name)
                    for name in ("train", "validation", "test")
                },
            })
            audit["folds"].append(entry)
    write_json(output_root / f"{experiment}_audit.json", audit)
    return audit
