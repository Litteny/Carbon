from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import project_path
from .constants import (
    MISSING_COLUMNS, MODIS_COLUMNS, MONTH_COLUMNS, POI_COLUMNS,
    VIIRS_COLUMNS, WEATHER_COLUMNS,
)
from .utils import write_json


KEYS = ["cell_id", "period"]
GRID_RESOLUTION = 1.0 / 120.0


def _read_many(files: Iterable[Path], columns: List[str] = None) -> pd.DataFrame:
    paths = sorted(files)
    if not paths:
        raise FileNotFoundError("No input files matched")
    frames = [pd.read_csv(path, usecols=columns) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    result["cell_id"] = result["cell_id"].astype(str)
    result["period"] = result["period"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    return result


def _assert_unique(frame: pd.DataFrame, keys: List[str], label: str) -> None:
    duplicate_count = int(frame.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(f"{label} contains {duplicate_count} duplicate keys: {keys}")


def build_panel(config: Dict) -> Tuple[pd.DataFrame, Dict]:
    raw_root = project_path(config["raw_root"])
    feature_dir = project_path(config["feature_dir"])
    feature_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(
        project_path(config["mapping_file"]),
        dtype={"city_id": str, "cell_id": str, "admin_id": str},
    )
    mapping["city_id"] = mapping["city_id"].str.lower()
    mapping = mapping[[
        "city_id", "cell_id", "admin_id", "admin_name", "admin_type",
        "assignment_method", "eligible_cross_region",
    ]]
    _assert_unique(mapping, ["city_id", "cell_id"], "admin mapping")

    city_frames = []
    city_audit = {}
    for city_id, directory_name in config["cities"].items():
        city_dir = raw_root / directory_name
        emissions = _read_many(
            (city_dir / "CO2").glob("emission_panel_*.csv"),
            ["cell_id", "row", "col", "lon", "lat", "period", "emission_tc"],
        )
        _assert_unique(emissions, KEYS, f"{city_id} emissions")

        poi = _read_many(
            (city_dir / "POI").glob("poi_grid_*.csv"),
            KEYS + POI_COLUMNS,
        )
        modis = _read_many(
            (city_dir / "modis").glob("rs_grid_*.csv"),
            KEYS + MODIS_COLUMNS,
        )
        viirs = _read_many(
            (city_dir / "VIIRS").glob("*.csv"),
            KEYS + VIIRS_COLUMNS,
        )
        weather = _read_many(
            (city_dir / "weather").glob("weather_openmeteo_monthly_*.csv"),
            KEYS + WEATHER_COLUMNS,
        )
        for label, frame in (("POI", poi), ("MODIS", modis), ("VIIRS", viirs), ("weather", weather)):
            _assert_unique(frame, KEYS, f"{city_id} {label}")

        panel = emissions
        for frame in (poi, modis, viirs, weather):
            panel = panel.merge(frame, on=KEYS, how="left", validate="one_to_one")
        panel.insert(0, "city_id", city_id)
        panel = panel.merge(mapping, on=["city_id", "cell_id"], how="left", validate="many_to_one")

        for source_column, flag_column in zip(MODIS_COLUMNS + WEATHER_COLUMNS, MISSING_COLUMNS):
            panel[flag_column] = panel[source_column].isna().astype("int8")
        month = panel["period"].str[-2:].astype(int)
        for month_number, column in enumerate(MONTH_COLUMNS, start=1):
            panel[column] = (month == month_number).astype("int8")
        panel["log1p_emission"] = np.log1p(panel["emission_tc"].clip(lower=0))

        city_frames.append(panel)
        city_audit[city_id] = {
            "rows": len(panel),
            "grids": int(panel["cell_id"].nunique()),
            "periods": sorted(panel["period"].unique().tolist()),
            "unassigned_grids": int(panel.loc[panel["admin_id"].isna(), "cell_id"].nunique()),
        }

    result = pd.concat(city_frames, ignore_index=True)
    result = result.sort_values(["city_id", "period", "row", "col"]).reset_index(drop=True)
    _assert_unique(result, ["city_id", "cell_id", "period"], "unified panel")
    expected_rows = int(config.get("expected_rows", 0))
    if expected_rows and len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} panel rows, found {len(result)}")
    if result["period"].nunique() != 36:
        raise ValueError(f"Expected 36 periods, found {result['period'].nunique()}")

    output = feature_dir / "grid_month_panel.parquet"
    result.to_parquet(output, index=False)
    audit = {"panel_file": str(output), "rows": len(result), "cities": city_audit}
    write_json(feature_dir / "dataset_audit.json", audit)
    return result, audit


def build_neighbors(panel: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    records = []
    unique_grids = panel[["city_id", "cell_id", "row", "col"]].drop_duplicates()
    periods = sorted(panel["period"].astype(str).unique())
    for city_id, grids in unique_grids.groupby("city_id", sort=True):
        lookup = {(int(row.row), int(row.col)): str(row.cell_id) for row in grids.itertuples()}
        static_edges = []
        for row in grids.itertuples():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = lookup.get((int(row.row) + dr, int(row.col) + dc))
                    if neighbor is not None:
                        static_edges.append((str(row.cell_id), neighbor, dr, dc))
        for period in periods:
            records.extend((city_id, period, *edge) for edge in static_edges)
    neighbors = pd.DataFrame(records, columns=[
        "city_id", "period", "cell_id", "neighbor_cell_id", "offset_row", "offset_col",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    neighbors.to_parquet(output_path, index=False)
    return neighbors


def _grid_lookup(city_dir: Path) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], Tuple[str, float, float]]]:
    grid_file = next(city_dir.glob("*_grid_1km_odiac.geojson"))
    import geopandas as gpd
    grids = gpd.read_file(grid_file)[["cell_id", "row", "col", "lon", "lat"]]
    grids["cell_id"] = grids["cell_id"].astype(str)
    lookup = {
        (int(row.row), int(row.col)): (str(row.cell_id), float(row.lon), float(row.lat))
        for row in grids.itertuples()
    }
    return grids, lookup


def _map_poi_points(points: pd.DataFrame, grids: pd.DataFrame, lookup: Dict) -> pd.DataFrame:
    x_origin = float(np.median(grids["lon"] - grids["col"] * GRID_RESOLUTION))
    y_origin = float(np.median(grids["lat"] + grids["row"] * GRID_RESOLUTION))
    cols = np.floor((points["lon"].to_numpy() - x_origin) / GRID_RESOLUTION + 0.5).astype(np.int32)
    rows = np.floor((y_origin - points["lat"].to_numpy()) / GRID_RESOLUTION + 0.5).astype(np.int32)
    mapped = [lookup.get((int(row), int(col))) for row, col in zip(rows, cols)]
    keep = np.fromiter((item is not None for item in mapped), dtype=bool, count=len(mapped))
    result = points.loc[keep, ["lon", "lat", "category"]].copy()
    selected = [item for item in mapped if item is not None]
    result["cell_id"] = [item[0] for item in selected]
    center_lon = np.asarray([item[1] for item in selected])
    center_lat = np.asarray([item[2] for item in selected])
    min_lon = center_lon - GRID_RESOLUTION / 2
    max_lat = center_lat + GRID_RESOLUTION / 2
    result["pixel_x"] = np.clip(
        np.floor((result["lon"].to_numpy() - min_lon) / GRID_RESOLUTION * 256), 0, 255,
    ).astype(np.int16)
    result["pixel_y"] = np.clip(
        np.floor((max_lat - result["lat"].to_numpy()) / GRID_RESOLUTION * 256), 0, 255,
    ).astype(np.int16)
    return result


def build_sparse_poi(config: Dict) -> Dict:
    raw_root = project_path(config["raw_root"])
    output_root = project_path(config["feature_dir"]) / "poi_sparse"
    output_root.mkdir(parents=True, exist_ok=True)
    category_to_channel = {name: index for index, name in enumerate(POI_COLUMNS)}
    audit = {"categories": category_to_channel, "cities": {}, "mismatches": []}

    for city_id, directory_name in config["cities"].items():
        city_dir = raw_root / directory_name
        grids, lookup = _grid_lookup(city_dir)
        cell_ids = np.asarray(
            grids.sort_values(["row", "col"])["cell_id"].astype(str).tolist(), dtype="U32",
        )
        cell_to_index = {cell_id: index for index, cell_id in enumerate(cell_ids)}
        city_output = output_root / city_id
        city_output.mkdir(parents=True, exist_ok=True)
        monthly_audit = []

        for points_file in sorted((city_dir / "poi_points").glob("poi_points_*.csv")):
            period = points_file.stem.rsplit("_", 1)[-1]
            points = pd.read_csv(points_file, usecols=["lon", "lat", "category"])
            unknown = sorted(set(points["category"].dropna()) - set(category_to_channel))
            if unknown:
                raise ValueError(f"Unknown POI categories in {points_file}: {unknown}")
            mapped = _map_poi_points(points, grids, lookup)
            mapped["channel"] = mapped["category"].map(category_to_channel).astype(np.int16)
            mapped["cell_index"] = mapped["cell_id"].map(cell_to_index).astype(np.int32)
            grouped = (
                mapped.groupby(["cell_index", "pixel_y", "pixel_x", "channel"], sort=True)
                .size().rename("count").reset_index()
            )
            counts_per_cell = grouped.groupby("cell_index").size().reindex(range(len(cell_ids)), fill_value=0)
            offsets = np.concatenate(([0], np.cumsum(counts_per_cell.to_numpy(dtype=np.int64))))
            np.savez_compressed(
                city_output / f"{period}.npz",
                cell_id=cell_ids,
                offsets=offsets,
                pixel_y=grouped["pixel_y"].to_numpy(dtype=np.int16),
                pixel_x=grouped["pixel_x"].to_numpy(dtype=np.int16),
                channel=grouped["channel"].to_numpy(dtype=np.int16),
                count=grouped["count"].to_numpy(dtype=np.int32),
            )

            expected_file = city_dir / "POI" / f"poi_grid_{period}.csv"
            expected = pd.read_csv(expected_file, usecols=["cell_id"] + POI_COLUMNS).set_index("cell_id")
            actual = mapped.groupby(["cell_id", "category"]).size().unstack(fill_value=0)
            actual = actual.reindex(index=expected.index, columns=POI_COLUMNS, fill_value=0)
            difference = actual.to_numpy(dtype=np.int64) - expected[POI_COLUMNS].to_numpy(dtype=np.int64)
            mismatch_count = int(np.count_nonzero(difference))
            absolute_difference = int(np.abs(difference).sum())
            expected_count = int(expected[POI_COLUMNS].to_numpy(dtype=np.int64).sum())
            discrepant_fraction = absolute_difference / max(expected_count, 1)
            if mismatch_count:
                audit["mismatches"].append({
                    "city_id": city_id, "period": period, "cells_categories": mismatch_count,
                    "maximum_absolute_difference": int(np.abs(difference).max()),
                    "absolute_count_difference": absolute_difference,
                    "discrepant_count_fraction": discrepant_fraction,
                })
            monthly_audit.append({
                "period": period, "source_points": len(points), "mapped_points": len(mapped),
                "sparse_entries": len(grouped), "count_mismatches": mismatch_count,
                "absolute_count_difference": absolute_difference,
                "discrepant_count_fraction": discrepant_fraction,
            })
        audit["cities"][city_id] = monthly_audit

    tolerance = config.get("poi_verification", {})
    maximum = int(tolerance.get("max_absolute_difference", 0))
    fraction = float(tolerance.get("max_discrepant_count_fraction", 0.0))
    failures = [
        item for item in audit["mismatches"]
        if item["maximum_absolute_difference"] > maximum
        or item["discrepant_count_fraction"] > fraction
    ]
    audit["verification_status"] = "PASS" if not audit["mismatches"] else ("WARN" if not failures else "FAIL")
    audit["verification_failures"] = failures
    write_json(project_path(config["feature_dir"]) / "poi_audit.json", audit)
    if failures:
        raise ValueError(f"POI verification exceeded tolerance for {len(failures)} city-month files")
    return audit


def build_all(config: Dict, skip_poi: bool = False) -> Dict:
    panel, audit = build_panel(config)
    neighbors = build_neighbors(panel, project_path(config["feature_dir"]) / "neighbors.parquet")
    audit["neighbor_edges"] = len(neighbors)
    if not skip_poi:
        audit["poi"] = build_sparse_poi(config)
    write_json(project_path(config["feature_dir"]) / "dataset_audit.json", audit)
    return audit
