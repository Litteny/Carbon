from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


KEY_COLUMNS = ("city_id", "cell_id", "period")


def validate_panel(frame: pd.DataFrame, outcome: str) -> None:
    required = set(KEY_COLUMNS) | {outcome}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Causal panel is missing columns: {', '.join(missing)}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Causal panel contains duplicate city-grid-period rows")
    periods = frame["period"].astype(str)
    if not periods.str.fullmatch(r"\d{6}").all():
        raise ValueError("period must use YYYYMM format")


def add_shifted_feature(
    frame: pd.DataFrame,
    feature: str,
    lag: int,
    output_column: str = "treatment",
) -> pd.DataFrame:
    """Add X[t-lag]; a negative lag is a future-feature negative control."""
    result = frame.sort_values(list(KEY_COLUMNS)).copy()
    result[output_column] = result.groupby(
        ["city_id", "cell_id"], sort=False, observed=True,
    )[feature].shift(lag)
    return result


def within_standard_deviation(
    values: pd.Series,
    entities: pd.Series,
) -> float:
    centered = values.astype(float) - values.astype(float).groupby(entities).transform("mean")
    result = float(centered.std(ddof=1))
    return result if np.isfinite(result) else 0.0


def select_controls(
    family: str,
    treatment: str,
    available: Iterable[str],
    configured: dict,
) -> List[str]:
    candidates: Sequence[str] = configured.get(family, configured.get("default", ()))
    available_set = set(available)
    return [str(value) for value in candidates if value != treatment and value in available_set]
