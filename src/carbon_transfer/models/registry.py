from __future__ import annotations

from typing import Dict, Iterable

from carbon_transfer.constants import MODEL_NAMES

from .base import ModelSpec


_MODEL_SPECS: Dict[str, ModelSpec] = {
    "lightgbm": ModelSpec("lightgbm", "lightgbm"),
    "bpnn": ModelSpec("bpnn", "bpnn"),
    "carbongcn": ModelSpec("carbongcn", "carbon_gcn"),
    "opencarbon_core": ModelSpec("opencarbon_core", "open_carbon", {"variant": "core"}),
    "opencarbon_monthly": ModelSpec("opencarbon_monthly", "open_carbon", {"variant": "monthly"}),
    "opencarbon_monthly_noviirs": ModelSpec(
        "opencarbon_monthly_noviirs",
        "open_carbon",
        {"variant": "monthly", "use_viirs": False},
    ),
    "opencarbon_monthly_vrex": ModelSpec(
        "opencarbon_monthly_vrex",
        "open_carbon",
        {"variant": "monthly", "poi_input_mode": "precomputed", "training_strategy": "vrex"},
    ),
    "opencarbon_monthly_precomputed": ModelSpec(
        "opencarbon_monthly_precomputed",
        "open_carbon",
        {"variant": "monthly", "poi_input_mode": "precomputed", "training_strategy": "erm"},
    ),
}


def list_model_ids() -> Iterable[str]:
    return tuple(MODEL_NAMES)


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return _MODEL_SPECS[model_id]
    except KeyError as error:
        available = ", ".join(list_model_ids())
        raise ValueError(f"Unknown model {model_id}; available models: {available}") from error


def validate_model_ids(model_ids: Iterable[str]) -> None:
    unknown = sorted(set(model_ids) - set(_MODEL_SPECS))
    if unknown:
        available = ", ".join(list_model_ids())
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Available models: {available}")
