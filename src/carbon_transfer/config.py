from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    representation_dim: int = 128
    dropout: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 25
    contrastive_warmup_epochs: int = 5
    contrastive_weight: float = 0.01
    patience: int = 8
    batch_size: int = 16
    gradient_accumulation: int = 1
    amp: bool = True
    num_workers: int = 0
    temperature: float = 0.07
    persistent_workers: bool = True
    prefetch_factor: int = 2
    neighborhood_aggregation: str = "spatial_attention"
    invariance_weight: float = 1.0
    invariance_warmup_epochs: int = 2
    environment_key: str = "city_id"
    poi_input_mode: str = "dense"
    input_scope: str = "default"
    record_train_r2: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    panel_file: str
    neighbors_file: str
    poi_dir: str
    split_dir: str
    artifact_dir: str
    models: Sequence[str]
    seed: int = 42
    device: str = "auto"
    report_dir: Optional[str] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    config["_config_path"] = str(path)
    return config


def apply_overrides(config: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def apply_run_namespace(config: Dict[str, Any], run_name: Optional[str]) -> Dict[str, Any]:
    """Isolate one invocation's runs and reports below a stable run name."""
    if not run_name:
        return config
    value = str(run_name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(
            "run_name must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    config["artifact_dir"] = str(Path(config["artifact_dir"]) / value)
    if config.get("report_dir"):
        config["report_dir"] = str(Path(config["report_dir"]) / value)
    config["run_name"] = value
    return config


def validate_experiment_config(config: Mapping[str, Any]) -> ExperimentConfig:
    required = ["panel_file", "neighbors_file", "poi_dir", "split_dir", "artifact_dir", "models"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing experiment config keys: {', '.join(missing)}")
    training = TrainingConfig(**dict(config.get("training", {})))
    if training.poi_input_mode not in {"dense", "precomputed", "tabular"}:
        raise ValueError(
            "training.poi_input_mode must be 'dense', 'precomputed', or 'tabular'"
        )
    if training.poi_input_mode == "precomputed":
        required_embedding = [
            key for key in ("poi_embedding_dir", "poi_embedding_checkpoint_template")
            if key not in config
        ]
        if required_embedding:
            raise ValueError(
                "Precomputed POI input requires experiment config keys: "
                + ", ".join(required_embedding)
            )
    return ExperimentConfig(
        panel_file=str(config["panel_file"]),
        neighbors_file=str(config["neighbors_file"]),
        poi_dir=str(config["poi_dir"]),
        split_dir=str(config["split_dir"]),
        artifact_dir=str(config["artifact_dir"]),
        models=tuple(str(value) for value in config["models"]),
        seed=int(config.get("seed", 42)),
        device=str(config.get("device", "auto")),
        report_dir=str(config["report_dir"]) if config.get("report_dir") else None,
        training=training,
    )


def project_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path
