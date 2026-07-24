from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    return config


def project_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path
