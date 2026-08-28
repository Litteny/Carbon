from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    adapter: str
    parameters: Mapping[str, object] = field(default_factory=dict)
