from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentTask:
    experiment: str
    protocol: str
    fold: str
    model_id: str
    seed: int
    manifest_path: Path

    @property
    def identity(self) -> str:
        return (
            f"experiment={self.experiment} protocol={self.protocol} "
            f"fold={self.fold} model={self.model_id} seed={self.seed}"
        )
