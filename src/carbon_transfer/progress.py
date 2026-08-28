from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO


OPEN_CARBON_MODELS = {
    "opencarbon_core",
    "opencarbon_monthly",
    "opencarbon_monthly_noviirs",
    "opencarbon_monthly_precomputed",
    "opencarbon_monthly_vrex",
}


def is_open_carbon(model_name: str) -> bool:
    return model_name in OPEN_CARBON_MODELS


def run_is_complete(artifact_dir: Path, fold: str, model: str, seed: int) -> bool:
    matches = artifact_dir.glob(f"*/{fold}/{model}/seed_{seed}/COMPLETE")
    return next(matches, None) is not None


class RunProgress:
    def __init__(self, stream: TextIO = sys.stderr) -> None:
        self.stream = stream
        self.started_at = time.monotonic()

    def emit(self, event: str, **fields: Any) -> Dict[str, Any]:
        payload = {"event": event, **fields}
        parts = [f"[{event}]"]
        for key, value in payload.items():
            if key == "event" or value is None:
                continue
            parts.append(f"{key}={value}")
        print(" ".join(parts), file=self.stream, flush=True)
        return payload

    def task_start(self, fold: str, model: str, seed: int, experiment: Optional[str] = None) -> Dict[str, Any]:
        return self.emit("task_start", experiment=experiment, fold=fold, model=model, seed=seed)

    def task_resume(self, fold: str, model: str, seed: int, epoch: int) -> Dict[str, Any]:
        return self.emit("task_resume", fold=fold, model=model, seed=seed, epoch=epoch)

    def task_skip(self, fold: str, model: str, seed: int) -> Dict[str, Any]:
        return self.emit("task_skip", fold=fold, model=model, seed=seed)

    def task_end(self, fold: str, model: str, seed: int, status: str) -> Dict[str, Any]:
        elapsed = f"{time.monotonic() - self.started_at:.1f}"
        return self.emit("task_end", fold=fold, model=model, seed=seed, status=status, elapsed_s=elapsed)

    def task_error(self, fold: str, model: str, seed: int, error: BaseException) -> Dict[str, Any]:
        return self.emit("task_error", fold=fold, model=model, seed=seed, error=repr(error))

    def early_stop(self, fold: str, model: str, seed: int, epoch: int, stale: int) -> Dict[str, Any]:
        return self.emit("early_stop", fold=fold, model=model, seed=seed, epoch=epoch, stale=stale)

    def lightgbm_start(self, fold: str, model: str, seed: int) -> Dict[str, Any]:
        return self.emit("fit_start", fold=fold, model=model, seed=seed)

    def lightgbm_end(self, fold: str, model: str, seed: int, best_iteration: int) -> Dict[str, Any]:
        return self.emit("fit_end", fold=fold, model=model, seed=seed, best_iteration=best_iteration)


class EpochProgress:
    def __init__(
        self,
        fold: str,
        model: str,
        seed: int,
        start_epoch: int,
        max_epochs: int,
        stream: TextIO = sys.stderr,
    ) -> None:
        self.fold = fold
        self.model = model
        self.seed = int(seed)
        self.max_epochs = int(max_epochs)
        self.stream = stream
        self.started_at = time.monotonic()
        self.events: List[Dict[str, Any]] = []
        self.start_epoch = int(start_epoch)

    def update(
        self,
        entry: Dict[str, Any],
        best_loss: float,
        stale: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        elapsed = time.monotonic() - self.started_at
        event = {
            "event": "epoch",
            "fold": self.fold,
            "model": self.model,
            "seed": self.seed,
            "epoch": int(entry["epoch"]),
            "max_epoch": self.max_epochs,
            "train_loss": float(entry.get("train_loss", entry.get("train_mae", 0.0))),
            "validation_mae": float(entry["validation_mae"]),
            "validation_r2": float(entry["validation_r2"]),
            "best_validation_mae": float(best_loss),
            "stale": int(stale),
            "elapsed_s": round(elapsed, 1),
        }
        if extra:
            event.update(extra)
        self.events.append(event)
        tokens = [
            f"[epoch {event['epoch']}/{self.max_epochs}]",
            f"fold={self.fold}",
            f"model={self.model}",
            f"seed={self.seed}",
            f"train_loss={event['train_loss']:.4f}",
            f"validation_mae={event['validation_mae']:.4f}",
            f"validation_r2={event['validation_r2']:.4f}",
            f"best_validation_mae={event['best_validation_mae']:.4f}",
            f"stale={event['stale']}",
            f"elapsed_s={event['elapsed_s']:.1f}",
        ]
        if "train_regression_mae" in event:
            tokens.append(f"train_regression_mae={float(event['train_regression_mae']):.4f}")
        if "train_contrastive" in event:
            tokens.append(f"train_contrastive={float(event['train_contrastive']):.4f}")
        if "train_r2" in event:
            tokens.append(f"train_r2={float(event['train_r2']):.4f}")
        for key in (
            "train_environment_mae",
            "train_environment_risk_variance",
            "train_invariance_penalty",
        ):
            if key in event:
                tokens.append(f"{key}={float(event[key]):.4f}")
        print(" ".join(tokens), file=self.stream, flush=True)
        return event

    def close(self) -> None:
        return None

    def __enter__(self) -> "EpochProgress":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def write_training_log(path: Path, events: Iterable[Dict[str, Any]], epochs: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"events": list(events), "epochs": list(epochs)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Backward-compatible names for older tests/imports. They now emit stable lines for TTY and non-TTY.
OpenCarbonEpochProgress = EpochProgress


class OpenCarbonTaskProgress:
    def __init__(self, tasks: Iterable[tuple], stream: TextIO = sys.stderr) -> None:
        self.stream = stream
        self.total = sum(1 for task in tasks if is_open_carbon(str(task[1])))
        self.completed = 0

    def finish(self, fold: str, model: str, status: str) -> None:
        if not is_open_carbon(model):
            return
        self.completed += 1
        print(
            f"[task_progress] completed={self.completed}/{self.total} fold={fold} model={model} status={status}",
            file=self.stream,
            flush=True,
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> "OpenCarbonTaskProgress":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
