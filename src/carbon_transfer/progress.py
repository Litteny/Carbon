from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, TextIO

from tqdm import tqdm


OPEN_CARBON_MODELS = {
    "opencarbon_core",
    "opencarbon_monthly",
    "opencarbon_monthly_noviirs",
}
_TASK_PROGRESS_DEPTH = 0


def is_open_carbon(model_name: str) -> bool:
    return model_name in OPEN_CARBON_MODELS


def run_is_complete(artifact_dir: Path, fold: str, model: str, seed: int) -> bool:
    matches = artifact_dir.glob(f"*/{fold}/{model}/seed_{seed}/COMPLETE")
    return next(matches, None) is not None


def _is_interactive(stream: TextIO) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


class OpenCarbonTaskProgress:
    def __init__(self, tasks: Iterable[tuple], stream: TextIO = sys.stderr) -> None:
        self.stream = stream
        self.total = sum(1 for task in tasks if is_open_carbon(str(task[1])))
        self.completed = 0
        self.bar = tqdm(
            total=self.total,
            desc="OpenCarbon tasks",
            unit="task",
            dynamic_ncols=True,
            disable=not _is_interactive(stream) or self.total == 0,
            file=stream,
            position=0,
        )

    def finish(self, fold: str, model: str, status: str) -> None:
        if not is_open_carbon(model):
            return
        self.completed += 1
        if self.bar.disable:
            print(
                f"[OpenCarbon tasks] {self.completed}/{self.total} "
                f"fold={fold} model={model} status={status}",
                file=self.stream,
                flush=True,
            )
        else:
            self.bar.set_postfix_str(f"{fold} {status}", refresh=False)
            self.bar.update(1)

    def close(self) -> None:
        self.bar.close()

    def __enter__(self) -> "OpenCarbonTaskProgress":
        global _TASK_PROGRESS_DEPTH
        _TASK_PROGRESS_DEPTH += 1
        return self

    def __exit__(self, *_: object) -> None:
        global _TASK_PROGRESS_DEPTH
        self.close()
        _TASK_PROGRESS_DEPTH -= 1


class OpenCarbonEpochProgress:
    def __init__(
        self,
        fold: str,
        model: str,
        start_epoch: int,
        max_epochs: int,
        stream: TextIO = sys.stderr,
    ) -> None:
        self.fold = fold
        self.model = model
        self.stream = stream
        self.max_epochs = max_epochs
        self.bar = tqdm(
            total=max_epochs,
            initial=start_epoch,
            desc=f"{model} {fold}",
            unit="epoch",
            dynamic_ncols=True,
            disable=not _is_interactive(stream),
            file=stream,
            position=1 if _TASK_PROGRESS_DEPTH else 0,
            leave=False,
        )

    def update(self, entry: Dict, best_loss: float, stale: int) -> None:
        values = {
            "train": f"{float(entry['train_loss']):.4f}",
            "val": f"{float(entry['validation_mae']):.4f}",
            "best": f"{best_loss:.4f}",
            "stale": stale,
        }
        if self.bar.disable:
            print(
                f"[OpenCarbon epoch] fold={self.fold} model={self.model} "
                f"epoch={entry['epoch']}/{self.max_epochs} "
                f"train_loss={values['train']} validation_mae={values['val']} "
                f"best={values['best']} stale={stale}",
                file=self.stream,
                flush=True,
            )
        else:
            self.bar.set_postfix(values, refresh=False)
            self.bar.update(1)

    def close(self) -> None:
        self.bar.close()

    def __enter__(self) -> "OpenCarbonEpochProgress":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
