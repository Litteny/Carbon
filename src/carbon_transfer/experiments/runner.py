from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from carbon_transfer.config import project_path
from carbon_transfer.models.registry import list_model_ids, validate_model_ids
from carbon_transfer.evaluation import (
    aggregate_runs,
    aggregate_annual_cross_region_runs,
    aggregate_single_month_multiseed_runs,
    aggregate_single_month_runs,
    aggregate_three_year_cross_region_runs,
)
from carbon_transfer.progress import RunProgress, run_is_complete
from carbon_transfer.training import train_run
from carbon_transfer.utils import write_json

from .task import ExperimentTask


def _split_roots(config: Dict) -> List[Path]:
    split_dir = project_path(config["split_dir"])
    if "split_experiment" in config:
        return [split_dir / str(config["split_experiment"])]
    experiment_config = str(config.get("experiment_config", ""))
    if "single_month" in experiment_config:
        return [split_dir / "single_month_cross_region"]
    roots = [split_dir / "cross_city", split_dir / "cross_region"]
    return [path for path in roots if path.exists()]


def _protocol_for_experiment(experiment: str) -> str:
    if experiment.startswith("three_year_cross_region"):
        return "three_year_cross_region"
    if experiment.startswith("annual_cross_region"):
        return "annual_cross_region"
    if experiment == "single_month_cross_region_multiseed":
        return "single_month_multiseed"
    if experiment == "single_month_cross_region":
        return "single_month"
    return "stage1"


def _seeds_for_fold(config: Dict, fold: str) -> List[int]:
    if config.get("independent_training_seeds"):
        values = config.get("training_seeds", config.get("seeds", [42]))
        return [int(value) for value in values]
    if "seeds" not in config:
        return [int(config.get("seed", 42))]
    match = re.search(r"_split_seed_(\d+)$", fold)
    if match:
        seed = int(match.group(1))
        configured = {int(value) for value in config["seeds"]}
        return [seed] if seed in configured else []
    return [int(value) for value in config["seeds"]]


def discover_tasks(
    config: Dict,
    folds: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    experiments: Optional[Sequence[str]] = None,
) -> List[ExperimentTask]:
    selected_folds = {str(value) for value in folds} if folds else None
    selected_experiments = {str(value) for value in experiments} if experiments else None
    selected_models = list(models or config.get("models", list_model_ids()))
    validate_model_ids(selected_models)

    tasks: List[ExperimentTask] = []
    for root in _split_roots(config):
        if not root.exists() or (selected_experiments and root.name not in selected_experiments):
            continue
        experiment = root.name
        protocol = _protocol_for_experiment(experiment)
        for manifest_path in sorted(root.glob("*.parquet")):
            fold = manifest_path.stem
            fixed_split_seed = config.get("fixed_split_seed")
            if fixed_split_seed is not None:
                match = re.search(r"_split_seed_(\d+)$", fold)
                if match and int(match.group(1)) != int(fixed_split_seed):
                    continue
            if selected_folds and fold not in selected_folds:
                continue
            configured_fold_seeds = _seeds_for_fold(config, fold)
            if re.search(r"_split_seed_(\d+)$", fold):
                selected_seed_values = ({int(value) for value in seeds} if seeds else None)
                fold_seeds = [
                    value for value in configured_fold_seeds
                    if selected_seed_values is None or value in selected_seed_values
                ]
            else:
                fold_seeds = [int(value) for value in (seeds or configured_fold_seeds)]
            for seed in fold_seeds:
                for model_id in selected_models:
                    tasks.append(ExperimentTask(experiment, protocol, fold, model_id, seed, manifest_path))
    if selected_folds:
        found = {task.fold for task in tasks}
        missing = sorted(selected_folds - found)
        if missing:
            raise ValueError(f"Unknown fold(s): {', '.join(missing)}")
    return tasks


def _report_dir(config: Dict, protocol: str) -> Path:
    if "report_dir" in config:
        return project_path(config["report_dir"])
    if protocol == "stage1":
        return project_path("outputs/stage1/reports")
    return project_path(f"outputs/{protocol}/reports")


def aggregate_for_config(config: Dict, tasks: Iterable[ExperimentTask]) -> None:
    task_list = list(tasks)
    if not task_list:
        return
    protocol = task_list[0].protocol
    protocols = {task.protocol for task in task_list}
    if protocols != {protocol}:
        raise ValueError(f"Cannot aggregate mixed protocols: {sorted(protocols)}")
    artifact_dir = project_path(config["artifact_dir"])
    report_dir = _report_dir(config, protocol)
    if protocol == "three_year_cross_region":
        aggregate_three_year_cross_region_runs(artifact_dir, report_dir, len(task_list))
    elif protocol == "annual_cross_region":
        aggregate_annual_cross_region_runs(artifact_dir, report_dir, len(task_list))
    elif protocol == "single_month":
        aggregate_single_month_runs(artifact_dir, report_dir, len(task_list))
    elif protocol == "single_month_multiseed":
        aggregate_single_month_multiseed_runs(artifact_dir, report_dir, len(task_list))
    else:
        experiments = sorted({task.experiment for task in task_list})
        aggregate_runs(artifact_dir, report_dir, experiments=experiments)


def run_experiment(
    config: Dict,
    tasks: Sequence[ExperimentTask],
    force: bool = False,
    aggregate_tasks: Optional[Sequence[ExperimentTask]] = None,
) -> int:
    if not tasks:
        raise ValueError("No experiment tasks discovered")
    failures = []
    progress = RunProgress()
    artifact_dir = project_path(config["artifact_dir"])
    for task in tasks:
        was_complete = run_is_complete(artifact_dir, task.fold, task.model_id, task.seed)
        try:
            progress.task_start(task.fold, task.model_id, task.seed, task.experiment)
            train_run(config, task.fold, task.model_id, task.seed, force)
            status = "skipped" if was_complete and not force else "completed"
            progress.task_end(task.fold, task.model_id, task.seed, status)
        except Exception as error:
            failures.append({
                "experiment": task.experiment,
                "fold": task.fold,
                "model": task.model_id,
                "seed": task.seed,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            })
            progress.task_error(task.fold, task.model_id, task.seed, error)
            print(f"FAILED {task.identity}: {error}", file=sys.stderr, flush=True)
    aggregate_for_config(config, aggregate_tasks or tasks)
    write_json(artifact_dir / f"{tasks[0].experiment}_failures.json", failures)
    return 1 if failures else 0
