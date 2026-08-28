from __future__ import annotations

import argparse
from typing import Optional, Sequence

from carbon_transfer.config import apply_run_namespace, load_config
from carbon_transfer.experiments import (
    ExperimentTask,
    aggregate_for_config,
    discover_tasks,
    run_experiment,
)
from carbon_transfer.models.registry import list_model_ids


DEFAULT_CONFIG = "configs/experiments/annual_cross_region_2022.yaml"
EXPERIMENT = "annual_cross_region_2022"
CITY_ALIASES = {
    "chicago": "chicago",
    "nyc": "nyc",
    "new_york": "nyc",
    "singapore": "singapore",
    "tokyo": "tokyo",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _task_city(task: ExperimentTask) -> str:
    marker = "_split_seed_"
    prefix, separator, _ = task.fold.rpartition(marker)
    if not separator or "_" not in prefix:
        raise ValueError(f"Unexpected annual fold: {task.fold}")
    return prefix.split("_", 1)[1]


def _selected_cities(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    return {CITY_ALIASES[value] for value in values}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run annual within-city cross-region experiments")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cities", nargs="+", choices=tuple(CITY_ALIASES))
    parser.add_argument("--models", nargs="+", choices=tuple(list_model_ids()))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--patience", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.evaluate_only and args.force:
        parser.error("--force cannot be used with --evaluate-only")

    config = load_config(args.config)
    try:
        apply_run_namespace(config, args.run_name)
    except ValueError as error:
        parser.error(str(error))
    config["split_experiment"] = EXPERIMENT
    if args.models:
        config["models"] = list(args.models)
    if args.seeds:
        config["seeds"] = list(args.seeds)
    if args.device:
        config["device"] = args.device

    training = dict(config.get("training", {}))
    for argument, key in (
        (args.epochs, "max_epochs"),
        (args.patience, "patience"),
        (args.batch_size, "batch_size"),
    ):
        if argument is not None:
            training[key] = argument
    config["training"] = training

    protocol_tasks = discover_tasks(
        config,
        models=args.models,
        seeds=args.seeds,
        experiments=[EXPERIMENT],
    )
    if not protocol_tasks:
        parser.error(
            f"No tasks found for {EXPERIMENT}; build the required split manifests first"
        )

    cities = _selected_cities(args.cities)
    selected_tasks = [
        task for task in protocol_tasks
        if cities is None or _task_city(task) in cities
    ]
    if not selected_tasks:
        parser.error("No tasks match the requested cities, models, and seeds")

    if args.dry_run:
        for task in selected_tasks:
            print(f"city={_task_city(task)} {task.identity}")
        print(f"tasks={len(selected_tasks)}")
        return 0

    if args.evaluate_only:
        aggregate_for_config(config, protocol_tasks)
        print(f"aggregated experiment={EXPERIMENT} tasks={len(protocol_tasks)}")
        return 0

    return run_experiment(
        config,
        selected_tasks,
        force=args.force,
        aggregate_tasks=protocol_tasks,
    )


if __name__ == "__main__":
    raise SystemExit(main())
