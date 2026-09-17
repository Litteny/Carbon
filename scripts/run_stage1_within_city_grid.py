from __future__ import annotations

import argparse
from typing import Optional, Sequence

from carbon_transfer.config import apply_run_namespace, load_config
from carbon_transfer.constants import STAGE1_MODEL_NAMES
from carbon_transfer.experiments import (
    ExperimentTask,
    aggregate_for_config,
    discover_tasks,
    run_experiment,
)
from carbon_transfer.models.registry import list_model_ids


DEFAULT_CONFIG = "configs/experiments/stage1_within_city_grid.yaml"
DEFAULT_EXPERIMENT = "stage1_within_city_grid_2021_2023"
EXPERIMENT_PREFIX = "stage1_within_city_grid"
CITY_ALIASES = {
    "chicago": "chicago",
    "nyc": "nyc",
    "new_york": "nyc",
    "singapore": "singapore",
    "tokyo": "tokyo",
}
AGGREGATIONS = ("mean_mlp_gate", "spatial_attention")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _task_city(task: ExperimentTask) -> str:
    prefix = "2021-2023_"
    suffix = "_grid_fixed"
    if not task.fold.startswith(prefix) or not task.fold.endswith(suffix):
        raise ValueError(f"Unexpected stage1 grid fold: {task.fold}")
    return task.fold[len(prefix):-len(suffix)]


def _selected_cities(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    return {CITY_ALIASES[value] for value in values}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run stage1 same-month within-city grid holdout experiments",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cities", nargs="+", choices=tuple(CITY_ALIASES))
    parser.add_argument("--models", nargs="+", choices=tuple(list_model_ids()))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--patience", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--device")
    parser.add_argument("--neighborhood-aggregation", choices=AGGREGATIONS)
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
    experiment = str(config.get("experiment", DEFAULT_EXPERIMENT))
    if not experiment.startswith(EXPERIMENT_PREFIX):
        parser.error(
            "Stage1 within-city grid experiment names must start with "
            f"{EXPERIMENT_PREFIX}"
        )
    config["split_experiment"] = experiment
    configured_models = set(str(value) for value in config.get("models", []))
    if not configured_models or not configured_models.issubset(set(STAGE1_MODEL_NAMES)):
        parser.error(
            "Stage1 within-city grid experiments only support: "
            + ", ".join(STAGE1_MODEL_NAMES)
        )
    if args.models:
        if not set(args.models).issubset(set(STAGE1_MODEL_NAMES)):
            parser.error(
                "Stage1 within-city grid experiments only support: "
                + ", ".join(STAGE1_MODEL_NAMES)
            )
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
        (args.neighborhood_aggregation, "neighborhood_aggregation"),
    ):
        if argument is not None:
            training[key] = argument
    config["training"] = training

    protocol_tasks = discover_tasks(
        config,
        models=args.models,
        seeds=args.seeds,
        experiments=[experiment],
    )
    if not protocol_tasks:
        parser.error(f"No tasks found for {experiment}; build split manifests first")

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
        print(f"aggregated experiment={experiment} tasks={len(protocol_tasks)}")
        return 0

    return run_experiment(
        config,
        selected_tasks,
        force=args.force,
        aggregate_tasks=protocol_tasks,
    )


if __name__ == "__main__":
    raise SystemExit(main())
