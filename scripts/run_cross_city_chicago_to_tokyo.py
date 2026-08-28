from __future__ import annotations

import argparse
from typing import Optional, Sequence

from carbon_transfer.config import apply_run_namespace, load_config
from carbon_transfer.experiments import aggregate_for_config, discover_tasks, run_experiment
from carbon_transfer.models.registry import list_model_ids


DEFAULT_CONFIG = "configs/experiments/cross_city_chicago_to_tokyo_opencarbon.yaml"
EXPERIMENT = "cross_city_chicago_to_tokyo"
TEST_CITY = "tokyo"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenCarbon with Chicago training, NYC/Singapore validation, "
            "and Tokyo testing"
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cities", nargs="+", choices=(TEST_CITY,))
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

    selected_models = list(config.get("models", ()))
    invalid = [model for model in selected_models if not model.startswith("opencarbon_")]
    if invalid:
        parser.error("This protocol only supports OpenCarbon models: " + ", ".join(invalid))

    tasks = discover_tasks(
        config,
        models=args.models,
        seeds=args.seeds,
        experiments=[EXPERIMENT],
    )
    if not tasks:
        parser.error(f"No tasks found for {EXPERIMENT}; build split manifests first")

    if args.dry_run:
        for task in tasks:
            print(
                "train=chicago validation=nyc,singapore test=tokyo " + task.identity
            )
        print(f"tasks={len(tasks)}")
        return 0
    if args.evaluate_only:
        aggregate_for_config(config, tasks)
        print(f"aggregated experiment={EXPERIMENT} tasks={len(tasks)}")
        return 0
    return run_experiment(config, tasks, force=args.force, aggregate_tasks=tasks)


if __name__ == "__main__":
    raise SystemExit(main())
