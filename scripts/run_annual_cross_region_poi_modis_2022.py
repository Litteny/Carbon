from __future__ import annotations

import argparse
from typing import Optional, Sequence

from carbon_transfer.config import apply_run_namespace, load_config
from carbon_transfer.experiments import aggregate_for_config, discover_tasks, run_experiment

DEFAULT_CONFIG = "configs/experiments/annual_cross_region_poi_modis_2022.yaml"
EXPERIMENT = "annual_cross_region_2022"
CITY_ALIASES = {"nyc": "nyc", "new_york": "nyc", "tokyo": "tokyo"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run annual POI+MODIS OpenCarbon experiments")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cities", nargs="+", choices=tuple(CITY_ALIASES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int); parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int); parser.add_argument("--device"); parser.add_argument("--run-name")
    mode = parser.add_mutually_exclusive_group(); mode.add_argument("--dry-run", action="store_true"); mode.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config); apply_run_namespace(config, args.run_name)
    # Reuse the manifests produced by the prior annual cross-region ablation.
    config["split_experiment"] = EXPERIMENT
    config["fixed_split_seed"] = 42
    if args.seeds: config["training_seeds"] = list(args.seeds)
    if args.device: config["device"] = args.device
    training = dict(config.get("training", {}))
    for value, key in ((args.epochs, "max_epochs"), (args.patience, "patience"), (args.batch_size, "batch_size")):
        if value is not None: training[key] = value
    config["training"] = training
    tasks = discover_tasks(config, seeds=args.seeds, experiments=[EXPERIMENT])
    cities = {CITY_ALIASES[v] for v in args.cities} if args.cities else set(config["cities"])
    tasks = [task for task in tasks if task.fold.split("_split_seed_", 1)[0].split("_", 1)[1] in cities]
    if not tasks: raise SystemExit("No matching tasks found; build annual split manifests first")
    if args.dry_run:
        for task in tasks: print(f"city={task.fold.split('_split_seed_', 1)[0].split('_', 1)[1]} {task.identity}")
        print(f"tasks={len(tasks)}"); return 0
    if args.evaluate_only: aggregate_for_config(config, tasks); return 0
    return run_experiment(config, tasks, force=args.force, aggregate_tasks=tasks)


if __name__ == "__main__":
    raise SystemExit(main())
