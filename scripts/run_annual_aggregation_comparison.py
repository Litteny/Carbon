from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from carbon_transfer.config import load_config, project_path
from carbon_transfer.experiments import discover_tasks
from carbon_transfer.experiments.aggregation_comparison import compare_annual_aggregations
from carbon_transfer.progress import RunProgress, run_is_complete
from carbon_transfer.training import train_run


DEFAULT_CONFIG = "configs/experiments/annual_cross_region_2022_mean_mlp_gate.yaml"
EXPERIMENT = "annual_cross_region_2022"
FOLD = "2022_chicago_split_seed_42"
MODEL = "opencarbon_monthly"
SEED = 42


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Chicago annual mean-MLP-gate aggregation comparison",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--patience", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--device")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _comparison(config: dict, candidate_run: Path) -> dict:
    return compare_annual_aggregations(
        project_path(config["baseline_run_dir"]),
        candidate_run,
        project_path(config["report_dir"]),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.evaluate_only and args.force:
        parser.error("--force cannot be used with --evaluate-only")

    config = load_config(args.config)
    config["split_experiment"] = EXPERIMENT
    config["models"] = [MODEL]
    config["seeds"] = [SEED]
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
    if training.get("neighborhood_aggregation") != "mean_mlp_gate":
        parser.error("comparison config must use training.neighborhood_aggregation=mean_mlp_gate")
    config["training"] = training

    tasks = discover_tasks(
        config, folds=[FOLD], models=[MODEL], seeds=[SEED], experiments=[EXPERIMENT],
    )
    if len(tasks) != 1:
        parser.error(f"Expected exactly one comparison task, found {len(tasks)}")
    task = tasks[0]
    candidate_run = (
        project_path(config["artifact_dir"]) / EXPERIMENT / FOLD / MODEL / f"seed_{SEED}"
    )
    if args.dry_run:
        print(f"aggregation=mean_mlp_gate {task.identity}")
        print(f"baseline={project_path(config['baseline_run_dir'])}")
        print(f"candidate={candidate_run}")
        print("tasks=1")
        return 0
    if args.evaluate_only:
        _comparison(config, candidate_run)
        print(f"compared fold={FOLD} model={MODEL} seed={SEED}")
        return 0

    progress = RunProgress()
    was_complete = run_is_complete(project_path(config["artifact_dir"]), FOLD, MODEL, SEED)
    progress.task_start(FOLD, MODEL, SEED, EXPERIMENT)
    try:
        run_dir = train_run(config, FOLD, MODEL, SEED, args.force)
        progress.task_end(FOLD, MODEL, SEED, "skipped" if was_complete and not args.force else "completed")
        _comparison(config, run_dir)
    except Exception as error:
        progress.task_error(FOLD, MODEL, SEED, error)
        raise
    print(f"compared fold={FOLD} model={MODEL} seed={SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
