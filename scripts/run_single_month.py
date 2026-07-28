#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the four 2022-08 regional-transfer tasks")
    parser.add_argument("--config", default="configs/single_month_gpu.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from carbon_transfer.config import load_config, project_path
    from carbon_transfer.evaluation import aggregate_single_month_runs
    from carbon_transfer.progress import OpenCarbonTaskProgress, run_is_complete
    from carbon_transfer.training import train_run
    from carbon_transfer.utils import write_json

    config = load_config(args.config)
    split_root = project_path(config["split_dir"]) / "single_month_cross_region"
    folds = [path.stem for path in sorted(split_root.glob(f"{config['period']}_*.parquet"))]
    models = list(config.get("models", ["opencarbon_monthly"]))
    if len(folds) != 4:
        raise RuntimeError(f"Expected 4 single-month folds, found {len(folds)}")
    if models != ["opencarbon_monthly"]:
        raise ValueError("The pilot configuration must run only opencarbon_monthly")
    tasks = [(fold, model, int(config["seed"])) for fold in folds for model in models]
    if args.dry_run:
        for fold, model, seed in tasks:
            print(f"fold={fold} model={model} seed={seed}")
        print(f"tasks={len(tasks)}")
        return 0

    failures = []
    with OpenCarbonTaskProgress(tasks) as progress:
        for fold, model, seed in tasks:
            was_complete = run_is_complete(project_path(config["artifact_dir"]), fold, model, seed)
            try:
                print(f"[single-month] fold={fold} model={model} seed={seed}", flush=True)
                train_run(config, fold, model, seed, args.force)
                status = "skipped" if was_complete and not args.force else "completed"
                progress.finish(fold, model, status)
            except Exception as error:
                failures.append({
                    "fold": fold, "model": model, "seed": seed,
                    "error": repr(error), "traceback": traceback.format_exc(),
                })
                print(f"FAILED fold={fold} model={model}: {error}", file=sys.stderr, flush=True)
                progress.finish(fold, model, "failed")
    artifact_dir = project_path(config["artifact_dir"])
    aggregate_single_month_runs(artifact_dir, project_path(config["report_dir"]), len(tasks))
    write_json(artifact_dir / "single_month_failures.json", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
