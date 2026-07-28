#!/usr/bin/env python
from __future__ import annotations

import argparse
import traceback
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main() -> int:
    parser = argparse.ArgumentParser(description="Run all 48 single-seed stage-1 tasks")
    parser.add_argument("--config", default="configs/stage1_gpu.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the 48-task matrix without training")
    args = parser.parse_args()
    from carbon_transfer.config import load_config, project_path
    from carbon_transfer.evaluation import aggregate_runs
    from carbon_transfer.progress import OpenCarbonTaskProgress, run_is_complete
    from carbon_transfer.training import train_run
    from carbon_transfer.utils import write_json

    config = load_config(args.config)
    split_dir = project_path(config["split_dir"])
    folds = [path.stem for path in sorted((split_dir / "cross_city").glob("*.parquet"))]
    folds += [path.stem for path in sorted((split_dir / "cross_region").glob("*.parquet"))]
    if len(folds) != 8:
        raise RuntimeError(f"Expected 8 stage-1 folds, found {len(folds)}")
    if args.dry_run:
        for fold in folds:
            for model in config["models"]:
                print(f"fold={fold} model={model} seed={config['seed']}")
        print(f"tasks={len(folds) * len(config['models'])}")
        return 0
    failures = []
    tasks = [(fold, model, int(config["seed"])) for fold in folds for model in config["models"]]
    with OpenCarbonTaskProgress(tasks) as progress:
        for fold, model, seed in tasks:
            was_complete = run_is_complete(project_path(config["artifact_dir"]), fold, model, seed)
            try:
                print(f"[stage1] fold={fold} model={model} seed={seed}", flush=True)
                train_run(config, fold, model, seed, args.force)
                status = "skipped" if was_complete and not args.force else "completed"
                progress.finish(fold, model, status)
            except Exception as error:
                failures.append({
                    "fold": fold, "model": model, "seed": int(config["seed"]),
                    "error": repr(error), "traceback": traceback.format_exc(),
                })
                print(f"[stage1] FAILED fold={fold} model={model}: {error}", file=sys.stderr, flush=True)
                progress.finish(fold, model, "failed")
    artifact_dir = project_path(config["artifact_dir"])
    aggregate_runs(artifact_dir, project_path("reports/stage1"))
    write_json(artifact_dir / "stage1_failures.json", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
