#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 12 single-month cross-region multi-seed tasks")
    parser.add_argument("--config", default="configs/single_month_multiseed_gpu.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from carbon_transfer.config import load_config, project_path
    from carbon_transfer.evaluation import aggregate_single_month_multiseed_runs
    from carbon_transfer.progress import OpenCarbonTaskProgress, run_is_complete
    from carbon_transfer.training import train_run
    from carbon_transfer.utils import write_json

    config = load_config(args.config)
    experiment = config["split_experiment"]
    split_root = project_path(config["split_dir"]) / experiment
    seeds = [int(seed) for seed in config["seeds"]]
    models = list(config.get("models", ["opencarbon_monthly"]))
    if models != ["opencarbon_monthly"]:
        raise ValueError("The multi-seed experiment must run only opencarbon_monthly")
    tasks = []
    pattern = re.compile(r"_split_seed_(\d+)$")
    for path in sorted(split_root.glob(f"{config['period']}_*.parquet")):
        match = pattern.search(path.stem)
        if match and int(match.group(1)) in seeds:
            tasks.append((path.stem, "opencarbon_monthly", int(match.group(1))))
    if len(tasks) != 12:
        raise RuntimeError(f"Expected 12 multi-seed tasks, found {len(tasks)}")
    if args.dry_run:
        for fold, model, seed in tasks:
            print(f"fold={fold} model={model} seed={seed}")
        print(f"tasks={len(tasks)}")
        return 0

    artifact_dir = project_path(config["artifact_dir"])
    failures = []
    with OpenCarbonTaskProgress(tasks) as progress:
        for fold, model, seed in tasks:
            was_complete = run_is_complete(artifact_dir, fold, model, seed)
            try:
                print(f"[multiseed] fold={fold} model={model} seed={seed}", flush=True)
                train_run(config, fold, model, seed, args.force)
                progress.finish(fold, model, "skipped" if was_complete and not args.force else "completed")
            except Exception as error:
                failures.append({
                    "fold": fold, "model": model, "seed": seed,
                    "error": repr(error), "traceback": traceback.format_exc(),
                })
                print(f"FAILED fold={fold} model={model}: {error}", file=sys.stderr, flush=True)
                progress.finish(fold, model, "failed")
    aggregate_single_month_multiseed_runs(artifact_dir, project_path(config["report_dir"]), len(tasks))
    write_json(artifact_dir / "multiseed_failures.json", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
