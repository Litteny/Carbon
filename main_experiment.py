#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from carbon_transfer.config import load_config, project_path
from carbon_transfer.constants import MODEL_NAMES

DEFAULT_MODEL = "opencarbon_monthly"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one or more carbon-transfer experiments")
    parser.add_argument("--config", default="configs/stage1_gpu.yaml", help="Experiment YAML file")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--city", help="Cross-city target, for example chicago")
    target.add_argument("--fold", help="Exact cross-city or cross-region fold name")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_NAMES)
    parser.add_argument(
        "--models", action="append", metavar="MODEL[,MODEL...]",
        help="Run multiple models; may be repeated and takes precedence over --model",
    )
    parser.add_argument("--seed", type=int, help="Random seed; defaults to the config value")
    parser.add_argument("--device", help="Device override, for example auto, cpu, cuda, or cuda:0")
    parser.add_argument("--force", action="store_true", help="Replace an existing run")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print tasks without training")
    return parser


def discover_folds(split_dir: Path) -> Mapping[str, Path]:
    paths = sorted((split_dir / "cross_city").glob("*.parquet"))
    paths += sorted((split_dir / "cross_region").glob("*.parquet"))
    return {path.stem: path for path in paths}


def resolve_fold(city: Optional[str], fold: Optional[str], folds: Iterable[str]) -> str:
    available = sorted(set(folds))
    if fold:
        if fold not in available:
            raise ValueError(f"Unknown fold '{fold}'. Available folds: {', '.join(available)}")
        return fold

    normalized_city = (city or "").strip().lower()
    city_folds = {name.removeprefix("target_"): name for name in available if name.startswith("target_")}
    if normalized_city not in city_folds:
        choices = ", ".join(sorted(city_folds))
        raise ValueError(f"Unknown city '{city}'. Available cities: {choices}")
    return city_folds[normalized_city]


def resolve_models(model: str, model_groups: Optional[Sequence[str]]) -> List[str]:
    if not model_groups:
        return [model]
    models = [item.strip() for group in model_groups for item in group.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name")
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Available models: {', '.join(MODEL_NAMES)}"
        )
    return list(dict.fromkeys(models))


def prepare_tasks(args: argparse.Namespace) -> Tuple[dict, str, List[str], int]:
    config = load_config(args.config)
    folds = discover_folds(project_path(config["split_dir"]))
    if not folds:
        raise ValueError(f"No split manifests found under {project_path(config['split_dir'])}")
    fold = resolve_fold(args.city, args.fold, folds)
    models = resolve_models(args.model, args.models)
    seed = int(config["seed"] if args.seed is None else args.seed)
    if args.device:
        config["device"] = args.device
    return config, fold, models, seed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config, fold, models, seed = prepare_tasks(args)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    for model in models:
        print(f"fold={fold} model={model} seed={seed} device={config.get('device', 'auto')}")
    if args.dry_run:
        print(f"tasks={len(models)}")
        return 0

    from carbon_transfer.training import train_run
    from carbon_transfer.progress import OpenCarbonTaskProgress, run_is_complete
    from carbon_transfer.utils import write_json

    failures = []
    tasks = [(fold, model, seed) for model in models]
    with OpenCarbonTaskProgress(tasks) as progress:
        for _, model, _ in tasks:
            was_complete = run_is_complete(project_path(config["artifact_dir"]), fold, model, seed)
            try:
                run_dir = train_run(config, fold, model, seed, args.force)
                print(run_dir)
                status = "skipped" if was_complete and not args.force else "completed"
                progress.finish(fold, model, status)
            except Exception as error:
                failures.append({
                    "fold": fold,
                    "model": model,
                    "seed": seed,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                })
                print(f"FAILED fold={fold} model={model}: {error}", file=sys.stderr, flush=True)
                progress.finish(fold, model, "failed")

    if failures:
        failure_path = project_path(config["artifact_dir"]) / "experiment_failures.json"
        write_json(failure_path, failures)
        print(f"Failure details: {failure_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
