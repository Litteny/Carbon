from __future__ import annotations

import argparse
from typing import Optional, Sequence

from carbon_transfer.config import load_config
from carbon_transfer.models.registry import list_model_ids
from carbon_transfer.data import build_all
from carbon_transfer.splits import build_splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carbon", description="Carbon transfer data CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="Model registry commands")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="List available model IDs")

    data = sub.add_parser("data", help="Data commands")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_build = data_sub.add_parser("build", help="Build feature datasets")
    data_build.add_argument("--config", default="configs/data.yaml")
    data_build.add_argument("--skip-poi", action="store_true")

    splits = sub.add_parser("splits", help="Split manifest commands")
    splits_sub = splits.add_subparsers(dest="splits_command", required=True)
    splits_build = splits_sub.add_parser("build", help="Build split manifests")
    splits_build.add_argument("--config", default="configs/experiments.yaml")

    validate = sub.add_parser("validate", help="Validate data or experiment configuration")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    validate_data = validate_sub.add_parser("data")
    validate_data.add_argument("--config", required=True)
    validate_exp = validate_sub.add_parser("experiment")
    validate_exp.add_argument("--config", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "models":
        for model in list_model_ids():
            print(model)
        return 0
    if args.command == "data":
        audit = build_all(load_config(args.config), skip_poi=args.skip_poi)
        print(f"Built {audit['rows']} panel rows and {audit['neighbor_edges']} neighbor edges")
        return 0
    if args.command == "splits":
        audit = build_splits(load_config(args.config))
        print(f"Built {len(audit['folds'])} split manifests")
        return 0

    config = load_config(args.config)
    if args.validate_command == "experiment":
        from carbon_transfer.experiments import discover_tasks

        tasks = discover_tasks(config)
        print(f"valid experiment config; tasks={len(tasks)}")
    else:
        required = ["raw_root", "feature_dir"]
        missing = [key for key in required if key not in config]
        if missing:
            parser.error(f"Missing data config keys: {', '.join(missing)}")
        print("valid data config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
