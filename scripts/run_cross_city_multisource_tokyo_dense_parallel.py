"""Run the three dense Tokyo-held-out folds concurrently."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


FOLDS = (
    "chicago_nyc_train_singapore_validation_tokyo_test",
    "chicago_singapore_train_nyc_validation_tokyo_test",
    "nyc_singapore_train_chicago_validation_tokyo_test",
)
DEFAULT_CONFIG = "configs/experiments/cross_city_multisource_tokyo_opencarbon.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the three dense multisource Tokyo folds in parallel",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="Shared device, e.g. cuda or cuda:0")
    parser.add_argument(
        "--devices",
        nargs=3,
        metavar=("DEVICE1", "DEVICE2", "DEVICE3"),
        help="One device per fold; overrides --device",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.devices and args.device:
        raise SystemExit("use either --device or --devices, not both")
    devices = list(args.devices or [args.device or "cuda"] * len(FOLDS))
    root = Path(__file__).resolve().parents[1]
    commands = []
    for fold, device in zip(FOLDS, devices):
        command = [
            sys.executable,
            str(root / "scripts" / "run_cross_city_multisource_tokyo.py"),
            "--config", args.config,
            "--folds", fold,
            "--models", "opencarbon_monthly",
            "--device", device,
        ]
        if args.force:
            command.append("--force")
        commands.append(command)
        print(" ".join(command), flush=True)
    if args.dry_run:
        return 0

    processes = []
    for index, command in enumerate(commands):
        environment = os.environ.copy()
        environment.setdefault("PYTHONPATH", str(root / "src"))
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=None,
            stderr=None,
        )
        processes.append((FOLDS[index], process))
    failures = []
    for fold, process in processes:
        return_code = process.wait()
        if return_code:
            failures.append(f"{fold} (exit {return_code})")
    if failures:
        print("Dense fold failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("All dense folds completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
