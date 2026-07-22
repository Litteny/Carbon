#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_transfer.config import load_config
from carbon_transfer.training import train_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one carbon-transfer experiment task")
    parser.add_argument("--config", default="configs/stage1_gpu.yaml")
    parser.add_argument("--fold", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_dir = train_run(load_config(args.config), args.fold, args.model, args.seed, args.force)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
