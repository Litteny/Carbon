#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_transfer.config import load_config
from carbon_transfer.splits import build_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic transfer split manifests")
    parser.add_argument("--config", default="configs/experiments.yaml")
    args = parser.parse_args()
    audit = build_splits(load_config(args.config))
    print(f"Built {len(audit['folds'])} split manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
