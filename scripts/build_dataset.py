#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carbon_transfer.config import load_config
from carbon_transfer.data import build_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the unified grid-month dataset")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--skip-poi", action="store_true", help="Skip sparse POI generation")
    args = parser.parse_args()
    audit = build_all(load_config(args.config), skip_poi=args.skip_poi)
    print(f"Built {audit['rows']} panel rows and {audit['neighbor_edges']} neighbor edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
