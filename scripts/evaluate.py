#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carbon_transfer.evaluation import evaluate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute metrics for an experiment run")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    summary = evaluate_run(Path(args.run_dir))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
