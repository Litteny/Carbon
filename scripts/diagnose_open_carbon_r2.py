from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from carbon_transfer.r2_diagnostics import diagnose_open_carbon_run


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose where OpenCarbon R² first becomes negative",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--num-workers", type=_non_negative_int)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = diagnose_open_carbon_run(
        args.run_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_predictions=args.save_predictions,
        force=args.force,
    )
    print(f"diagnostics={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
