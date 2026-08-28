from __future__ import annotations

import argparse
from typing import Optional, Sequence

from carbon_transfer.config import load_config, project_path
from carbon_transfer.transfer_features.screening import describe_screening, run_feature_transfer_screening


DEFAULT_CONFIG = "configs/experiments/feature_transfer_screening.yaml"


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model-independent cross-city feature screening")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--bootstrap-iterations", type=_nonnegative_int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for key in ("panel_file", "causal_run_dir", "report_dir"):
        if key not in config:
            parser.error(f"Missing transfer-screening config key: {key}")
        config[key] = str(project_path(config[key]))
    if args.bootstrap_iterations is not None:
        config["bootstrap_iterations"] = args.bootstrap_iterations
    try:
        if args.dry_run:
            result = describe_screening(config, args.features)
            print(f"rows={result['rows']} cities={result['cities']} periods={result['periods']}")
            print(f"features={len(result['features'])} bootstrap_iterations={result['bootstrap_iterations']}")
            for feature in result["features"]:
                print(f"feature={feature}")
            return 0
        result = run_feature_transfer_screening(config, features=args.features)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"screened features={result['features_screened']} report_dir={result['report_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
