from __future__ import annotations

import argparse
from typing import Optional, Sequence

from carbon_transfer.causal.analysis import describe_analysis, run_feature_causal_analysis
from carbon_transfer.config import apply_run_namespace, load_config, project_path


DEFAULT_CONFIG = "configs/experiments/feature_causal_analysis.yaml"


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run feature-level observational causal diagnostics",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--lags", nargs="+", type=int)
    parser.add_argument("--placebo-iterations", type=_nonnegative_int)
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _resolve_paths(config: dict) -> dict:
    for key in ("panel_file", "artifact_dir", "report_dir"):
        if key not in config:
            raise ValueError(f"Missing causal analysis config key: {key}")
        config[key] = str(project_path(config[key]))
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        apply_run_namespace(config, args.run_name)
        _resolve_paths(config)
    except ValueError as error:
        parser.error(str(error))
    if args.placebo_iterations is not None:
        config["placebo_iterations"] = args.placebo_iterations
    if args.dry_run:
        try:
            description = describe_analysis(config, args.features, args.lags)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        print(
            f"rows={description['rows']} cities={description['cities']} "
            f"periods={description['periods']}"
        )
        print(f"features={len(description['features'])} lags={description['lags']}")
        for feature in description["features"]:
            print(f"feature={feature}")
        print(f"tasks={description['tasks']}")
        return 0
    try:
        result = run_feature_causal_analysis(
            config, features=args.features, lags=args.lags, force=args.force,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(
        f"analyzed features={result['features_analyzed']} "
        f"estimates={result['estimates']} report_dir={result['report_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
