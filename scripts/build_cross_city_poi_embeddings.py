from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from carbon_transfer.config import load_config, project_path
from carbon_transfer.poi_embeddings import build_fold_embeddings
from carbon_transfer.utils import resolve_device


DEFAULT_CONFIG = "configs/experiments/stage1_cross_city_vrex.yaml"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fold-specific frozen POI embeddings for cross-city experiments",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--work-dir", default="/tmp/carbon_poi_embeddings")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if str(config.get("split_experiment")) != "cross_city":
        raise ValueError("POI embedding builder only supports split_experiment=cross_city")
    split_root = project_path(config["split_dir"]) / "cross_city"
    available = sorted(path.stem for path in split_root.glob("*.parquet"))
    selected = list(args.folds or available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown fold(s): {', '.join(unknown)}")
    device = resolve_device(args.device or config.get("device", "auto"))
    panel_path = project_path(config["panel_file"])
    poi_root = project_path(config["poi_dir"])
    output_root = project_path(config["poi_embedding_dir"])
    template = str(config["poi_embedding_checkpoint_template"])
    seed = int(config.get("seed", 42))
    dimension = int(config["training"]["representation_dim"])
    for fold_id in selected:
        checkpoint = project_path(template.format(fold=fold_id, seed=seed))
        manifest = split_root / f"{fold_id}.parquet"
        output = output_root / fold_id
        if not checkpoint.exists():
            raise FileNotFoundError(f"Source checkpoint is absent: {checkpoint}")
        manifest_rows = len(pd.read_parquet(manifest, columns=["city_id"]))
        print(
            f"fold={fold_id} checkpoint={checkpoint} output={output} "
            f"manifest_rows={manifest_rows} device={device}",
            flush=True,
        )
        if not args.dry_run:
            metadata = build_fold_embeddings(
                fold_id=fold_id,
                panel_path=panel_path,
                manifest_path=manifest,
                poi_root=poi_root,
                checkpoint_path=checkpoint,
                output_dir=output,
                representation_dim=dimension,
                device=device,
                batch_size=args.batch_size,
                force=args.force,
                work_dir=Path(args.work_dir),
            )
            print(
                f"built fold={fold_id} samples={metadata['samples']} "
                f"dimension={metadata['representation_dim']}",
                flush=True,
            )
    print(f"folds={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
