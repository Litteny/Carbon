from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .config import project_path
from .datasets import (
    OpenCarbonDataset,
    PrecomputedOpenCarbonDataset,
    fixed_neighborhood_indices,
)
from .models import OpenCarbonModel
from .poi_embeddings import (
    POIEmbeddingStore,
    expected_metadata as expected_poi_embedding_metadata,
)
from .training import _load_fold, _open_carbon_poi_input_mode
from .utils import resolve_device, sha256_file, write_json


STAGE_ORDER = ("train", "validation", "test")


def _finite_pairs(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    observed = frame["y_log"].to_numpy(dtype=float)
    predicted = frame["pred_log"].to_numpy(dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return observed[valid], predicted[valid]


def calculate_r2_diagnostics(frame: pd.DataFrame) -> Dict[str, object]:
    """Calculate consistently defined log-space diagnostics for one group."""
    observed, predicted = _finite_pairs(frame)
    count = int(len(observed))
    if not count:
        return {
            "n": 0, "r2": None, "r2_status": "undefined_no_finite_samples",
        }
    residual = predicted - observed
    sse = float(np.square(residual).sum())
    centered = observed - observed.mean()
    sst = float(np.square(centered).sum())
    r2_defined = count >= 2 and sst > 0.0
    r2 = float(1.0 - sse / sst) if r2_defined else None
    bias = float(residual.mean())
    adjusted = predicted - bias
    debiased_sse = float(np.square(adjusted - observed).sum())
    debiased_r2 = float(1.0 - debiased_sse / sst) if r2_defined else None
    observed_std = float(observed.std(ddof=0))
    predicted_std = float(predicted.std(ddof=0))
    range_ratio = predicted_std / observed_std if observed_std > 0 else None
    spearman = None
    if count >= 2 and np.unique(observed).size > 1 and np.unique(predicted).size > 1:
        value = float(spearmanr(observed, predicted).statistic)
        spearman = value if math.isfinite(value) else None
    return {
        "n": count,
        "r2": r2,
        "r2_status": "ok" if r2_defined else "undefined_less_than_2_or_constant",
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "spearman": spearman,
        "sse": sse,
        "sst": sst,
        "observed_mean": float(observed.mean()),
        "predicted_mean": float(predicted.mean()),
        "mean_bias": bias,
        "debiased_r2": debiased_r2,
        "observed_std": observed_std,
        "predicted_std": predicted_std,
        "dynamic_range_ratio": range_ratio,
    }


def grouped_r2_diagnostics(
    predictions: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    grouper: object = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, group in predictions.groupby(grouper, sort=True, dropna=False):
        key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
        row = dict(zip(group_columns, key_values))
        row.update(calculate_r2_diagnostics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_defined(values: Iterable[object]) -> Optional[float]:
    numeric = [float(value) for value in values if value is not None and pd.notna(value)]
    return float(np.mean(numeric)) if numeric else None


def _stage_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in STAGE_ORDER:
        group = predictions[predictions["split"] == split]
        row: Dict[str, object] = {"split": split, **calculate_r2_diagnostics(group)}
        monthly = grouped_r2_diagnostics(group, ["period"])
        row["monthly_r2_mean"] = _mean_defined(monthly.get("r2", []))
        row["monthly_debiased_r2_mean"] = _mean_defined(
            monthly.get("debiased_r2", []),
        )
        row["monthly_r2_valid_periods"] = int(monthly.get("r2", pd.Series(dtype=float)).notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def diagnose_r2_stages(summary: pd.DataFrame) -> Dict[str, object]:
    """Identify the first split with negative pooled and monthly-mean R²."""
    ordered = summary.set_index("split").reindex(STAGE_ORDER)

    def first_negative(column: str) -> Optional[str]:
        for split, value in ordered[column].items():
            if value is not None and pd.notna(value) and float(value) < 0.0:
                return str(split)
        return None

    pooled_stage = first_negative("r2")
    monthly_stage = first_negative("monthly_r2_mean")
    interpretations = {
        "train": "The best checkpoint does not beat the train-mean baseline.",
        "validation": "R² first becomes negative within source-domain validation.",
        "test": "R² first becomes negative after transfer to the unseen target city.",
        None: "R² remains non-negative across all defined splits.",
    }
    test_row = ordered.loc["test"]
    test_r2 = test_row.get("r2")
    test_debiased = test_row.get("debiased_r2")
    bias_diagnosis = "not_applicable"
    if pd.notna(test_r2) and float(test_r2) < 0 and pd.notna(test_debiased):
        bias_diagnosis = (
            "mean_level_shift_dominant" if float(test_debiased) >= 0
            else "spatial_or_dynamic_range_failure_remains_after_debiasing"
        )
    monthly_bias_diagnosis = "not_applicable"
    test_monthly_r2 = test_row.get("monthly_r2_mean")
    test_monthly_debiased = test_row.get("monthly_debiased_r2_mean")
    if (
        pd.notna(test_monthly_r2) and float(test_monthly_r2) < 0
        and pd.notna(test_monthly_debiased)
    ):
        monthly_bias_diagnosis = (
            "monthly_mean_level_shift_dominant"
            if float(test_monthly_debiased) >= 0
            else "monthly_spatial_or_dynamic_range_failure_remains_after_debiasing"
        )
    return {
        "pooled_first_negative_split": pooled_stage,
        "pooled_interpretation": interpretations[pooled_stage],
        "monthly_mean_first_negative_split": monthly_stage,
        "monthly_mean_interpretation": interpretations[monthly_stage],
        "test_bias_diagnosis": bias_diagnosis,
        "test_monthly_bias_diagnosis": monthly_bias_diagnosis,
    }


def _load_run_metadata(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Run metadata is absent: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), dict):
        raise ValueError(f"Invalid run metadata: {path}")
    model = str(metadata.get("model", ""))
    if not model.startswith("opencarbon_"):
        raise ValueError(f"R² diagnostics only support OpenCarbon runs, found {model!r}")
    return metadata


def _validate_run_inputs(metadata: Mapping[str, object], config: Mapping[str, object]) -> Path:
    panel_path = project_path(str(config["panel_file"]))
    manifest_path = Path(str(metadata["manifest"]))
    if sha256_file(panel_path) != metadata.get("panel_sha256"):
        raise ValueError("Panel hash does not match run metadata")
    if sha256_file(manifest_path) != metadata.get("manifest_sha256"):
        raise ValueError("Manifest hash does not match run metadata")
    return manifest_path


def _build_inference_components(
    run_dir: Path,
    metadata: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[OpenCarbonModel, Dict[str, pd.DataFrame], Dict[str, DataLoader], bool, bool]:
    config = dict(metadata["config"])
    settings = dict(config["training"])
    fold_id = str(metadata["fold_id"])
    model_name = str(metadata["model"])
    seed = int(metadata["seed"])
    manifest_path = _validate_run_inputs(metadata, config)
    frames, loaded_manifest, _ = _load_fold(config, fold_id)
    if loaded_manifest.resolve() != manifest_path.resolve():
        raise ValueError("Resolved manifest differs from run metadata")

    saved_preprocessor = joblib.load(run_dir / "preprocessor.joblib")
    if not isinstance(saved_preprocessor, dict) or "pipeline" not in saved_preprocessor:
        raise ValueError("OpenCarbon preprocessor metadata is invalid")
    pipeline = saved_preprocessor["pipeline"]
    remote_columns = list(saved_preprocessor["remote_columns"])
    environment_columns = list(saved_preprocessor["environment_columns"])
    feature_columns = remote_columns + environment_columns
    transformed = {
        split: pipeline.transform(frame[feature_columns]).astype(np.float32)
        for split, frame in frames.items()
    }
    remote_size = len(remote_columns)
    is_precomputed = str(saved_preprocessor.get(
        "poi_input_mode", _open_carbon_poi_input_mode(model_name, settings),
    )) == "precomputed"
    neighbors = pd.read_parquet(project_path(str(config["neighbors_file"])))
    neighbors["period"] = neighbors["period"].astype(str)

    embedding_store = None
    if is_precomputed:
        checkpoint_template = str(config["poi_embedding_checkpoint_template"])
        embedding_checkpoint = project_path(checkpoint_template.format(fold=fold_id, seed=seed))
        expected = expected_poi_embedding_metadata(
            fold_id, embedding_checkpoint, project_path(str(config["panel_file"])),
            manifest_path, project_path(str(config["poi_dir"])),
            int(settings["representation_dim"]),
        )
        embedding_store = POIEmbeddingStore(
            project_path(str(config["poi_embedding_dir"])) / fold_id, expected,
        )

    datasets = {}
    for split, values in transformed.items():
        neighborhoods = fixed_neighborhood_indices(frames[split], neighbors)
        if is_precomputed:
            assert embedding_store is not None
            rows = embedding_store.rows_for_frame(frames[split])
            datasets[split] = PrecomputedOpenCarbonDataset(
                frames[split], values[:, :remote_size], values[:, remote_size:],
                neighborhoods, embedding_store.take(rows),
            )
        else:
            datasets[split] = OpenCarbonDataset(
                frames[split], values[:, :remote_size], values[:, remote_size:],
                neighborhoods, project_path(str(config["poi_dir"])),
                poi_cache_size=int(config.get("poi_cache_size", 4)),
            )
    loaders = {
        split: DataLoader(
            dataset, batch_size=int(batch_size), shuffle=False,
            num_workers=int(num_workers), pin_memory=device.type == "cuda",
            collate_fn=dataset.collate,
            **({"persistent_workers": True, "prefetch_factor": 2} if num_workers > 0 else {}),
        )
        for split, dataset in datasets.items()
    }
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    aggregation = str(checkpoint.get(
        "neighborhood_aggregation", settings.get("neighborhood_aggregation", "spatial_attention"),
    ))
    model = OpenCarbonModel(
        remote_size, len(environment_columns), int(settings["representation_dim"]),
        float(settings["dropout"]), aggregation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    amp_enabled = bool(settings.get("amp", True)) and device.type == "cuda"
    return model, frames, loaders, is_precomputed, amp_enabled


def _predict_splits(
    model: OpenCarbonModel,
    frames: Mapping[str, pd.DataFrame],
    loaders: Mapping[str, DataLoader],
    is_precomputed: bool,
    amp_enabled: bool,
    device: torch.device,
) -> pd.DataFrame:
    outputs = []
    with torch.no_grad():
        for split in STAGE_ORDER:
            split_predictions = []
            for batch in loaders[split]:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    if is_precomputed:
                        prediction, _, _ = model.forward_encoded(
                            batch["poi_embedding"], batch["remote"], batch["environment"],
                            batch["neighborhood_indices"], batch["neighborhood_mask"],
                        )
                    else:
                        prediction, _, _ = model(
                            batch["poi"], batch["remote"], batch["environment"],
                            batch["neighborhood_indices"], batch["neighborhood_mask"],
                        )
                split_predictions.append(prediction.float().cpu().numpy())
            frame = frames[split][["city_id", "period", "cell_id", "log1p_emission"]].copy()
            frame = frame.rename(columns={"log1p_emission": "y_log"})
            frame["pred_log"] = np.concatenate(split_predictions)
            frame.insert(0, "split", split)
            outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def _format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "undefined"
    return f"{float(value):.4f}"


def _write_markdown(
    path: Path, metadata: Mapping[str, object], summary: pd.DataFrame,
    diagnosis: Mapping[str, object],
) -> None:
    lines = [
        "# OpenCarbon R² stage diagnosis", "",
        f"- Fold: `{metadata['fold_id']}`", f"- Model: `{metadata['model']}`",
        f"- Seed: `{metadata['seed']}`", "", "## Stage summary", "",
        "| split | n | pooled R² | monthly mean R² | monthly debiased R² | MAE | debiased R² | bias | range ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.n} | {_format_value(row.r2)} | "
            f"{_format_value(row.monthly_r2_mean)} | "
            f"{_format_value(row.monthly_debiased_r2_mean)} | {_format_value(row.mae)} | "
            f"{_format_value(row.debiased_r2)} | {_format_value(row.mean_bias)} | "
            f"{_format_value(row.dynamic_range_ratio)} |"
        )
    lines.extend([
        "", "## Automatic diagnosis", "",
        f"- Pooled R² first negative split: `{diagnosis['pooled_first_negative_split']}`",
        f"- {diagnosis['pooled_interpretation']}",
        f"- Monthly-mean R² first negative split: "
        f"`{diagnosis['monthly_mean_first_negative_split']}`",
        f"- {diagnosis['monthly_mean_interpretation']}",
        f"- Test bias diagnosis: `{diagnosis['test_bias_diagnosis']}`", "",
        f"- Test monthly bias diagnosis: `{diagnosis['test_monthly_bias_diagnosis']}`", "",
        "> R² is diagnostic only; model selection remains based on validation MAE.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose_open_carbon_run(
    run_dir: Path,
    device_name: str = "auto",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    save_predictions: bool = False,
    force: bool = False,
) -> Path:
    run_dir = Path(run_dir).resolve()
    metadata = _load_run_metadata(run_dir)
    config = dict(metadata["config"])
    settings = dict(config["training"])
    output_dir = run_dir / "diagnostics" / "r2"
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"R² diagnostics already exist: {output_dir}; pass --force to replace them"
            )
        shutil.rmtree(output_dir)
    device = resolve_device(device_name)
    effective_batch_size = int(batch_size or settings.get("batch_size", 16))
    effective_workers = int(
        settings.get("num_workers", 0) if num_workers is None else num_workers
    )
    model, frames, loaders, is_precomputed, amp_enabled = _build_inference_components(
        run_dir, metadata, device, effective_batch_size, effective_workers,
    )
    predictions = _predict_splits(
        model, frames, loaders, is_precomputed, amp_enabled, device,
    )
    summary = _stage_summary(predictions)
    by_city = grouped_r2_diagnostics(predictions, ["split", "city_id"])
    by_month = grouped_r2_diagnostics(predictions, ["split", "period"])
    by_city_month = grouped_r2_diagnostics(
        predictions, ["split", "city_id", "period"],
    )
    diagnosis = diagnose_r2_stages(summary)

    output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output_dir / "r2_stage_summary.csv", index=False)
    by_city.to_csv(output_dir / "r2_by_city.csv", index=False)
    by_month.to_csv(output_dir / "r2_by_month.csv", index=False)
    by_city_month.to_csv(output_dir / "r2_by_city_month.csv", index=False)
    payload = {
        "run_dir": str(run_dir),
        "fold_id": metadata["fold_id"],
        "model": metadata["model"],
        "seed": metadata["seed"],
        "checkpoint": str(run_dir / "best.pt"),
        "poi_input_mode": "precomputed" if is_precomputed else "dense",
        "batch_size": effective_batch_size,
        "num_workers": effective_workers,
        "amp_enabled": amp_enabled,
        "diagnosis": diagnosis,
        "stages": summary.where(pd.notna(summary), None).to_dict("records"),
    }
    write_json(output_dir / "r2_diagnosis.json", payload)
    _write_markdown(output_dir / "r2_diagnosis.md", metadata, summary, diagnosis)
    if save_predictions:
        predictions.to_parquet(output_dir / "split_predictions.parquet", index=False)
    return output_dir
