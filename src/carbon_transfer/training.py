from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import project_path
from .constants import (
    MISSING_COLUMNS, MODEL_NAMES, MODIS_COLUMNS, MONTH_COLUMNS, POI_COLUMNS,
    STAGE1_B0_MODEL, STAGE1_FEATURE_COLUMNS, STAGE1_MODEL_NAMES, STAGE1_M1_MODEL,
    TABULAR_FEATURES, VIIRS_COLUMNS, WEATHER_COLUMNS,
)
from .datasets import (
    BalancedEnvironmentBatchSampler,
    OpenCarbonDataset,
    PrecomputedOpenCarbonDataset,
    Stage1OpenCarbonDataset,
    TabularOpenCarbonDataset,
    fixed_neighborhood_indices,
    normalized_adjacency,
)
from .metrics import (
    calculate_admin_metrics, calculate_annual_admin_metrics, calculate_metrics,
    calculate_pooled_metrics, calculate_yearly_metrics, prediction_frame,
)
from .models import BPNN, CarbonGCN, OpenCarbonModel, Stage1OpenCarbonModel, contrastive_loss
from .poi_embeddings import (
    POIEmbeddingStore,
    expected_metadata as expected_poi_embedding_metadata,
    extract_poi_encoder_state,
    validate_embedding_cache,
)
from .progress import EpochProgress, RunProgress, write_training_log
from .utils import resolve_device, set_seed, sha256_file, write_json


def _validation_r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return whole-validation-set R² in log space, or NaN when undefined."""
    predicted = predictions.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    observed = targets.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    if predicted.numel() < 2 or predicted.numel() != observed.numel():
        return float("nan")
    if not torch.isfinite(predicted).all() or not torch.isfinite(observed).all():
        return float("nan")
    total_sum_squares = torch.sum((observed - observed.mean()).square())
    if total_sum_squares <= 0:
        return float("nan")
    residual_sum_squares = torch.sum((observed - predicted).square())
    value = 1.0 - residual_sum_squares / total_sum_squares
    return float(value.item()) if torch.isfinite(value) else float("nan")


def _resolve_manifest(split_dir: Path, fold_id: str) -> Tuple[Path, str]:
    direct = split_dir / f"{fold_id}.parquet"
    if direct.exists():
        return direct, direct.parent.name
    matches = list(split_dir.rglob(f"{fold_id}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one split manifest for {fold_id}, found {len(matches)}")
    return matches[0], matches[0].parent.name


def _load_fold(config: Dict, fold_id: str) -> Tuple[Dict[str, pd.DataFrame], Path, str]:
    panel_path = project_path(config["panel_file"])
    manifest_path, experiment = _resolve_manifest(project_path(config["split_dir"]), fold_id)
    panel = pd.read_parquet(panel_path)
    panel["period"] = panel["period"].astype(str)
    manifest = pd.read_parquet(manifest_path, columns=["city_id", "cell_id", "period", "split"])
    manifest["period"] = manifest["period"].astype(str)
    joined = manifest.merge(panel, on=["city_id", "cell_id", "period"], how="left", validate="one_to_one")
    if joined["log1p_emission"].isna().any():
        raise ValueError("Split manifest contains keys absent from the panel")
    frames = {
        name: group.drop(columns="split").reset_index(drop=True)
        for name, group in joined.groupby("split", sort=False)
    }
    if set(frames) != {"train", "validation", "test"}:
        raise ValueError(f"Fold {fold_id} does not contain train/validation/test")
    limit = config.get("smoke_max_rows_per_split")
    if limit:
        limit = int(limit)
        limited = {}
        for name, frame in frames.items():
            candidate = frame.copy()
            candidate["_smoke_rank"] = candidate.groupby("city_id").cumcount()
            candidate = candidate.sort_values(["_smoke_rank", "city_id"]).head(limit)
            limited[name] = candidate.drop(columns="_smoke_rank").reset_index(drop=True)
        frames = limited
    return frames, manifest_path, experiment


def _load_stage1_fold(
    config: Dict, fold_id: str,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Path, str]:
    """Load a stage-1 fold plus the complete label-free context frame."""
    panel_path = project_path(config["panel_file"])
    manifest_path, experiment = _resolve_manifest(project_path(config["split_dir"]), fold_id)
    panel = pd.read_parquet(panel_path)
    panel["period"] = panel["period"].astype(str)
    manifest = pd.read_parquet(manifest_path, columns=["city_id", "cell_id", "period", "split"])
    manifest["period"] = manifest["period"].astype(str)
    joined = manifest.merge(
        panel, on=["city_id", "cell_id", "period"], how="left", validate="one_to_one",
    )
    if joined["log1p_emission"].isna().any():
        raise ValueError("Stage1 split manifest contains keys absent from the panel")
    frames = {
        name: group.drop(columns="split").reset_index(drop=True)
        for name, group in joined.groupby("split", sort=False)
    }
    if set(frames) != {"train", "validation", "test"}:
        raise ValueError(f"Fold {fold_id} does not contain train/validation/test")
    full_frame = joined.drop(columns="split").sort_values(
        ["city_id", "period", "cell_id"],
    ).reset_index(drop=True)
    if full_frame["city_id"].astype(str).nunique() != 1:
        raise ValueError("Stage1 within-city fold must contain exactly one city")
    return frames, full_frame, manifest_path, experiment


def _period_ids(periods: pd.Series) -> np.ndarray:
    values = periods.astype(str)
    parsed = values.str[:4].astype(int) * 12 + values.str[4:6].astype(int) - (2021 * 12 + 1)
    result = parsed.to_numpy(dtype=np.int64)
    if np.any(result < 0) or np.any(result >= 36):
        raise ValueError("Stage1 period IDs must cover 2021-01 through 2023-12")
    return result


def _stage1_feature_mask(model_name: str) -> np.ndarray:
    if model_name not in STAGE1_MODEL_NAMES:
        raise ValueError(f"Unknown stage1 model: {model_name}")
    allowed = set(VIIRS_COLUMNS)
    if model_name == STAGE1_M1_MODEL:
        allowed.update(MODIS_COLUMNS)
        allowed.update(WEATHER_COLUMNS)
        allowed.update(
            f"{name}_is_missing" for name in MODIS_COLUMNS + WEATHER_COLUMNS
        )
    return np.asarray([column in allowed for column in STAGE1_FEATURE_COLUMNS], dtype=bool)


def _pipeline(scale: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _validate_neighborhood_aggregation(checkpoint: Dict, configured: str) -> None:
    checkpoint_aggregation = str(checkpoint.get(
        "neighborhood_aggregation", "spatial_attention",
    ))
    if checkpoint_aggregation != configured:
        raise ValueError(
            f"Checkpoint neighborhood_aggregation={checkpoint_aggregation} does not match "
            f"configured neighborhood_aggregation={configured}"
        )


def _environment_mapping(frame: pd.DataFrame, environment_key: str) -> Dict[str, int]:
    if environment_key not in frame:
        raise ValueError(f"Environment column is absent: {environment_key}")
    values = sorted(frame[environment_key].astype(str).unique().tolist())
    if len(values) < 2:
        raise ValueError("VREx training requires at least two source environments")
    return {value: index for index, value in enumerate(values)}


def _vrex_objective(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    environment_ids: torch.Tensor,
    invariance_weight: float,
    penalty_active: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    environments = torch.unique(environment_ids, sorted=True)
    if len(environments) < 2 or torch.any(environments < 0):
        raise ValueError("Each VREx batch must contain at least two known environments")
    risks = torch.stack([
        torch.mean(torch.abs(predictions[environment_ids == environment] - targets[
            environment_ids == environment
        ]))
        for environment in environments
    ])
    environment_mae = risks.mean()
    risk_variance = risks.var(unbiased=False)
    penalty = risk_variance * float(invariance_weight) if penalty_active else risk_variance * 0.0
    return environment_mae + penalty, environment_mae, risk_variance, penalty


def _environment_macro_validation_mae(
    absolute_errors: torch.Tensor,
    environment_ids: torch.Tensor,
) -> Tuple[float, Dict[int, float]]:
    environments = torch.unique(environment_ids, sorted=True)
    if len(environments) < 2 or torch.any(environments < 0):
        raise ValueError("VREx validation requires at least two known environments")
    values = {
        int(environment.item()): float(absolute_errors[
            environment_ids == environment
        ].mean().item())
        for environment in environments
    }
    return float(np.mean(list(values.values()))), values


def _validate_vrex_checkpoint(
    checkpoint: Mapping[str, object],
    environment_mapping: Mapping[str, int],
    settings: Mapping[str, object],
) -> None:
    expected = {
        "training_strategy": "vrex",
        "environment_mapping": dict(environment_mapping),
        "environment_key": str(settings.get("environment_key", "city_id")),
        "invariance_weight": float(settings.get("invariance_weight", 1.0)),
        "invariance_warmup_epochs": int(settings.get("invariance_warmup_epochs", 2)),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Checkpoint {key}={checkpoint.get(key)!r} does not match configured {value!r}"
            )


def _environment_checkpoint_fields(
    is_vrex: bool,
    environment_mapping: Mapping[str, int],
    settings: Mapping[str, object],
) -> Dict[str, object]:
    if not is_vrex:
        return {}
    return {
        "training_strategy": "vrex",
        "environment_mapping": dict(environment_mapping),
        "environment_key": str(settings.get("environment_key", "city_id")),
        "invariance_weight": float(settings.get("invariance_weight", 1.0)),
        "invariance_warmup_epochs": int(settings.get("invariance_warmup_epochs", 2)),
    }


def _precomputed_checkpoint_fields(
    is_precomputed: bool,
    embedding_metadata: Mapping[str, object],
) -> Dict[str, object]:
    if not is_precomputed:
        return {}
    return {
        "poi_input_mode": "precomputed",
        "poi_embedding_checkpoint_sha256": embedding_metadata["checkpoint_sha256"],
        "poi_embedding_metadata": dict(embedding_metadata),
    }


def _validate_precomputed_checkpoint(
    checkpoint: Mapping[str, object],
    embedding_metadata: Mapping[str, object],
) -> None:
    expected = _precomputed_checkpoint_fields(True, embedding_metadata)
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Checkpoint {key}={checkpoint.get(key)!r} does not match configured {value!r}"
            )


def _torch_train_tabular(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    settings: Dict,
    device: torch.device,
    run_dir: Path,
    fold_id: str = "unknown",
    model_name: str = "bpnn",
    seed: int = 42,
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=int(settings["batch_size"]), shuffle=True,
    )
    validation_tensor = torch.from_numpy(validation_x.astype(np.float32)).to(device)
    validation_target = torch.from_numpy(validation_y.astype(np.float32)).to(device)
    best_loss, best_epoch, stale, start_epoch = float("inf"), -1, 0, 0
    log = []
    events = []
    latest_path = run_dir / "latest.pt"
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        stale = int(checkpoint.get("stale", 0))
        log = checkpoint.get("log", [])
    max_epochs = int(settings["max_epochs"])
    with EpochProgress(fold_id, model_name, seed, start_epoch, max_epochs) as progress:
        for epoch in range(start_epoch, max_epochs):
            model.train()
            train_losses = []
            for features, targets in loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(features)
                loss = torch.mean(torch.abs(predictions - targets))
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                validation_prediction = model(validation_tensor)
                validation_loss = float(torch.mean(torch.abs(
                    validation_prediction - validation_target
                )).cpu())
            entry = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_mae": validation_loss,
                "validation_r2": _validation_r2(validation_prediction, validation_target),
            }
            log.append(entry)
            if validation_loss < best_loss - 1e-8:
                best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
                torch.save({"model_state": model.state_dict(), "epoch": best_epoch}, run_dir / "best.pt")
            else:
                stale += 1
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1, "best_loss": best_loss, "stale": stale, "log": log,
            }, latest_path)
            events.append(progress.update(entry, best_loss, stale))
            write_training_log(run_dir / "training_log.json", events, log)
            if stale >= int(settings["patience"]):
                RunProgress().early_stop(fold_id, model_name, seed, epoch + 1, stale)
                break
    write_training_log(run_dir / "training_log.json", events, log)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return model


def _train_lightgbm(frames: Dict[str, pd.DataFrame], run_dir: Path, fold_id: str, model_name: str, seed: int) -> np.ndarray:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    features = TABULAR_FEATURES
    preprocessor = _pipeline(scale=False)
    train_x = preprocessor.fit_transform(frames["train"][features])
    validation_x = preprocessor.transform(frames["validation"][features])
    test_x = preprocessor.transform(frames["test"][features])
    progress = RunProgress()
    progress.lightgbm_start(fold_id, model_name, seed)
    model = LGBMRegressor(
        objective="regression_l1", n_estimators=1000, learning_rate=0.03,
        num_leaves=31, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        train_x, frames["train"]["log1p_emission"],
        eval_set=[(validation_x, frames["validation"]["log1p_emission"])],
        eval_metric="l1", callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    joblib.dump(model, run_dir / "model.joblib")
    best_iteration = int(model.best_iteration_ or 0)
    event = progress.lightgbm_end(fold_id, model_name, seed, best_iteration)
    write_training_log(run_dir / "training_log.json", [event], [{"best_iteration": best_iteration}])
    return model.predict(test_x)


def _train_bpnn(
    frames: Dict[str, pd.DataFrame], settings: Dict, device: torch.device, run_dir: Path, fold_id: str, seed: int,
) -> np.ndarray:
    preprocessor = _pipeline(scale=True)
    train_x = preprocessor.fit_transform(frames["train"][TABULAR_FEATURES]).astype(np.float32)
    validation_x = preprocessor.transform(frames["validation"][TABULAR_FEATURES]).astype(np.float32)
    test_x = preprocessor.transform(frames["test"][TABULAR_FEATURES]).astype(np.float32)
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    model = BPNN(train_x.shape[1], int(settings["representation_dim"]), float(settings["dropout"]))
    model = _torch_train_tabular(
        model, train_x, frames["train"]["log1p_emission"].to_numpy(),
        validation_x, frames["validation"]["log1p_emission"].to_numpy(),
        settings, device, run_dir, fold_id, "bpnn", seed,
    )
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(test_x).to(device)).cpu().numpy()


def _train_gcn(
    frames: Dict[str, pd.DataFrame], neighbors: pd.DataFrame, settings: Dict,
    device: torch.device, run_dir: Path, fold_id: str, seed: int,
) -> np.ndarray:
    preprocessor = _pipeline(scale=True)
    arrays = {"train": preprocessor.fit_transform(frames["train"][TABULAR_FEATURES]).astype(np.float32)}
    arrays.update({
        name: preprocessor.transform(frames[name][TABULAR_FEATURES]).astype(np.float32)
        for name in ("validation", "test")
    })
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    tensors = {name: torch.from_numpy(array).to(device) for name, array in arrays.items()}
    targets = {
        name: torch.from_numpy(frames[name]["log1p_emission"].to_numpy(dtype=np.float32)).to(device)
        for name in frames
    }
    adjacency = {name: normalized_adjacency(frames[name], neighbors, device) for name in frames}
    model = CarbonGCN(
        arrays["train"].shape[1], int(settings["representation_dim"]), float(settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]),
    )
    best_loss, stale, start_epoch = float("inf"), 0, 0
    log = []
    events = []
    latest_path = run_dir / "latest.pt"
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        stale = int(checkpoint.get("stale", 0))
        log = checkpoint.get("log", [])
    max_epochs = int(settings["max_epochs"])
    with EpochProgress(fold_id, "carbongcn", seed, start_epoch, max_epochs) as progress:
        for epoch in range(start_epoch, max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tensors["train"], adjacency["train"])
            loss = torch.mean(torch.abs(prediction - targets["train"]))
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_prediction = model(
                    tensors["validation"], adjacency["validation"],
                )
                validation_loss = float(torch.mean(torch.abs(
                    validation_prediction - targets["validation"]
                )).cpu())
            entry = {
                "epoch": epoch + 1,
                "train_loss": float(loss.detach().cpu()),
                "validation_mae": validation_loss,
                "validation_r2": _validation_r2(
                    validation_prediction, targets["validation"],
                ),
            }
            log.append(entry)
            if validation_loss < best_loss - 1e-8:
                best_loss, stale = validation_loss, 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch + 1}, run_dir / "best.pt")
            else:
                stale += 1
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1, "best_loss": best_loss, "stale": stale, "log": log,
            }, latest_path)
            events.append(progress.update(entry, best_loss, stale))
            write_training_log(run_dir / "training_log.json", events, log)
            if stale >= int(settings["patience"]):
                RunProgress().early_stop(fold_id, "carbongcn", seed, epoch + 1, stale)
                break
    write_training_log(run_dir / "training_log.json", events, log)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)["model_state"])
    model.eval()
    with torch.no_grad():
        return model(tensors["test"], adjacency["test"]).cpu().numpy()


def _open_carbon_feature_sets(model_name: str, ablation: str = "baseline") -> Tuple[List[str], List[str]]:
    no_viirs = model_name == "opencarbon_monthly_noviirs"
    monthly = model_name != "opencarbon_core"
    remote = MODIS_COLUMNS + [f"{name}_is_missing" for name in MODIS_COLUMNS]
    if not no_viirs:
        remote += VIIRS_COLUMNS
    environment = WEATHER_COLUMNS + [f"{name}_is_missing" for name in WEATHER_COLUMNS]
    if monthly:
        environment += MONTH_COLUMNS
    if ablation == "no_modis":
        remote = [name for name in remote if name not in MODIS_COLUMNS and name not in {
            f"{value}_is_missing" for value in MODIS_COLUMNS
        }]
    elif ablation == "no_viirs":
        remote = [name for name in remote if name not in VIIRS_COLUMNS]
    elif ablation == "no_weather":
        environment = [name for name in environment if name not in WEATHER_COLUMNS and name not in {
            f"{value}_is_missing" for value in WEATHER_COLUMNS
        }]
    elif ablation == "no_month":
        environment = [name for name in environment if name not in MONTH_COLUMNS]
    elif ablation not in {"baseline", "no_poi"}:
        raise ValueError(f"Unknown input_ablation={ablation!r}")
    return remote, environment


def _open_carbon_features_for_scope(
    model_name: str, scope: str, ablation: str = "baseline",
) -> Tuple[List[str], List[str]]:
    """Return explicitly scoped OpenCarbon inputs for protocol-level experiments."""
    if scope == "poi_modis":
        remote = list(MODIS_COLUMNS)
        if ablation == "no_poi":
            return remote, []
        if ablation != "baseline":
            raise ValueError(f"Unknown input_ablation={ablation!r}")
        return remote, []
    return _open_carbon_feature_sets(model_name, ablation)


def _open_carbon_poi_input_mode(
    model_name: str, settings: Mapping[str, object],
) -> str:
    """Resolve POI input mode while preserving legacy precomputed model IDs."""
    if (
        model_name in {"opencarbon_monthly_precomputed", "opencarbon_monthly_vrex"}
        and str(settings.get("poi_input_mode", "dense")) != "tabular"
    ):
        return "precomputed"
    mode = str(settings.get("poi_input_mode", "dense"))
    if mode not in {"dense", "precomputed", "tabular", "none"}:
        raise ValueError(
            f"Unknown poi_input_mode={mode!r}; expected 'dense', 'precomputed', 'tabular', or 'none'"
        )
    return mode


def _train_stage1_open_carbon(
    model_name: str,
    fold_id: str,
    frames: Dict[str, pd.DataFrame],
    full_frame: pd.DataFrame,
    neighbors: pd.DataFrame,
    config: Dict,
    settings: Dict,
    device: torch.device,
    run_dir: Path,
    seed: int,
) -> np.ndarray:
    """Train the POI-free B0/M1 model with full-panel feature context."""
    active_mask = _stage1_feature_mask(model_name)
    preprocessor = _pipeline(scale=True)
    preprocessor.fit(
        frames["train"][list(STAGE1_FEATURE_COLUMNS)],
    )
    full_values = preprocessor.transform(
        full_frame[list(STAGE1_FEATURE_COLUMNS)],
    ).astype(np.float32)
    full_values[:, ~active_mask] = 0.0
    period_ids = _period_ids(full_frame["period"])
    context_frame = full_frame[["city_id", "cell_id", "period"]].copy()
    neighborhoods = fixed_neighborhood_indices(context_frame, neighbors)
    joblib.dump({
        "pipeline": preprocessor,
        "feature_columns": list(STAGE1_FEATURE_COLUMNS),
        "active_feature_columns": [
            column for column, active in zip(STAGE1_FEATURE_COLUMNS, active_mask) if active
        ],
        "context_scope": "full_city_same_period_open_features_no_labels",
        "target_transform": "log1p_emission_tc",
    }, run_dir / "preprocessor.joblib")

    # The model receives a fixed-width feature vector for both B0 and M1;
    # B0's disallowed feature positions are neutralized after train-fitted scaling.
    datasets = {
        name: Stage1OpenCarbonDataset(
            frames[name], context_frame, full_values, period_ids, neighborhoods,
        )
        for name in ("train", "validation", "test")
    }
    num_workers = int(settings.get("num_workers", 0))
    worker_options = {}
    if num_workers > 0:
        worker_options = {
            "persistent_workers": bool(settings.get("persistent_workers", True)),
            "prefetch_factor": int(settings.get("prefetch_factor", 2)),
        }
    loaders = {}
    for name, dataset in datasets.items():
        loaders[name] = DataLoader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=dataset.collate,
            **worker_options,
        )

    model = Stage1OpenCarbonModel(
        len(STAGE1_FEATURE_COLUMNS),
        int(settings["representation_dim"]),
        int(settings.get("time_embedding_dim", 32)),
        36,
        float(settings["dropout"]),
        str(settings.get("neighborhood_aggregation", "mean_mlp_gate")),
    ).to(device)
    neighborhood_aggregation = model.neighborhood_aggregation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    amp_enabled = bool(settings.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation = int(settings.get("gradient_accumulation", 1))
    best_loss, stale, start_epoch = float("inf"), 0, 0
    latest_path = run_dir / "latest.pt"
    log = []
    events = []
    checkpoint_fields = {
        "stage1_model": model_name,
        "neighborhood_aggregation": neighborhood_aggregation,
        "active_feature_columns": [
            column for column, active in zip(STAGE1_FEATURE_COLUMNS, active_mask) if active
        ],
        "time_embedding_dim": int(settings.get("time_embedding_dim", 32)),
        "num_periods": 36,
    }
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("stage1_model") != model_name:
            raise ValueError("Stage1 checkpoint model does not match the configured model")
        _validate_neighborhood_aggregation(checkpoint, neighborhood_aggregation)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        stale = int(checkpoint.get("stale", stale))
        log = checkpoint.get("log", [])

    def forward_batch(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return model(
            batch["features"], batch["period_id"],
            batch["neighborhood_indices"], batch["neighborhood_mask"],
        )

    max_epochs = int(settings["max_epochs"])
    with EpochProgress(fold_id, model_name, seed, start_epoch, max_epochs) as progress:
        for epoch in range(start_epoch, max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_losses = []
            for step, batch in enumerate(loaders["train"]):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    prediction = forward_batch(batch)
                    loss = torch.mean(torch.abs(prediction - batch["target"]))
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loaders["train"]):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            validation_predictions = []
            validation_targets = []
            with torch.no_grad():
                for batch in loaders["validation"]:
                    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        prediction = forward_batch(batch)
                    validation_predictions.append(prediction.float().cpu())
                    validation_targets.append(batch["target"].float().cpu())
            validation_prediction = torch.cat(validation_predictions)
            validation_target = torch.cat(validation_targets)
            validation_loss = float(torch.mean(torch.abs(
                validation_prediction - validation_target,
            )).item())
            entry = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_mae": validation_loss,
                "validation_r2": _validation_r2(validation_prediction, validation_target),
            }
            log.append(entry)
            if validation_loss < best_loss - 1e-8:
                best_loss, stale = validation_loss, 0
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    **checkpoint_fields,
                }, run_dir / "best.pt")
            else:
                stale += 1
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "stale": stale,
                "log": log,
                **checkpoint_fields,
            }, latest_path)
            events.append(progress.update(entry, best_loss, stale))
            write_training_log(run_dir / "training_log.json", events, log)
            if stale >= int(settings["patience"]):
                RunProgress().early_stop(fold_id, model_name, seed, epoch + 1, stale)
                break

    model.load_state_dict(torch.load(
        run_dir / "best.pt", map_location=device, weights_only=True,
    )["model_state"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loaders["test"]:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = forward_batch(batch)
            predictions.append(prediction.float().cpu().numpy())
    return np.concatenate(predictions)


def _train_open_carbon(
    model_name: str,
    fold_id: str,
    frames: Dict[str, pd.DataFrame],
    neighbors: pd.DataFrame,
    config: Dict,
    settings: Dict,
    device: torch.device,
    run_dir: Path,
    seed: int,
) -> np.ndarray:
    is_vrex = model_name == "opencarbon_monthly_vrex"
    is_precomputed = _open_carbon_poi_input_mode(model_name, settings) == "precomputed"
    is_tabular = _open_carbon_poi_input_mode(model_name, settings) == "tabular"
    environment_key = str(settings.get("environment_key", "city_id"))
    environment_mapping = (
        _environment_mapping(frames["train"], environment_key) if is_vrex else {}
    )
    input_ablation = str(settings.get("input_ablation", "baseline"))
    input_scope = str(settings.get("input_scope", "default"))
    remote_columns, environment_columns = _open_carbon_features_for_scope(
        model_name, input_scope, input_ablation,
    )
    poi_input_mode = str(settings.get("poi_input_mode", "dense"))
    feature_columns = (
        POI_COLUMNS + remote_columns + environment_columns
        if poi_input_mode == "tabular"
        else remote_columns + environment_columns
    )
    preprocessor = _pipeline(scale=True)
    transformed = {
        "train": preprocessor.fit_transform(frames["train"][feature_columns]).astype(np.float32),
    }
    transformed.update({
        name: preprocessor.transform(frames[name][feature_columns]).astype(np.float32)
        for name in ("validation", "test")
    })
    joblib.dump({
        "pipeline": preprocessor,
        "remote_columns": remote_columns,
        "environment_columns": environment_columns,
        "environment_key": environment_key if is_vrex else None,
        "environment_mapping": environment_mapping,
        "poi_input_mode": str(settings.get("poi_input_mode", "dense")),
        "input_scope": input_scope,
    }, run_dir / "preprocessor.joblib")
    poi_size = len(POI_COLUMNS)
    remote_size = len(remote_columns)
    representation_dim = int(settings["representation_dim"])
    embedding_metadata: Dict[str, object] = {}
    embedding_store = None
    if is_precomputed:
        embedding_root = project_path(config["poi_embedding_dir"])
        checkpoint_template = str(config["poi_embedding_checkpoint_template"])
        embedding_checkpoint = project_path(checkpoint_template.format(fold=fold_id, seed=seed))
        panel_path = project_path(config["panel_file"])
        manifest_path, _ = _resolve_manifest(project_path(config["split_dir"]), fold_id)
        expected = expected_poi_embedding_metadata(
            fold_id,
            embedding_checkpoint,
            panel_path,
            manifest_path,
            project_path(config["poi_dir"]),
            representation_dim,
        )
        embedding_store = POIEmbeddingStore(embedding_root / fold_id, expected)
        embedding_metadata = dict(embedding_store.metadata)
    datasets = {}
    for name, values in transformed.items():
        neighborhoods = fixed_neighborhood_indices(frames[name], neighbors)
        if poi_input_mode == "tabular":
            start = 0
            end = poi_size
            datasets[name] = TabularOpenCarbonDataset(
                frames[name], values[:, start:end], values[:, end:end + remote_size],
                values[:, end + remote_size:], neighborhoods,
                environment_mapping=environment_mapping, environment_key=environment_key,
            )
        elif is_precomputed:
            assert embedding_store is not None
            embedding_rows = embedding_store.rows_for_frame(frames[name])
            datasets[name] = PrecomputedOpenCarbonDataset(
                frames[name], values[:, :remote_size], values[:, remote_size:], neighborhoods,
                embedding_store.take(embedding_rows),
                environment_mapping=environment_mapping,
                environment_key=environment_key,
            )
        else:
            datasets[name] = OpenCarbonDataset(
                frames[name], values[:, :remote_size], values[:, remote_size:], neighborhoods,
                project_path(config["poi_dir"]),
                poi_cache_size=int(config.get("poi_cache_size", 4)),
                environment_mapping=environment_mapping,
                environment_key=environment_key,
                zero_poi=input_ablation == "no_poi",
            )
    num_workers = int(settings.get("num_workers", 0))
    worker_options = {}
    if num_workers > 0:
        worker_options = {
            "persistent_workers": bool(settings.get("persistent_workers", True)),
            "prefetch_factor": int(settings.get("prefetch_factor", 2)),
        }
    loaders = {}
    train_sampler = None
    for name, dataset in datasets.items():
        loader_options = {
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
            "collate_fn": dataset.collate,
            **worker_options,
        }
        if name == "train" and is_vrex:
            train_sampler = BalancedEnvironmentBatchSampler(
                dataset.environment_ids,
                int(settings["batch_size"]),
                seed,
            )
            loaders[name] = DataLoader(
                dataset,
                batch_sampler=train_sampler,
                **loader_options,
            )
        else:
            loaders[name] = DataLoader(
                dataset,
                batch_size=int(settings["batch_size"]),
                shuffle=name == "train",
                **loader_options,
            )
    train_evaluation_loader = None
    if bool(settings.get("record_train_r2", False)):
        train_evaluation_loader = DataLoader(
            datasets["train"],
            batch_size=int(settings["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=datasets["train"].collate,
            **worker_options,
        )
    model = OpenCarbonModel(
        remote_size, len(environment_columns),
        representation_dim, float(settings["dropout"]),
        str(settings.get("neighborhood_aggregation", "spatial_attention")),
        poi_input_mode=poi_input_mode,
    ).to(device)
    if is_precomputed:
        assert embedding_store is not None
        checkpoint_path = Path(str(embedding_store.metadata["checkpoint_path"]))
        encoder_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.poi_encoder.load_state_dict(extract_poi_encoder_state(encoder_checkpoint), strict=True)
        model.poi_encoder.requires_grad_(False)
        model.poi_encoder.eval()
    neighborhood_aggregation = model.neighborhood_aggregation
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]),
    )
    amp_enabled = bool(settings.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation = int(settings.get("gradient_accumulation", 1))
    best_loss, stale, start_epoch = float("inf"), 0, 0
    latest_path = run_dir / "latest.pt"
    log = []
    events = []
    checkpoint_fields = _environment_checkpoint_fields(
        is_vrex, environment_mapping, settings,
    )
    checkpoint_fields.update(_precomputed_checkpoint_fields(
        is_precomputed, embedding_metadata,
    ))
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        _validate_neighborhood_aggregation(checkpoint, neighborhood_aggregation)
        if is_vrex:
            _validate_vrex_checkpoint(checkpoint, environment_mapping, settings)
        if is_tabular:
            pass
        elif is_precomputed:
            _validate_precomputed_checkpoint(checkpoint, embedding_metadata)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        stale = int(checkpoint.get("stale", 0))
        log = checkpoint.get("log", [])

    max_epochs = int(settings["max_epochs"])
    with EpochProgress(fold_id, model_name, seed, start_epoch, max_epochs) as progress:
        for epoch in range(start_epoch, max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            if is_precomputed:
                model.poi_encoder.eval()
            optimizer.zero_grad(set_to_none=True)
            train_losses = []
            for step, batch in enumerate(loaders["train"]):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    if is_precomputed:
                        prediction, poi_representation, remote_representation = model.forward_encoded(
                            batch["poi_embedding"], batch["remote"], batch["environment"],
                            batch["neighborhood_indices"], batch["neighborhood_mask"],
                        )
                    else:
                        prediction, poi_representation, remote_representation = model(
                            batch["poi"], batch["remote"], batch["environment"],
                            batch["neighborhood_indices"], batch["neighborhood_mask"],
                        )
                    if is_vrex:
                        _, environment_mae, risk_variance, invariance_penalty = (
                            _vrex_objective(
                                prediction,
                                batch["target"],
                                batch["environment_id"],
                                float(settings.get("invariance_weight", 1.0)),
                                epoch >= int(settings.get("invariance_warmup_epochs", 2)),
                            )
                        )
                        regression = environment_mae
                    else:
                        regression = torch.mean(torch.abs(prediction - batch["target"]))
                        environment_mae = regression
                        risk_variance = torch.zeros((), device=device)
                        invariance_penalty = torch.zeros((), device=device)
                    if epoch >= int(settings["contrastive_warmup_epochs"]) and len(prediction) > 1:
                        contrast = contrastive_loss(
                            poi_representation, remote_representation,
                            float(settings.get("temperature", 0.07)),
                        )
                        loss = (
                            regression
                            + invariance_penalty
                            + float(settings["contrastive_weight"]) * contrast
                        )
                    else:
                        contrast = torch.zeros((), device=device)
                        loss = regression + invariance_penalty
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loaders["train"]):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                train_losses.append((
                    float(loss.detach().cpu()), float(regression.detach().cpu()),
                    float(contrast.detach().cpu()),
                    float(environment_mae.detach().cpu()),
                    float(risk_variance.detach().cpu()),
                    float(invariance_penalty.detach().cpu()),
                ))

            model.eval()
            train_evaluation_predictions = []
            train_evaluation_targets = []
            if train_evaluation_loader is not None:
                with torch.no_grad():
                    for batch in train_evaluation_loader:
                        batch = {
                            key: value.to(device, non_blocking=True)
                            for key, value in batch.items()
                        }
                        with torch.amp.autocast(
                            device_type=device.type, enabled=amp_enabled,
                        ):
                            if is_precomputed:
                                prediction, _, _ = model.forward_encoded(
                                    batch["poi_embedding"],
                                    batch["remote"],
                                    batch["environment"],
                                    batch["neighborhood_indices"],
                                    batch["neighborhood_mask"],
                                )
                            else:
                                prediction, _, _ = model(
                                    batch["poi"],
                                    batch["remote"],
                                    batch["environment"],
                                    batch["neighborhood_indices"],
                                    batch["neighborhood_mask"],
                                )
                        train_evaluation_predictions.append(prediction.float().cpu())
                        train_evaluation_targets.append(batch["target"].float().cpu())
            validation_losses = []
            validation_predictions = []
            validation_targets = []
            with torch.no_grad():
                for batch in loaders["validation"]:
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
                    absolute_error = torch.abs(prediction - batch["target"]).float().cpu()
                    validation_losses.append(absolute_error)
                    validation_predictions.append(prediction.float().cpu())
                    validation_targets.append(batch["target"].float().cpu())
            validation_errors = torch.cat(validation_losses)
            validation_environment_mae: Dict[int, float] = {}
            validation_loss = float(validation_errors.mean().item())
            entry = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean([value[0] for value in train_losses])),
                "train_regression_mae": float(np.mean([value[1] for value in train_losses])),
                "train_contrastive": float(np.mean([value[2] for value in train_losses])),
                "train_environment_mae": float(np.mean([value[3] for value in train_losses])),
                "train_environment_risk_variance": float(np.mean([
                    value[4] for value in train_losses
                ])),
                "train_invariance_penalty": float(np.mean([value[5] for value in train_losses])),
                "validation_mae": validation_loss,
                "validation_r2": _validation_r2(
                    torch.cat(validation_predictions), torch.cat(validation_targets),
                ),
                "validation_environment_mae": validation_environment_mae,
            }
            if train_evaluation_loader is not None:
                entry["train_r2"] = _validation_r2(
                    torch.cat(train_evaluation_predictions),
                    torch.cat(train_evaluation_targets),
                )
            log.append(entry)
            selection_active = not is_vrex or epoch >= int(
                settings.get("invariance_warmup_epochs", 2)
            )
            if selection_active:
                if validation_loss < best_loss - 1e-8:
                    best_loss, stale = validation_loss, 0
                    torch.save({
                        "model_state": model.state_dict(),
                        "epoch": epoch + 1,
                        "neighborhood_aggregation": neighborhood_aggregation,
                        **checkpoint_fields,
                    }, run_dir / "best.pt")
                else:
                    stale += 1
            else:
                best_loss, stale = float("inf"), 0
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1, "best_loss": best_loss, "stale": stale, "log": log,
                "neighborhood_aggregation": neighborhood_aggregation,
                **checkpoint_fields,
            }, latest_path)
            extra = {
                "train_regression_mae": entry["train_regression_mae"],
                "train_contrastive": entry["train_contrastive"],
            }
            if "train_r2" in entry:
                extra["train_r2"] = entry["train_r2"]
            if is_vrex:
                extra.update({
                    "train_environment_mae": entry["train_environment_mae"],
                    "train_environment_risk_variance": entry[
                        "train_environment_risk_variance"
                    ],
                    "train_invariance_penalty": entry["train_invariance_penalty"],
                })
            events.append(progress.update(entry, best_loss, stale, extra))
            write_training_log(run_dir / "training_log.json", events, log)
            if selection_active and stale >= int(settings["patience"]):
                RunProgress().early_stop(fold_id, model_name, seed, epoch + 1, stale)
                break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)["model_state"])
    model.eval()
    if is_vrex:
        validation_errors = []
        with torch.no_grad():
            for batch in loaders["validation"]:
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
                validation_errors.append(torch.abs(prediction - batch["target"]).float().cpu())
        errors = torch.cat(validation_errors).numpy()
        validation_cities = sorted(frames["validation"]["city_id"].astype(str).unique())
        validation_mae = float(errors.mean())
        audit = pd.DataFrame([{
            "environment_key": environment_key,
            "environment": ",".join(validation_cities),
            "environment_id": -1,
            "environment_role": "unseen_validation",
            "train_samples": 0,
            "validation_samples": int(len(errors)),
            "validation_mae": validation_mae,
            "environment_macro_mae": validation_mae,
            "worst_environment_mae": validation_mae,
            "environment_mae_std": 0.0,
        }])
        audit.to_csv(run_dir / "environment_metrics.csv", index=False)
    predictions = []
    with torch.no_grad():
        for batch in loaders["test"]:
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
            predictions.append(prediction.float().cpu().numpy())
    return np.concatenate(predictions)


def train_run(config: Dict, fold_id: str, model_name: str, seed: int = 42, force: bool = False) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name}; expected one of {MODEL_NAMES}")
    set_seed(seed)
    full_frame = None
    if model_name in STAGE1_MODEL_NAMES:
        frames, full_frame, manifest_path, experiment = _load_stage1_fold(config, fold_id)
    else:
        frames, manifest_path, experiment = _load_fold(config, fold_id)
    settings = config["training"]

    uses_precomputed_poi = (
        model_name.startswith("opencarbon_")
        and _open_carbon_poi_input_mode(model_name, settings) == "precomputed"
    )
    if (
        model_name in {"opencarbon_monthly_precomputed", "opencarbon_monthly_vrex"}
        and not experiment.startswith("cross_city")
    ):
        raise ValueError(f"{model_name} is only supported for cross-city protocols")
    run_dir = project_path(config["artifact_dir"]) / experiment / fold_id / model_name / f"seed_{seed}"
    complete_marker = run_dir / "COMPLETE"
    if complete_marker.exists() and not force:
        RunProgress().task_skip(fold_id, model_name, seed)
        return run_dir
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.get("device", "auto"))
    neighbors = pd.read_parquet(project_path(config["neighbors_file"]))
    neighbors["period"] = neighbors["period"].astype(str)
    metadata = {
        "experiment": experiment, "fold_id": fold_id, "model": model_name, "seed": int(seed),
        "device": str(device), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "panel_sha256": sha256_file(project_path(config["panel_file"])),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": deepcopy(config),
    }
    if model_name.startswith("opencarbon_"):
        metadata["input_ablation"] = str(settings.get("input_ablation", "baseline"))
    if model_name in STAGE1_MODEL_NAMES:
        metadata["stage1_protocol"] = {
            "input_scope": "viirs" if model_name == STAGE1_B0_MODEL else "viirs_modis_weather",
            "poi_input_mode": "none",
            "time_encoding": "learned_period_embedding_36",
            "context_scope": "full_city_same_period_open_features_no_labels",
            "target_transform": "log1p_emission_tc",
            "neighborhood_aggregation": str(settings.get("neighborhood_aggregation")),
        }
    if uses_precomputed_poi:
        missing = [
            key for key in ("poi_embedding_dir", "poi_embedding_checkpoint_template")
            if key not in config
        ]
        if missing:
            raise ValueError(
                "Precomputed POI input requires config keys: " + ", ".join(missing)
            )
        checkpoint_template = str(config["poi_embedding_checkpoint_template"])
        embedding_checkpoint = project_path(checkpoint_template.format(fold=fold_id, seed=seed))
        embedding_cache = project_path(config["poi_embedding_dir"]) / fold_id
        expected_embedding = expected_poi_embedding_metadata(
            fold_id,
            embedding_checkpoint,
            project_path(config["panel_file"]),
            manifest_path,
            project_path(config["poi_dir"]),
            int(settings["representation_dim"]),
        )
        embedding_metadata = validate_embedding_cache(embedding_cache, expected_embedding)
        metadata["poi_embedding"] = {
            "cache_dir": str(embedding_cache),
            "metadata_sha256": sha256_file(embedding_cache / "metadata.json"),
            "checkpoint_sha256": embedding_metadata["checkpoint_sha256"],
            "samples": embedding_metadata["samples"],
            "representation_dim": embedding_metadata["representation_dim"],
            "dtype": embedding_metadata["dtype"],
            "poi_encoder_frozen": True,
        }
    if model_name in {"opencarbon_monthly_precomputed", "opencarbon_monthly_vrex"}:
        metadata["causal_robustness"] = {
            "scope": "causal-inspired cross-city prediction",
            "poi_input_mode": "precomputed",
            "poi_encoder_frozen": True,
            "environment_key": str(settings.get("environment_key", "city_id")),
            "objective": (
                "mean_environment_mae + invariance_weight * variance_environment_mae"
                if model_name == "opencarbon_monthly_vrex" else "sample_mae"
            ),
            "invariance_weight": float(settings.get("invariance_weight", 1.0)),
            "invariance_warmup_epochs": int(settings.get("invariance_warmup_epochs", 2)),
            "claim_boundary": (
                "POI, remote sensing, weather, and month are proxy observations; "
                "this run does not identify policy treatment effects"
            ),
        }
    write_json(run_dir / "run_metadata.json", metadata)

    if model_name in STAGE1_MODEL_NAMES:
        assert full_frame is not None
        predictions = _train_stage1_open_carbon(
            model_name, fold_id, frames, full_frame, neighbors,
            config, settings, device, run_dir, seed,
        )
    elif model_name == "lightgbm":
        predictions = _train_lightgbm(frames, run_dir, fold_id, model_name, seed)
    elif model_name == "bpnn":
        predictions = _train_bpnn(frames, settings, device, run_dir, fold_id, seed)
    elif model_name == "carbongcn":
        predictions = _train_gcn(frames, neighbors, settings, device, run_dir, fold_id, seed)
    else:
        predictions = _train_open_carbon(
            model_name, fold_id, frames, neighbors, config, settings, device, run_dir, seed,
        )

    context = {"experiment": experiment, "fold_id": fold_id, "model": model_name, "seed": int(seed)}
    output = prediction_frame(frames["test"], predictions, context)
    output["split"] = "test"
    output["target_scale"] = "log1p_emission_tc"
    if model_name in STAGE1_MODEL_NAMES:
        output["neighborhood_aggregation"] = str(settings.get("neighborhood_aggregation"))
    output.to_parquet(run_dir / "predictions.parquet", index=False)
    monthly, summary = calculate_metrics(output)
    monthly.to_csv(run_dir / "metrics_monthly.csv", index=False)
    if model_name in STAGE1_MODEL_NAMES:
        calculate_pooled_metrics(output).to_csv(run_dir / "metrics_pooled.csv", index=False)
    if experiment.startswith("single_month_cross_region"):
        calculate_admin_metrics(output).to_csv(run_dir / "metrics_by_admin.csv", index=False)
    if experiment.startswith("annual_cross_region"):
        admin_monthly, admin_annual = calculate_annual_admin_metrics(output)
        admin_monthly.to_csv(run_dir / "metrics_by_admin_monthly.csv", index=False)
        admin_annual.to_csv(run_dir / "metrics_by_admin_annual.csv", index=False)
    if experiment.startswith("three_year_cross_region"):
        calculate_yearly_metrics(monthly).to_csv(run_dir / "metrics_yearly.csv", index=False)
        admin_monthly, admin_three_year = calculate_annual_admin_metrics(output)
        admin_monthly.to_csv(run_dir / "metrics_by_admin_monthly.csv", index=False)
        admin_three_year.to_csv(run_dir / "metrics_by_admin_three_year.csv", index=False)
    write_json(run_dir / "metrics_summary.json", summary)
    complete_marker.write_text("complete\n", encoding="utf-8")
    return run_dir
