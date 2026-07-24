from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

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
    TABULAR_FEATURES, VIIRS_COLUMNS, WEATHER_COLUMNS,
)
from .datasets import OpenCarbonDataset, fixed_neighborhood_indices, normalized_adjacency
from .metrics import calculate_metrics, prediction_frame
from .models import BPNN, CarbonGCN, OpenCarbonModel, contrastive_loss
from .utils import resolve_device, set_seed, sha256_file, write_json


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
        frames = {name: frame.head(limit).copy() for name, frame in frames.items()}
    return frames, manifest_path, experiment


def _pipeline(scale: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _torch_train_tabular(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    settings: Dict,
    device: torch.device,
    run_dir: Path,
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
    latest_path = run_dir / "latest.pt"
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        stale = int(checkpoint.get("stale", 0))
        log = checkpoint.get("log", [])
    for epoch in range(start_epoch, int(settings["max_epochs"])):
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
            validation_loss = float(torch.mean(torch.abs(model(validation_tensor) - validation_target)).cpu())
        log.append({"epoch": epoch + 1, "train_mae": float(np.mean(train_losses)), "validation_mae": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
            torch.save({"model_state": model.state_dict(), "epoch": best_epoch}, run_dir / "best.pt")
        else:
            stale += 1
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1, "best_loss": best_loss, "stale": stale, "log": log,
        }, latest_path)
        if stale >= int(settings["patience"]):
            break
    write_json(run_dir / "training_log.json", log)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return model


def _train_lightgbm(frames: Dict[str, pd.DataFrame], run_dir: Path) -> np.ndarray:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    features = TABULAR_FEATURES
    preprocessor = _pipeline(scale=False)
    train_x = preprocessor.fit_transform(frames["train"][features])
    validation_x = preprocessor.transform(frames["validation"][features])
    test_x = preprocessor.transform(frames["test"][features])
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
    write_json(run_dir / "training_log.json", {"best_iteration": int(model.best_iteration_)})
    return model.predict(test_x)


def _train_bpnn(
    frames: Dict[str, pd.DataFrame], settings: Dict, device: torch.device, run_dir: Path,
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
        settings, device, run_dir,
    )
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(test_x).to(device)).cpu().numpy()


def _train_gcn(
    frames: Dict[str, pd.DataFrame], neighbors: pd.DataFrame, settings: Dict,
    device: torch.device, run_dir: Path,
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
    latest_path = run_dir / "latest.pt"
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        stale = int(checkpoint.get("stale", 0))
        log = checkpoint.get("log", [])
    for epoch in range(start_epoch, int(settings["max_epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(tensors["train"], adjacency["train"])
        loss = torch.mean(torch.abs(prediction - targets["train"]))
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean(torch.abs(
                model(tensors["validation"], adjacency["validation"]) - targets["validation"]
            )).cpu())
        log.append({"epoch": epoch + 1, "train_mae": float(loss.detach().cpu()), "validation_mae": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss, stale = validation_loss, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1}, run_dir / "best.pt")
        else:
            stale += 1
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1, "best_loss": best_loss, "stale": stale, "log": log,
        }, latest_path)
        if stale >= int(settings["patience"]):
            break
    write_json(run_dir / "training_log.json", log)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)["model_state"])
    model.eval()
    with torch.no_grad():
        return model(tensors["test"], adjacency["test"]).cpu().numpy()


def _open_carbon_feature_sets(model_name: str) -> Tuple[List[str], List[str]]:
    no_viirs = model_name == "opencarbon_monthly_noviirs"
    monthly = model_name != "opencarbon_core"
    remote = MODIS_COLUMNS + [f"{name}_is_missing" for name in MODIS_COLUMNS]
    if not no_viirs:
        remote += VIIRS_COLUMNS
    environment = WEATHER_COLUMNS + [f"{name}_is_missing" for name in WEATHER_COLUMNS]
    if monthly:
        environment += MONTH_COLUMNS
    return remote, environment


def _train_open_carbon(
    model_name: str,
    frames: Dict[str, pd.DataFrame],
    neighbors: pd.DataFrame,
    config: Dict,
    settings: Dict,
    device: torch.device,
    run_dir: Path,
) -> np.ndarray:
    remote_columns, environment_columns = _open_carbon_feature_sets(model_name)
    feature_columns = remote_columns + environment_columns
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
    }, run_dir / "preprocessor.joblib")
    remote_size = len(remote_columns)
    datasets = {}
    for name, values in transformed.items():
        neighborhoods = fixed_neighborhood_indices(frames[name], neighbors)
        datasets[name] = OpenCarbonDataset(
            frames[name], values[:, :remote_size], values[:, remote_size:], neighborhoods,
            project_path(config["poi_dir"]),
        )
    loaders = {
        name: DataLoader(
            dataset, batch_size=int(settings["batch_size"]), shuffle=name == "train",
            num_workers=int(settings.get("num_workers", 0)), pin_memory=device.type == "cuda",
            collate_fn=dataset.collate,
        )
        for name, dataset in datasets.items()
    }
    model = OpenCarbonModel(
        remote_size, len(environment_columns),
        int(settings["representation_dim"]), float(settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]),
    )
    amp_enabled = bool(settings.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation = int(settings.get("gradient_accumulation", 1))
    best_loss, stale, start_epoch = float("inf"), 0, 0
    latest_path = run_dir / "latest.pt"
    log = []
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        stale = int(checkpoint.get("stale", 0))
        log = checkpoint.get("log", [])

    for epoch in range(start_epoch, int(settings["max_epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        for step, batch in enumerate(loaders["train"]):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                prediction, poi_representation, remote_representation = model(
                    batch["poi"], batch["remote"], batch["environment"],
                    batch["neighborhood_indices"], batch["neighborhood_mask"],
                )
                regression = torch.mean(torch.abs(prediction - batch["target"]))
                if epoch >= int(settings["contrastive_warmup_epochs"]) and len(prediction) > 1:
                    contrast = contrastive_loss(
                        poi_representation, remote_representation, float(settings.get("temperature", 0.07)),
                    )
                    loss = regression + float(settings["contrastive_weight"]) * contrast
                else:
                    contrast = torch.zeros((), device=device)
                    loss = regression
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loaders["train"]):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append((float(loss.detach().cpu()), float(regression.detach().cpu()), float(contrast.detach().cpu())))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in loaders["validation"]:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    prediction, _, _ = model(
                        batch["poi"], batch["remote"], batch["environment"],
                        batch["neighborhood_indices"], batch["neighborhood_mask"],
                    )
                validation_losses.append(float(torch.mean(torch.abs(prediction - batch["target"])).cpu()))
        validation_loss = float(np.mean(validation_losses))
        entry = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean([value[0] for value in train_losses])),
            "train_regression_mae": float(np.mean([value[1] for value in train_losses])),
            "train_contrastive": float(np.mean([value[2] for value in train_losses])),
            "validation_mae": validation_loss,
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
        write_json(run_dir / "training_log.json", log)
        if stale >= int(settings["patience"]):
            break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)["model_state"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loaders["test"]:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
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
    frames, manifest_path, experiment = _load_fold(config, fold_id)
    run_dir = project_path(config["artifact_dir"]) / experiment / fold_id / model_name / f"seed_{seed}"
    complete_marker = run_dir / "COMPLETE"
    if complete_marker.exists() and not force:
        return run_dir
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = config["training"]
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
    write_json(run_dir / "run_metadata.json", metadata)

    if model_name == "lightgbm":
        predictions = _train_lightgbm(frames, run_dir)
    elif model_name == "bpnn":
        predictions = _train_bpnn(frames, settings, device, run_dir)
    elif model_name == "carbongcn":
        predictions = _train_gcn(frames, neighbors, settings, device, run_dir)
    else:
        predictions = _train_open_carbon(
            model_name, frames, neighbors, config, settings, device, run_dir,
        )

    context = {"experiment": experiment, "fold_id": fold_id, "model": model_name, "seed": int(seed)}
    output = prediction_frame(frames["test"], predictions, context)
    output.to_parquet(run_dir / "predictions.parquet", index=False)
    monthly, summary = calculate_metrics(output)
    monthly.to_csv(run_dir / "metrics_monthly.csv", index=False)
    write_json(run_dir / "metrics_summary.json", summary)
    complete_marker.write_text("complete\n", encoding="utf-8")
    return run_dir
