# Carbon transfer experiment runner

This repository implements the single-seed first stage described in `PROJECT_TASK.md`:
four leave-one-city-out folds and four within-city pilot folds, with six models and seed 42.

## Local preparation (Windows)

```powershell
D:\CondaEnvs\urban_carbon\python.exe scripts/build_dataset.py --config configs/data.yaml
D:\CondaEnvs\urban_carbon\python.exe scripts/build_splits.py --config configs/experiments.yaml
D:\CondaEnvs\urban_carbon\python.exe scripts/train.py --config configs/smoke.yaml --fold target_chicago --model lightgbm --seed 42
```

Run the smoke command once for every model name in `configs/smoke.yaml`. Smoke artifacts are
diagnostic only and must not be included in formal results.

## Linux GPU environment

Use Python 3.9 and an NVIDIA driver compatible with CUDA 11.8 or newer. Install the matching
PyTorch wheel first if the index used by your server does not provide CUDA builds, then install
the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-linux.txt
python -m pip install -e .
```

Upload the complete project directory to the remote server. Formal training reads the prepared
`data/features/` and `data/splits/` directories; all configuration paths are relative to the
project root. After entering the uploaded project directory, confirm the matrix and then start or
resume all 48 tasks with:

```bash
python scripts/run_stage1.py --config configs/stage1_gpu.yaml --dry-run
python scripts/run_stage1.py --config configs/stage1_gpu.yaml
```

Completed tasks have an `artifacts/.../COMPLETE` marker and are skipped on rerun. OpenCarbon tasks
also save `latest.pt` after every epoch and resume it automatically. To intentionally replace a
completed result, pass `--force`.

Formal outputs are written below `artifacts/`; the consolidated single-seed report is written to
`reports/stage1/`. The exhaustive 213-region experiment and additional seeds are deliberately not
started by this command.

## Single-month cross-region pilot (2022-08)

Build the deterministic administrative-region splits and preview the four-task matrix:

```bash
python scripts/build_splits.py --config configs/experiments_single_month.yaml
python scripts/run_single_month.py --config configs/single_month_gpu.yaml --dry-run
```

Run the pilot with `opencarbon_monthly` and seed 42:

```bash
python scripts/run_single_month.py --config configs/single_month_gpu.yaml
```

OpenCarbon runs display overall task progress and per-task epoch progress in an interactive
terminal. Redirected logs use one stable line per epoch and include training loss, validation MAE,
best validation MAE, and the early-stopping counter.

The split manifests are written under `data/splits/single_month_cross_region/`. Training artifacts
and the four-city macro/grid-weighted report are isolated under `artifacts_single_month/` and
`reports/single_month/`. Completed runs are skipped unless `--force` is supplied.

For the joint split/training seed experiment (`42`, `43`, `44`), use:

```bash
python scripts/build_splits.py --config configs/experiments_single_month_multiseed.yaml
python scripts/run_single_month_multiseed.py --config configs/single_month_multiseed_gpu.yaml --dry-run
python scripts/run_single_month_multiseed.py --config configs/single_month_multiseed_gpu.yaml
```

This creates 12 tasks (four cities by three seeds). Each seed uses a different administrative-region
split and the same value for model initialization. Outputs are isolated under
`artifacts_single_month_multiseed/` and `reports/single_month_multiseed/`.
