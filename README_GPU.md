# Carbon transfer experiment runner

This repository implements the transfer experiments described in `PROJECT_TASK.md`. Reusable logic lives in `src/carbon_transfer/`; each experiment protocol has an explicit Python entry script.

## Linux GPU environment

Use Python 3.9 or newer and an NVIDIA driver compatible with the installed PyTorch build.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-linux.txt
python -m pip install -e .
```

Build data and split manifests from the project root:

```bash
carbon data build --config configs/data.yaml
carbon splits build --config configs/experiments.yaml
carbon splits build --config configs/experiments_single_month.yaml
carbon splits build --config configs/experiments_single_month_multiseed.yaml
carbon splits build --config configs/experiments/annual_cross_region_2022.yaml
```

## Experiment scripts

Stage 1 leave-one-city-out cross-city transfer:

```bash
python scripts/run_stage1_cross_city.py --dry-run
python scripts/run_stage1_cross_city.py --device cuda
```

Stage 1 within-city cross-region pilots:

```bash
python scripts/run_stage1_cross_region.py --dry-run
python scripts/run_stage1_cross_region.py --device cuda
```

Single-month and joint split/training-seed experiments:

```bash
python scripts/run_single_month_cross_region.py --dry-run
python scripts/run_single_month_cross_region.py --device cuda

python scripts/run_single_month_cross_region_multiseed.py --dry-run
python scripts/run_single_month_cross_region_multiseed.py --seeds 42 43 44 --device cuda
```

Annual within-city cross-region experiments:

```bash
python scripts/run_annual_cross_region.py --dry-run
python scripts/run_annual_cross_region.py \
  --cities chicago new_york singapore tokyo \
  --seeds 42 43 44 \
  --epochs 80 \
  --patience 12 \
  --batch-size 256 \
  --device cuda
```

Use `--evaluate-only` to rebuild the protocol report without training. Completed tasks have a `COMPLETE` marker and are skipped on rerun; use `--force` only when a completed result must be replaced.

BPNN, CarbonGCN, and OpenCarbon print exactly one permanent line after each completed epoch. LightGBM reports fit start/end and best iteration without fake epochs. Smoke outputs under `outputs/smoke/` remain diagnostic only.
