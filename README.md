# Carbon

The project uses a `src/` layout. Reusable data, model, training, evaluation, and experiment logic lives in the installable `carbon_transfer` package. Experiments are launched with explicit Python scripts under `scripts/`.

Install in editable mode first:

```bash
python -m pip install -e .
```

Data and manifest commands:

```bash
carbon models list
carbon data build --config configs/data.yaml
carbon splits build --config configs/experiments.yaml
carbon validate experiment --config configs/experiments/stage1_cross_city.yaml
```

Build and preview the fixed-role OpenCarbon protocol (Chicago training,
NYC/Singapore validation, Tokyo testing):

```bash
carbon splits build --config configs/splits/cross_city_chicago_to_tokyo.yaml
python scripts/run_cross_city_chicago_to_tokyo.py --dry-run
python scripts/run_cross_city_chicago_to_tokyo.py \
  --models opencarbon_monthly --device cuda
```

Precomputed and V-REx runs require the dense monthly checkpoint first. Build
their protocol-specific POI cache before launching those configurations:

```bash
python scripts/build_poi_embeddings.py \
  --config configs/experiments/cross_city_chicago_to_tokyo_precomputed.yaml \
  --dry-run
python scripts/build_poi_embeddings.py \
  --config configs/experiments/cross_city_chicago_to_tokyo_precomputed.yaml
python scripts/run_cross_city_chicago_to_tokyo.py \
  --config configs/experiments/cross_city_chicago_to_tokyo_vrex.yaml \
  --dry-run
```

The V-REx configuration also contains its strict frozen-embedding ERM control.
Run both under the same hyperparameters with:

```bash
python scripts/run_cross_city_chicago_to_tokyo.py \
  --config configs/experiments/cross_city_chicago_to_tokyo_vrex.yaml \
  --models opencarbon_monthly_precomputed opencarbon_monthly_vrex \
  --device cuda \
  --dry-run
python scripts/run_cross_city_chicago_to_tokyo.py \
  --config configs/experiments/cross_city_chicago_to_tokyo_vrex.yaml \
  --models opencarbon_monthly_precomputed opencarbon_monthly_vrex \
  --device cuda
```

Preview or run a Stage 1 cross-city matrix:

```bash
python scripts/run_stage1_cross_city.py \
  --cities singapore new_york tokyo chicago \
  --models lightgbm bpnn carbongcn opencarbon_core opencarbon_monthly \
  --seeds 42 123 2026 \
  --epochs 80 \
  --patience 12 \
  --batch-size 256 \
  --device cuda \
  --dry-run

python scripts/run_stage1_cross_city.py \
  --cities singapore new_york tokyo chicago \
  --models opencarbon_monthly \
  --seeds 42 \
  --device cuda
```

Every experiment script supports `--cities`, `--models`, `--seeds`, `--epochs`, `--patience`, `--batch-size`, `--device`, `--dry-run`, `--evaluate-only`, and `--force`. The alias `new_york` maps to the internal city ID `nyc`.

Generated runs and reports live under `outputs/<experiment>/`. Completed tasks contain a `COMPLETE` marker and are skipped unless `--force` is explicitly provided.

Feature-level causal-candidate diagnostics use a separate observational panel
protocol and never reuse prediction metrics as causal evidence. Preview all
30 estimable features and five temporal specifications before running:

```bash
python scripts/run_feature_causal_analysis.py --dry-run
python scripts/run_feature_causal_analysis.py \
  --features temperature_2m_mean_c modis_ndvi_mean \
  --lags 0 1 3 6 -1
```

The estimator absorbs grid and city-by-period fixed effects, clusters standard
errors by grid, reports per-city effects, and treats lag `-1` as a future-feature
negative control. Results under `outputs/feature_causal_analysis/` are graded
observational causal candidates (C/C+) or predictive evidence (D), not proof of
an intervention effect. Unknown or circular label provenance forces VIIRS to D;
verified VIIRS still remains proxy-limited under this observational protocol.

For a low-cost cross-city screen that does not train OpenCarbon, reuse the
contemporaneous per-city estimates and run:

```bash
python scripts/run_feature_transfer_screening.py --dry-run
python scripts/run_feature_transfer_screening.py
```

This protocol combines city-effect meta-analysis and I², leave-one-city-out
distribution support and Wasserstein distance, grid-grouped city-domain
classification, and month-block-bootstrap partial R². Its output under
`outputs/feature_transfer_screening/reports/` ranks statistically promising
features for later target-city ablation; it does not claim actual target-city
prediction gains.
