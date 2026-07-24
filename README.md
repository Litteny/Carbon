# Carbon

The Python package uses a `src/` layout. Models are split by implementation under
`src/carbon_transfer/models/`.

Run one or more experiments from the project root. Training hyperparameters come from the YAML
configuration; command-line options select the fold, models, seed, and device.

```powershell
python main_experiment.py --city chicago --model opencarbon_monthly
python main_experiment.py --fold target_tokyo --models bpnn,carbongcn,opencarbon_monthly --seed 42
python main_experiment.py --config configs/smoke.yaml --city singapore --dry-run
```

Available cross-city targets are discovered from `data/splits/cross_city`. Exact cross-city and
cross-region fold names can be passed with `--fold`. Use `python main_experiment.py --help` for
the complete interface.
