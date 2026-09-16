from __future__ import annotations
import argparse
from copy import deepcopy
from typing import Optional, Sequence
from carbon_transfer.config import apply_run_namespace, load_config
from carbon_transfer.experiments import aggregate_for_config, discover_tasks, run_experiment
from carbon_transfer.models.registry import list_model_ids

DEFAULT_CONFIG = "configs/experiments/annual_cross_region_2022_input_ablation.yaml"
EXPERIMENT = "annual_cross_region_2022"
ABLATIONS = ("baseline", "no_weather", "no_month", "no_modis", "no_viirs", "no_poi")
CITY_ALIASES = {"chicago":"chicago", "nyc":"nyc", "new_york":"nyc", "singapore":"singapore", "tokyo":"tokyo"}
def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0: raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed
def _task_city(task): return task.fold.split("_split_seed_", 1)[0].split("_", 1)[1]
def _selected_cities(values: Optional[Sequence[str]]): return None if not values else {CITY_ALIASES[value] for value in values}
def build_parser():
    parser = argparse.ArgumentParser(description="Run annual cross-region input ablations")
    parser.add_argument("--config", default=DEFAULT_CONFIG); parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS)); parser.add_argument("--cities", nargs="+", choices=tuple(CITY_ALIASES)); parser.add_argument("--models", nargs="+", choices=tuple(list_model_ids())); parser.add_argument("--epochs", type=_positive_int); parser.add_argument("--patience", type=_positive_int); parser.add_argument("--batch-size", type=_positive_int); parser.add_argument("--device"); parser.add_argument("--run-name"); mode=parser.add_mutually_exclusive_group(); mode.add_argument("--dry-run", action="store_true"); mode.add_argument("--evaluate-only", action="store_true"); parser.add_argument("--force", action="store_true")
    return parser
def main(argv: Optional[Sequence[str]] = None) -> int:
    args=build_parser().parse_args(argv); base=load_config(args.config); apply_run_namespace(base,args.run_name); cities=_selected_cities(args.cities)
    for ablation in args.ablations:
        config=deepcopy(base); config["split_experiment"]=EXPERIMENT; config["artifact_dir"]=f"{base['artifact_dir']}/{ablation}"; config["report_dir"]=f"{base['report_dir']}/{ablation}"; config["training"]=dict(config.get("training",{})); config["training"]["input_ablation"]=ablation
        if args.models: config["models"]=list(args.models)
        if args.device: config["device"]=args.device
        for value,key in ((args.epochs,"max_epochs"),(args.patience,"patience"),(args.batch_size,"batch_size")):
            if value is not None: config["training"][key]=value
        tasks=discover_tasks(config,models=args.models,experiments=[EXPERIMENT]); tasks=[task for task in tasks if cities is None or _task_city(task) in cities]
        if args.dry_run:
            for task in tasks: print(f"ablation={ablation} city={_task_city(task)} {task.identity}")
            print(f"ablation={ablation} tasks={len(tasks)}")
        elif args.evaluate_only: aggregate_for_config(config,tasks)
        else: run_experiment(config,tasks,force=args.force,aggregate_tasks=tasks)
    return 0
if __name__ == "__main__": raise SystemExit(main())
