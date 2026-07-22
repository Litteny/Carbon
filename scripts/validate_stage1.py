#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_transfer.constants import MODEL_NAMES
from carbon_transfer.utils import write_json


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    checks = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    panel = pd.read_parquet(root / "data/features/grid_month_panel.parquet")
    check("panel_row_count", len(panel) == 142992, len(panel))
    check(
        "panel_key_unique",
        not panel.duplicated(["city_id", "cell_id", "period"]).any(),
        int(panel.duplicated(["city_id", "cell_id", "period"]).sum()),
    )
    city_periods = panel.groupby("city_id")["period"].nunique().to_dict()
    check("four_cities_36_periods", city_periods == {"chicago": 36, "nyc": 36, "singapore": 36, "tokyo": 36}, city_periods)

    poi_files = list((root / "data/features/poi_sparse").rglob("*.npz"))
    check("poi_sparse_files", len(poi_files) == 144, len(poi_files))
    poi_audit = json.loads((root / "data/features/poi_audit.json").read_text(encoding="utf-8"))
    check("poi_verification_no_failure", not poi_audit.get("verification_failures"), poi_audit.get("verification_status"))

    manifests = sorted((root / "data/splits").rglob("*.parquet"))
    check("stage1_split_count", len(manifests) == 8, len(manifests))
    split_errors = []
    for path in manifests:
        frame = pd.read_parquet(path)
        if frame.duplicated(["city_id", "cell_id", "period"]).any():
            split_errors.append(f"{path.name}:duplicate_keys")
        if frame.groupby(["city_id", "cell_id"])["split"].nunique().max() != 1:
            split_errors.append(f"{path.name}:grid_crosses_splits")
    check("split_leakage", not split_errors, split_errors)

    smoke_root = root / "artifacts_smoke/cross_city/target_chicago"
    completed_models = sorted(
        path.parent.parent.name for path in smoke_root.rglob("seed_42/COMPLETE")
    )
    check("six_model_smoke", completed_models == sorted(MODEL_NAMES), completed_models)
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    report = {"status": status, "checks": checks}
    report_dir = root / "reports"
    write_json(report_dir / "stage1_setup_audit.json", report)
    lines = ["# Stage 1 setup audit", "", f"Overall status: **{status}**", "", "| Check | Status | Detail |", "| --- | --- | --- |"]
    for item in checks:
        detail = json.dumps(item["detail"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {item['name']} | {item['status']} | {detail} |")
    (report_dir / "stage1_setup_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
