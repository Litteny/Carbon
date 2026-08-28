import pytest

from carbon_transfer.config import apply_run_namespace


def test_run_namespace_isolates_artifacts_and_reports():
    config = {
        "artifact_dir": "outputs/example/runs",
        "report_dir": "outputs/example/reports",
    }
    result = apply_run_namespace(config, "lr3e-4_seed42")
    assert result["artifact_dir"] == "outputs/example/runs/lr3e-4_seed42"
    assert result["report_dir"] == "outputs/example/reports/lr3e-4_seed42"
    assert result["run_name"] == "lr3e-4_seed42"


def test_run_namespace_rejects_paths_and_empty_prefixes():
    config = {"artifact_dir": "outputs/runs"}
    with pytest.raises(ValueError, match="run_name"):
        apply_run_namespace(config, "../overwrite")
    with pytest.raises(ValueError, match="run_name"):
        apply_run_namespace(config, "-invalid")
