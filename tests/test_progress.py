from io import StringIO

from carbon_transfer.progress import EpochProgress, OpenCarbonTaskProgress, is_open_carbon


def test_only_open_carbon_variants_are_identified():
    assert is_open_carbon("opencarbon_core")
    assert is_open_carbon("opencarbon_monthly")
    assert is_open_carbon("opencarbon_monthly_noviirs")
    assert is_open_carbon("opencarbon_monthly_vrex")
    assert is_open_carbon("opencarbon_monthly_precomputed")
    assert not is_open_carbon("lightgbm")
    assert not is_open_carbon("bpnn")
    assert not is_open_carbon("carbongcn")


def test_epoch_progress_writes_one_stable_line_for_any_stream():
    output = StringIO()
    progress = EpochProgress("fold", "opencarbon_monthly", 42, 2, 5, output)
    event = progress.update(
        {
            "epoch": 3,
            "train_loss": 0.25,
            "validation_mae": 0.5,
            "validation_r2": 0.625,
        },
        0.4,
        1,
    )
    text = output.getvalue()
    assert event["event"] == "epoch"
    assert text.startswith("[epoch 3/5]")
    assert " epoch=" not in text
    assert "train_loss=0.2500" in text
    assert "validation_mae=0.5000" in text
    assert "validation_r2=0.6250" in text
    assert event["validation_r2"] == 0.625
    assert "best_validation_mae=0.4000" in text
    assert "\r" not in text


def test_epoch_progress_reports_open_carbon_extra_losses():
    output = StringIO()
    progress = EpochProgress("fold", "opencarbon_monthly", 42, 0, 2, output)
    progress.update(
        {
            "epoch": 1,
            "train_loss": 0.2,
            "validation_mae": 0.3,
            "validation_r2": -0.1,
        },
        0.3,
        0,
        {
            "train_regression_mae": 0.19,
            "train_contrastive": 0.11,
            "train_r2": 0.72,
        },
    )
    text = output.getvalue()
    assert "train_regression_mae=0.1900" in text
    assert "train_contrastive=0.1100" in text
    assert "train_r2=0.7200" in text


def test_epoch_progress_reports_vrex_losses_on_one_line():
    output = StringIO()
    progress = EpochProgress("fold", "opencarbon_monthly_vrex", 42, 0, 2, output)
    progress.update(
        {
            "epoch": 1,
            "train_loss": 0.4,
            "validation_mae": 0.3,
            "validation_r2": 0.2,
        },
        0.3,
        0,
        {
            "train_environment_mae": 0.2,
            "train_environment_risk_variance": 0.03,
            "train_invariance_penalty": 0.03,
        },
    )
    text = output.getvalue()
    assert text.count("\n") == 1
    assert "train_environment_mae=0.2000" in text
    assert "train_environment_risk_variance=0.0300" in text
    assert "train_invariance_penalty=0.0300" in text


def test_task_progress_counts_only_open_carbon():
    output = StringIO()
    tasks = [("a", "bpnn", 42), ("b", "opencarbon_core", 42)]
    progress = OpenCarbonTaskProgress(tasks, output)
    assert progress.total == 1
    progress.finish("a", "bpnn", "completed")
    assert output.getvalue() == ""
    progress.finish("b", "opencarbon_core", "skipped")
    assert "completed=1/1" in output.getvalue()
    assert "status=skipped" in output.getvalue()
