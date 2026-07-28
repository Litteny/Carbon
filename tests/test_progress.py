from io import StringIO

from carbon_transfer.progress import (
    OpenCarbonEpochProgress,
    OpenCarbonTaskProgress,
    is_open_carbon,
)


class TtyBuffer(StringIO):
    def isatty(self):
        return True


def test_only_open_carbon_variants_enable_progress():
    assert is_open_carbon("opencarbon_core")
    assert is_open_carbon("opencarbon_monthly")
    assert is_open_carbon("opencarbon_monthly_noviirs")
    assert not is_open_carbon("lightgbm")
    assert not is_open_carbon("bpnn")
    assert not is_open_carbon("carbongcn")


def test_non_tty_epoch_progress_writes_one_stable_line():
    output = StringIO()
    progress = OpenCarbonEpochProgress("fold", "opencarbon_monthly", 2, 5, output)
    assert progress.bar.n == 2
    assert progress.bar.total == 5
    progress.update({"epoch": 3, "train_loss": 0.25, "validation_mae": 0.5}, 0.4, 1)
    progress.close()
    text = output.getvalue()
    assert "epoch=3/5" in text
    assert "train_loss=0.2500" in text
    assert "validation_mae=0.5000" in text
    assert "\r" not in text


def test_non_tty_task_progress_counts_only_open_carbon():
    output = StringIO()
    tasks = [("a", "bpnn", 42), ("b", "opencarbon_core", 42)]
    progress = OpenCarbonTaskProgress(tasks, output)
    assert progress.total == 1
    progress.finish("a", "bpnn", "completed")
    assert output.getvalue() == ""
    progress.finish("b", "opencarbon_core", "skipped")
    progress.close()
    assert "1/1" in output.getvalue()
    assert "status=skipped" in output.getvalue()


def test_tty_epoch_progress_uses_dynamic_tqdm_output():
    output = TtyBuffer()
    progress = OpenCarbonEpochProgress("fold", "opencarbon_core", 0, 2, output)
    assert progress.bar.disable is False
    progress.update({"epoch": 1, "train_loss": 0.2, "validation_mae": 0.3}, 0.3, 0)
    progress.close()
    assert "\r" in output.getvalue()
    assert "opencarbon_core fold" in output.getvalue()


def test_epoch_bar_uses_second_line_inside_task_progress():
    output = TtyBuffer()
    with OpenCarbonTaskProgress([("fold", "opencarbon_core", 42)], output):
        epoch = OpenCarbonEpochProgress("fold", "opencarbon_core", 0, 2, output)
        assert epoch.bar.pos == -1
        epoch.close()
