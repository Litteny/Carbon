import math

import torch

from carbon_transfer.training import _validation_r2


def test_validation_r2_is_one_for_perfect_predictions():
    targets = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert _validation_r2(targets.clone(), targets) == 1.0


def test_validation_r2_allows_negative_values():
    targets = torch.tensor([0.0, 1.0, 2.0])
    predictions = torch.tensor([3.0, 3.0, 3.0])
    assert _validation_r2(predictions, targets) < 0.0


def test_validation_r2_is_nan_when_undefined():
    assert math.isnan(_validation_r2(torch.tensor([1.0]), torch.tensor([1.0])))
    assert math.isnan(_validation_r2(
        torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]),
    ))
    assert math.isnan(_validation_r2(
        torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 2.0]),
    ))
