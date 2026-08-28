from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from carbon_transfer.datasets import BalancedEnvironmentBatchSampler, OpenCarbonDataset
from carbon_transfer.training import (
    _environment_macro_validation_mae,
    _environment_mapping,
    _validate_vrex_checkpoint,
    _vrex_objective,
)


def test_balanced_environment_sampler_covers_every_environment_and_is_reproducible():
    environment_ids = [0, 0, 0, 0, 1, 1, 2]
    first = BalancedEnvironmentBatchSampler(environment_ids, batch_size=6, seed=42)
    second = BalancedEnvironmentBatchSampler(environment_ids, batch_size=6, seed=42)
    first_batches = list(first)
    second_batches = list(second)
    assert first_batches == second_batches
    assert len(first_batches) == 2
    for batch in first_batches:
        assert len(batch) == 6
        assert {environment_ids[index] for index in batch} == {0, 1, 2}


def test_balanced_environment_sampler_validates_environment_count_and_batch_size():
    with pytest.raises(ValueError, match="at least two"):
        BalancedEnvironmentBatchSampler([0, 0], batch_size=2, seed=42)
    with pytest.raises(ValueError, match="batch_size"):
        BalancedEnvironmentBatchSampler([0, 1, 2], batch_size=2, seed=42)


def test_vrex_objective_uses_environment_macro_mae_and_population_variance():
    prediction = torch.tensor([0.0, 2.0, 4.0, 8.0])
    target = torch.tensor([0.0, 0.0, 3.0, 3.0])
    environment = torch.tensor([0, 0, 1, 1])
    loss, macro, variance, penalty = _vrex_objective(
        prediction, target, environment, invariance_weight=2.0, penalty_active=True,
    )
    # Environment MAEs are 1 and 3, so macro=2 and population variance=1.
    assert torch.isclose(macro, torch.tensor(2.0))
    assert torch.isclose(variance, torch.tensor(1.0))
    assert torch.isclose(penalty, torch.tensor(2.0))
    assert torch.isclose(loss, torch.tensor(4.0))

    warmup_loss, _, _, warmup_penalty = _vrex_objective(
        prediction, target, environment, invariance_weight=2.0, penalty_active=False,
    )
    assert torch.isclose(warmup_penalty, torch.tensor(0.0))
    assert torch.isclose(warmup_loss, torch.tensor(2.0))


def test_environment_macro_validation_mae_weights_cities_equally():
    errors = torch.tensor([1.0, 1.0, 1.0, 9.0])
    environments = torch.tensor([0, 0, 0, 1])
    macro, values = _environment_macro_validation_mae(errors, environments)
    assert values == {0: 1.0, 1: 9.0}
    assert macro == 5.0


def test_environment_mapping_is_sorted_and_requires_multiple_environments():
    frame = pd.DataFrame({"city_id": ["tokyo", "chicago", "tokyo", "nyc"]})
    assert _environment_mapping(frame, "city_id") == {
        "chicago": 0, "nyc": 1, "tokyo": 2,
    }
    with pytest.raises(ValueError, match="at least two"):
        _environment_mapping(pd.DataFrame({"city_id": ["tokyo"]}), "city_id")


def test_open_carbon_dataset_environment_ids_come_from_explicit_mapping():
    frame = pd.DataFrame({
        "city_id": ["chicago", "target"],
        "log1p_emission": [0.0, 1.0],
    })
    dataset = OpenCarbonDataset(
        frame,
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
        np.asarray([[0] * 9, [1] * 9], dtype=np.int64),
        Path("unused"),
        environment_mapping={"chicago": 0},
    )
    assert dataset.environment_ids.tolist() == [0, -1]
    assert dataset.unknown_environments == ["target"]


def test_vrex_checkpoint_must_match_environment_and_objective_configuration():
    settings = {
        "environment_key": "city_id",
        "invariance_weight": 1.0,
        "invariance_warmup_epochs": 2,
    }
    checkpoint = {
        "training_strategy": "vrex",
        "environment_mapping": {"a": 0, "b": 1},
        **settings,
    }
    _validate_vrex_checkpoint(checkpoint, {"a": 0, "b": 1}, settings)
    with pytest.raises(ValueError, match="invariance_weight"):
        _validate_vrex_checkpoint(
            {**checkpoint, "invariance_weight": 2.0}, {"a": 0, "b": 1}, settings,
        )
