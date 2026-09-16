from pathlib import Path

import numpy as np
import pandas as pd
import torch

from carbon_transfer.constants import TABULAR_FEATURES, VIIRS_COLUMNS
from carbon_transfer.datasets import (
    SparsePOIStore, fixed_neighborhood_indices, masked_edges, normalized_adjacency,
)
from carbon_transfer.metrics import calculate_metrics
from carbon_transfer.models import (
    BPNN,
    CarbonGCN,
    MeanMLPGatedNeighborhoodAggregator,
    NeighborhoodAggregator,
    OpenCarbonModel,
    POIEncoder,
)
from carbon_transfer.models.bpnn import BPNN as SplitBPNN
from carbon_transfer.models.carbongcn import CarbonGCN as SplitCarbonGCN
from carbon_transfer.models.opencarbon import OpenCarbonModel as SplitOpenCarbonModel
from carbon_transfer.training import _open_carbon_feature_sets, _open_carbon_features_for_scope, _validate_neighborhood_aggregation


ROOT = Path(__file__).resolve().parents[1]


def test_model_package_preserves_public_exports():
    assert BPNN is SplitBPNN
    assert CarbonGCN is SplitCarbonGCN
    assert OpenCarbonModel is SplitOpenCarbonModel


def test_forbidden_identifiers_are_not_features():
    forbidden = {"city_id", "admin_id", "lon", "lat", "emission_tc", "log1p_emission"}
    assert forbidden.isdisjoint(TABULAR_FEATURES)


def test_no_viirs_variant_has_no_viirs_fields():
    remote, environment = _open_carbon_feature_sets("opencarbon_monthly_noviirs")
    assert not set(VIIRS_COLUMNS).intersection(remote + environment)
    assert not any("ntl_" in column for column in remote + environment)


def test_open_carbon_poi_encoder_shape():
    encoder = POIEncoder(channels=17, representation_dim=32)
    output = encoder(torch.zeros(2, 17, 256, 256))
    assert output.shape == (2, 32)


def test_open_carbon_forward_shape():
    model = OpenCarbonModel(8, 6, representation_dim=16)
    neighborhood_indices = torch.tensor([
        [0, 0, 0, 0, 0, 1, 0, 2, 0],
        [0, 0, 0, 0, 1, 0, 2, 0, 0],
    ])
    neighborhood_mask = torch.tensor([
        [False, False, False, False, True, True, False, True, False],
        [False, False, False, False, True, False, True, False, False],
    ])
    output, poi, remote = model(
        torch.zeros(3, 17, 256, 256), torch.zeros(3, 8),
        torch.zeros(3, 6), neighborhood_indices, neighborhood_mask,
    )
    assert output.shape == (2,)
    assert poi.shape == remote.shape == (2, 16)


def test_poi_modis_scope_excludes_environment_and_viirs():
    remote, environment = _open_carbon_features_for_scope("opencarbon_core", "poi_modis")
    assert remote == [
        "modis_ndvi_mean", "modis_ndvi_mean_pixel_count", "modis_evi_mean",
        "modis_evi_mean_pixel_count", "modis_red_reflectance_mean",
        "modis_red_reflectance_mean_pixel_count", "modis_nir_reflectance_mean",
        "modis_nir_reflectance_mean_pixel_count",
    ]
    assert environment == []


def test_mean_mlp_gate_uses_masked_mean_including_center():
    aggregator = MeanMLPGatedNeighborhoodAggregator(representation_dim=2, dropout=0.0).eval()
    with torch.no_grad():
        first = aggregator.neighborhood_mlp[0]
        second = aggregator.neighborhood_mlp[3]
        first.weight.copy_(torch.eye(2))
        first.bias.zero_()
        second.weight.copy_(torch.eye(2))
        second.bias.zero_()
        aggregator.gate.weight.zero_()
        aggregator.gate.bias.zero_()
    nodes = torch.tensor([[2.0, 4.0], [6.0, 8.0], [100.0, 100.0]])
    indices = torch.tensor([[2, 2, 2, 2, 0, 1, 2, 2, 2]])
    mask = torch.tensor([[False, False, False, False, True, True, False, False, False]])
    center, fused = aggregator(nodes, indices, mask)
    assert torch.equal(center, torch.tensor([[2.0, 4.0]]))
    assert torch.allclose(fused, torch.tensor([[3.0, 5.0]]))


def test_mean_mlp_gate_masked_slots_do_not_affect_result():
    aggregator = MeanMLPGatedNeighborhoodAggregator(representation_dim=4, dropout=0.0).eval()
    nodes = torch.randn(3, 4)
    mask = torch.tensor([[False, False, False, False, True, True, False, False, False]])
    first = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1, 1]])
    second = torch.tensor([[2, 2, 2, 2, 0, 1, 2, 2, 2]])
    with torch.no_grad():
        _, first_result = aggregator(nodes, first, mask)
        _, second_result = aggregator(nodes, second, mask)
    assert torch.allclose(first_result, second_result)


def test_open_carbon_selects_aggregation_and_rejects_unknown_value():
    mean_model = OpenCarbonModel(8, 6, 16, neighborhood_aggregation="mean_mlp_gate")
    assert isinstance(mean_model.neighborhood_aggregator, MeanMLPGatedNeighborhoodAggregator)
    try:
        OpenCarbonModel(8, 6, 16, neighborhood_aggregation="unknown")
    except ValueError as error:
        assert "Unknown neighborhood_aggregation" in str(error)
    else:
        raise AssertionError("unknown neighborhood aggregation was accepted")


def test_neighborhood_aggregation_checkpoint_compatibility():
    _validate_neighborhood_aggregation({}, "spatial_attention")
    _validate_neighborhood_aggregation({"neighborhood_aggregation": "mean_mlp_gate"}, "mean_mlp_gate")
    try:
        _validate_neighborhood_aggregation({}, "mean_mlp_gate")
    except ValueError as error:
        assert "spatial_attention" in str(error)
    else:
        raise AssertionError("legacy checkpoint was accepted for mean_mlp_gate")

def test_masked_neighborhood_slots_do_not_affect_context():
    aggregator = NeighborhoodAggregator(representation_dim=8, dropout=0.0).eval()
    nodes = torch.randn(3, 8)
    mask = torch.tensor([[False, False, False, False, True, True, False, False, False]])
    first = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1, 1]])
    second = torch.tensor([[2, 2, 2, 2, 0, 1, 2, 2, 2]])
    with torch.no_grad():
        first_center, first_context = aggregator(nodes, first, mask)
        second_center, second_context = aggregator(nodes, second, mask)
    assert torch.equal(first_center, second_center)
    assert torch.allclose(first_context, second_context)


def test_neighbor_masking_keeps_only_present_split_nodes():
    frame = pd.DataFrame({
        "city_id": ["x", "x"], "period": ["202101", "202101"], "cell_id": ["a", "b"],
    })
    neighbors = pd.DataFrame({
        "city_id": ["x", "x", "x"], "period": ["202101"] * 3,
        "cell_id": ["a", "a", "b"], "neighbor_cell_id": ["a", "outside", "a"],
    })
    sources, targets = masked_edges(frame, neighbors)
    assert list(zip(sources.tolist(), targets.tolist())) == [(0, 0), (1, 0)]
    adjacency = normalized_adjacency(frame, neighbors, torch.device("cpu"))
    assert adjacency.shape == (2, 2)
    assert adjacency._nnz() == 2


def test_fixed_neighborhood_is_row_major_and_masks_outside_split():
    frame = pd.DataFrame({
        "city_id": ["x", "x", "x"], "period": ["202101"] * 3,
        "cell_id": ["center", "right", "down"],
    })
    neighbors = pd.DataFrame({
        "city_id": ["x"] * 6, "period": ["202101"] * 6,
        "cell_id": ["center", "center", "center", "center", "right", "down"],
        "neighbor_cell_id": ["center", "right", "down", "outside", "right", "down"],
        "offset_row": [0, 0, 1, -1, 0, 0],
        "offset_col": [0, 1, 0, 0, 0, 0],
    })
    neighborhoods = fixed_neighborhood_indices(frame, neighbors)
    assert neighborhoods.shape == (3, 9)
    assert neighborhoods[0].tolist() == [-1, -1, -1, -1, 0, 1, -1, 2, -1]
    assert neighborhoods[1, 4] == 1


def test_sparse_poi_can_be_densified():
    store = SparsePOIStore(ROOT / "data" / "features" / "poi_sparse")
    dense = store.dense("chicago", "202101", "r0_c32")
    assert dense.shape == (17, 256, 256)
    expected = pd.read_csv(ROOT / "raw_data" / "Chicago" / "POI" / "poi_grid_202101.csv")
    expected_total = int(expected.loc[expected["cell_id"] == "r0_c32", "poi_total"].iloc[0])
    assert int(dense.sum()) == expected_total


def test_monthly_metrics_allow_undefined_small_r2():
    predictions = pd.DataFrame({
        "period": ["202101"] * 4,
        "y_log": [0.0, 1.0, 2.0, 3.0], "pred_log": [0.0, 1.1, 1.9, 3.0],
        "y_tc": [0.0, 1.0, 2.0, 3.0], "pred_tc": [0.0, 1.0, 2.0, 3.0],
    })
    monthly, summary = calculate_metrics(predictions)
    assert np.isnan(monthly.loc[0, "log_r2"])
    assert monthly.loc[0, "r2_status"].startswith("undefined")
    assert summary["single_seed_preliminary"] is True


def test_real_split_manifests_are_grid_consistent_and_disjoint():
    split_root = ROOT / "data" / "splits"
    manifests = sorted((split_root / "cross_city").glob("*.parquet"))
    manifests += sorted((split_root / "cross_region").glob("*.parquet"))
    assert len(manifests) == 8
    for path in manifests:
        frame = pd.read_parquet(path)
        assert not frame.duplicated(["city_id", "cell_id", "period"]).any()
        assert frame.groupby(["city_id", "cell_id"])["split"].nunique().max() == 1
        sets = {
            name: set(map(tuple, group[["city_id", "cell_id", "period"]].to_numpy()))
            for name, group in frame.groupby("split")
        }
        assert sets["train"].isdisjoint(sets["validation"])
        assert sets["train"].isdisjoint(sets["test"])
        assert sets["validation"].isdisjoint(sets["test"])
