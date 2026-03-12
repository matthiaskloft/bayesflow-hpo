"""Tests for newly added Phase 2 search spaces."""

import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.inference.consistency import ConsistencyModelSpace
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.inference.diffusion import DiffusionModelSpace
from bayesflow_hpo.search_spaces.inference.stable_consistency import (
    StableConsistencyModelSpace,
)
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace
from bayesflow_hpo.search_spaces.summary.fusion_transformer import (
    FusionTransformerSpace,
)
from bayesflow_hpo.search_spaces.summary.set_transformer import SetTransformerSpace
from bayesflow_hpo.search_spaces.summary.time_series_network import (
    TimeSeriesNetworkSpace,
)
from bayesflow_hpo.search_spaces.summary.time_series_transformer import (
    TimeSeriesTransformerSpace,
)


@pytest.mark.parametrize(
    ("space", "default_key", "optional_key"),
    [
        (
            DiffusionModelSpace(include_optional=False),
            "dm_subnet_width",
            "dm_noise_schedule",
        ),
        (
            ConsistencyModelSpace(include_optional=False),
            "cm_subnet_width",
            "cm_max_time",
        ),
        (
            StableConsistencyModelSpace(include_optional=False),
            "scm_subnet_width",
            "scm_sigma",
        ),
        (SetTransformerSpace(include_optional=False), "st_summary_dim", "st_mlp_width"),
        (
            TimeSeriesNetworkSpace(include_optional=False),
            "tsn_summary_dim",
            "tsn_recurrent_type",
        ),
        (
            TimeSeriesTransformerSpace(include_optional=False),
            "tst_summary_dim",
            "tst_mlp_width",
        ),
        (
            FusionTransformerSpace(include_optional=False),
            "ft_summary_dim",
            "ft_template_type",
        ),
    ],
)
def test_default_sampling_skips_optional(space, default_key, optional_key):
    params = space.sample(FakeTrial())
    assert default_key in params
    assert optional_key not in params


@pytest.mark.parametrize(
    ("space", "optional_key"),
    [
        (DiffusionModelSpace(include_optional=True), "dm_noise_schedule"),
        (ConsistencyModelSpace(include_optional=True), "cm_max_time"),
        (StableConsistencyModelSpace(include_optional=True), "scm_sigma"),
        (SetTransformerSpace(include_optional=True), "st_mlp_width"),
        (TimeSeriesNetworkSpace(include_optional=True), "tsn_recurrent_type"),
        (TimeSeriesTransformerSpace(include_optional=True), "tst_mlp_width"),
        (FusionTransformerSpace(include_optional=True), "ft_template_type"),
    ],
)
def test_optional_sampling_includes_optional(space, optional_key):
    params = space.sample(FakeTrial())
    assert optional_key in params


@pytest.mark.parametrize(
    ("space", "error_prefix"),
    [
        (CouplingFlowSpace(), "CouplingFlowSpace.build"),
        (DiffusionModelSpace(), "DiffusionModelSpace.build"),
        (ConsistencyModelSpace(), "ConsistencyModelSpace.build"),
        (StableConsistencyModelSpace(), "StableConsistencyModelSpace.build"),
        (DeepSetSpace(), "DeepSetSpace.build"),
        (SetTransformerSpace(), "SetTransformerSpace.build"),
        (TimeSeriesNetworkSpace(), "TimeSeriesNetworkSpace.build"),
        (TimeSeriesTransformerSpace(), "TimeSeriesTransformerSpace.build"),
        (FusionTransformerSpace(), "FusionTransformerSpace.build"),
    ],
)
def test_build_validates_required_keys(space, error_prefix):
    with pytest.raises(ValueError, match=error_prefix):
        space.build({})


def test_consistency_total_steps_from_training_keys(monkeypatch):
    captured = {}

    class FakeConsistencyModel:
        def __init__(self, total_steps, **kwargs):
            captured["total_steps"] = total_steps

    monkeypatch.setattr(
        "bayesflow_hpo.search_spaces.inference.consistency.bf.networks.ConsistencyModel",
        FakeConsistencyModel,
    )

    space = ConsistencyModelSpace()
    params = {
        "cm_subnet_width": 64,
        "cm_subnet_depth": 2,
        "cm_dropout": 0.1,
        "epochs": 10,
        "batches_per_epoch": 11,
    }
    space.build(params)
    assert captured["total_steps"] == 110
