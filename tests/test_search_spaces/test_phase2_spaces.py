"""Tests for search spaces: sampling, constants, validation, and build."""

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

# -- Constants are injected during sampling ----------------------------------


@pytest.mark.parametrize(
    ("space", "constant_key", "expected_value"),
    [
        (DiffusionModelSpace(), "dm_noise_schedule", "edm"),
        (DiffusionModelSpace(), "dm_prediction_type", "F"),
        (ConsistencyModelSpace(), "cm_max_time", 200),
        (ConsistencyModelSpace(), "cm_sigma2", 1.0),
        (ConsistencyModelSpace(), "cm_s0", 10),
        (ConsistencyModelSpace(), "cm_s1", 50),
        (StableConsistencyModelSpace(), "scm_sigma", 1.0),
        (CouplingFlowSpace(), "cf_use_actnorm", True),
        (CouplingFlowSpace(), "cf_transform", "affine"),
        (CouplingFlowSpace(), "cf_permutation", "random"),
        (CouplingFlowSpace(), "cf_activation", "silu"),
        (DeepSetSpace(), "ds_activation", "silu"),
        (DeepSetSpace(), "ds_spectral_normalization", False),
        (DeepSetSpace(), "ds_inner_pooling", "mean"),
        (DeepSetSpace(), "ds_output_pooling", "mean"),
        (SetTransformerSpace(), "st_mlp_width", 128),
        (SetTransformerSpace(), "st_mlp_depth", 2),
        (SetTransformerSpace(), "st_num_inducing_points", None),
        (TimeSeriesNetworkSpace(), "tsn_recurrent_type", "gru"),
        (TimeSeriesNetworkSpace(), "tsn_bidirectional", True),
        (TimeSeriesNetworkSpace(), "tsn_skip_steps", 4),
        (TimeSeriesTransformerSpace(), "tst_mlp_width", 128),
        (TimeSeriesTransformerSpace(), "tst_time_embedding", "time2vec"),
        (FusionTransformerSpace(), "ft_template_type", "lstm"),
    ],
)
def test_constant_injected_during_sampling(space, constant_key, expected_value):
    params = space.sample(FakeTrial())
    assert constant_key in params
    assert params[constant_key] == expected_value


# -- .constants property returns all fixed dims ------------------------------


@pytest.mark.parametrize(
    ("space", "expected_constant_keys"),
    [
        (
            CouplingFlowSpace(),
            {"cf_activation", "cf_transform", "cf_permutation", "cf_use_actnorm"},
        ),
        (
            DiffusionModelSpace(),
            {"dm_noise_schedule", "dm_prediction_type"},
        ),
        (
            ConsistencyModelSpace(),
            {"cm_max_time", "cm_sigma2", "cm_s0", "cm_s1"},
        ),
        (
            StableConsistencyModelSpace(),
            {"scm_sigma"},
        ),
        (
            DeepSetSpace(),
            {
                "ds_activation",
                "ds_spectral_normalization",
                "ds_inner_pooling",
                "ds_output_pooling",
            },
        ),
        (
            SetTransformerSpace(),
            {"st_mlp_width", "st_mlp_depth", "st_num_inducing_points"},
        ),
        (
            FusionTransformerSpace(),
            {"ft_template_type"},
        ),
        (
            TimeSeriesNetworkSpace(),
            {"tsn_recurrent_type", "tsn_bidirectional", "tsn_skip_steps"},
        ),
        (
            TimeSeriesTransformerSpace(),
            {"tst_mlp_width", "tst_time_embedding"},
        ),
    ],
)
def test_constants_property(space, expected_constant_keys):
    assert set(space.constants.keys()) == expected_constant_keys


# -- Build validates all keys are present ------------------------------------


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


# -- ConsistencyModel total_steps computation --------------------------------


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
    params = space.sample(FakeTrial())
    params["epochs"] = 10
    params["num_batches"] = 11
    space.build(params)
    assert captured["total_steps"] == 110
