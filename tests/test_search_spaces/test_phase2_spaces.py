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



_BUILD_OMISSION_CASES = [
    pytest.param(
        CouplingFlowSpace(),
        "bayesflow_hpo.search_spaces.inference.coupling_flow.bf.networks.CouplingFlow",
        {
            "cf_depth": 4,
            "cf_subnet_width": 64,
            "cf_subnet_depth": 2,
            "cf_dropout": 0.1,
        },
        ["transform", "permutation", "use_actnorm"],
        ["activation"],
        id="CouplingFlow",
    ),
    pytest.param(
        DiffusionModelSpace(),
        "bayesflow_hpo.search_spaces.inference.diffusion.bf.networks.DiffusionModel",
        {
            "dm_subnet_width": 64,
            "dm_subnet_depth": 2,
            "dm_dropout": 0.1,
            "dm_activation": "mish",
        },
        ["noise_schedule", "prediction_type"],
        [],
        id="DiffusionModel",
    ),
    pytest.param(
        ConsistencyModelSpace(),
        "bayesflow_hpo.search_spaces.inference.consistency.bf.networks.ConsistencyModel",
        {
            "cm_subnet_width": 64,
            "cm_subnet_depth": 2,
            "cm_dropout": 0.1,
            "epochs": 10,
            "batches_per_epoch": 10,
        },
        ["max_time", "sigma2", "s0", "s1"],
        [],
        id="ConsistencyModel",
    ),
    pytest.param(
        StableConsistencyModelSpace(),
        "bayesflow_hpo.search_spaces.inference.stable_consistency.bf.networks.StableConsistencyModel",
        {
            "scm_subnet_width": 64,
            "scm_subnet_depth": 2,
            "scm_dropout": 0.1,
        },
        ["sigma"],
        [],
        id="StableConsistencyModel",
    ),
    pytest.param(
        DeepSetSpace(),
        "bayesflow_hpo.search_spaces.summary.deep_set.bf.networks.DeepSet",
        {
            "ds_summary_dim": 16,
            "ds_depth": 2,
            "ds_width": 64,
            "ds_dropout": 0.1,
        },
        [
            "activation",
            "spectral_normalization",
            "inner_pooling",
            "output_pooling",
        ],
        [],
        id="DeepSet",
    ),
    pytest.param(
        SetTransformerSpace(),
        "bayesflow_hpo.search_spaces.summary.set_transformer.bf.networks.SetTransformer",
        {
            "st_summary_dim": 16,
            "st_embed_dim": 64,
            "st_num_heads": 4,
            "st_num_layers": 2,
            "st_dropout": 0.1,
        },
        ["num_inducing_points", "mlp_widths", "mlp_depths"],
        [],
        id="SetTransformer",
    ),
    pytest.param(
        TimeSeriesNetworkSpace(),
        "bayesflow_hpo.search_spaces.summary.time_series_network.bf.networks.TimeSeriesNetwork",
        {
            "tsn_summary_dim": 16,
            "tsn_recurrent_dim": 64,
            "tsn_filters": 32,
            "tsn_dropout": 0.1,
        },
        ["recurrent_type", "bidirectional", "skip_steps"],
        [],
        id="TimeSeriesNetwork",
    ),
    pytest.param(
        TimeSeriesTransformerSpace(),
        "bayesflow_hpo.search_spaces.summary.time_series_transformer.bf.networks.TimeSeriesTransformer",
        {
            "tst_summary_dim": 16,
            "tst_embed_dim": 64,
            "tst_num_heads": 4,
            "tst_num_layers": 2,
            "tst_dropout": 0.1,
        },
        ["time_embedding", "mlp_widths"],
        [],
        id="TimeSeriesTransformer",
    ),
    pytest.param(
        FusionTransformerSpace(),
        "bayesflow_hpo.search_spaces.summary.fusion_transformer.bf.networks.FusionTransformer",
        {
            "ft_summary_dim": 16,
            "ft_embed_dim": 64,
            "ft_num_heads": 4,
            "ft_num_layers": 2,
            "ft_template_dim": 64,
            "ft_dropout": 0.1,
        },
        ["template_type"],
        [],
        id="FusionTransformer",
    ),
]


@pytest.mark.parametrize(
    (
        "space",
        "monkeypatch_target",
        "required_params",
        "omitted_kwargs",
        "omitted_subnet_kwargs",
    ),
    _BUILD_OMISSION_CASES,
)
def test_build_omits_optional_kwargs_when_absent(
    monkeypatch,
    space,
    monkeypatch_target,
    required_params,
    omitted_kwargs,
    omitted_subnet_kwargs,
):
    captured = {}

    class FakeNetwork:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(monkeypatch_target, FakeNetwork)

    space.build(required_params)
    for key in omitted_kwargs:
        assert key not in captured, f"{key!r} should be omitted"
    subnet = captured.get("subnet_kwargs", {})
    for key in omitted_subnet_kwargs:
        assert key not in subnet, f"subnet_kwargs[{key!r}] should be omitted"


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
