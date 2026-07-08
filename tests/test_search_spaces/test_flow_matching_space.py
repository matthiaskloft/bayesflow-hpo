"""Tests for FlowMatching search space behavior."""

import inspect

import bayesflow as bf
import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.inference.flow_matching import FlowMatchingSpace


def test_sampling_includes_all_dimensions():
    params = FlowMatchingSpace().sample(FakeTrial())

    # Tuned dimensions
    assert "fm_subnet_width" in params
    assert "fm_subnet_depth" in params
    assert "fm_dropout" in params

    # Constants (always present with BF defaults)
    assert params["fm_activation"] == "mish"
    assert params["fm_use_optimal_transport"] is False
    assert params["fm_time_power_law_alpha"] == 0.0
    assert params["fm_time_embedding_dim"] == 32
    assert params["fm_integrate_method"] == "tsit5"
    assert params["fm_integrate_steps"] == "adaptive"
    assert params["fm_merge"] == "concat"
    assert params["fm_norm"] == "layer"
    assert params["fm_residual"] is True
    assert params["fm_spectral_normalization"] is False
    assert params["fm_kernel_initializer"] == "he_normal"


def test_build_validates_required_keys():
    with pytest.raises(ValueError, match="FlowMatchingSpace.build"):
        FlowMatchingSpace().build({})


def test_build_passes_all_params(monkeypatch):
    captured = {}

    class FakeFlowMatching:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "bayesflow_hpo.search_spaces.inference.flow_matching.bf.networks.FlowMatching",
        FakeFlowMatching,
    )

    space = FlowMatchingSpace()
    params = space.sample(FakeTrial())
    space.build(params)
    assert captured["use_optimal_transport"] is False
    assert captured["time_power_law_alpha"] == 0.0
    assert captured["subnet_kwargs"]["activation"] == "mish"
    assert captured["subnet_kwargs"]["time_embedding_dim"] == 32
    assert captured["subnet_kwargs"]["merge"] == "concat"
    assert captured["subnet_kwargs"]["norm"] == "layer"
    assert captured["subnet_kwargs"]["residual"] is True
    assert captured["subnet_kwargs"]["spectral_normalization"] is False
    assert captured["subnet_kwargs"]["kernel_initializer"] == "he_normal"
    assert captured["integrate_kwargs"] == {
        "method": "tsit5",
        "steps": "adaptive",
    }


def test_time_embedding_dim_is_constant():
    space = FlowMatchingSpace()
    dim = space.time_embedding_dim
    assert dim.name == "fm_time_embedding_dim"
    assert dim.constant == 32


def test_untuned_defaults_match_bayesflow_defaults():
    """Untuned constants track BayesFlow's own defaults where introspectable.

    ``FlowMatching.INTEGRATE_DEFAULT_CONFIG`` is read defensively in
    production code (:func:`_flowmatching_integrate_default`) because it
    isn't part of BayesFlow's public API contract and may be renamed or
    removed across versions. Mirror that same defensive lookup here so
    this test validates the real defaults when the attribute is
    available, and validates the documented fallback behavior
    (``"tsit5"``/``"adaptive"``) when it isn't — rather than crashing on
    BayesFlow versions that no longer expose it.
    """
    space = FlowMatchingSpace()
    timemlp_sig = inspect.signature(bf.networks.TimeMLP)
    integrate_defaults = getattr(
        bf.networks.FlowMatching, "INTEGRATE_DEFAULT_CONFIG", None
    )

    assert space.activation.constant == timemlp_sig.parameters["activation"].default
    assert (
        space.time_embedding_dim.constant
        == timemlp_sig.parameters["time_embedding_dim"].default
    )
    assert space.merge.constant == timemlp_sig.parameters["merge"].default
    assert space.norm.constant == timemlp_sig.parameters["norm"].default
    assert space.residual.constant == timemlp_sig.parameters["residual"].default
    assert (
        space.spectral_normalization.constant
        == timemlp_sig.parameters["spectral_normalization"].default
    )
    assert (
        space.kernel_initializer.constant
        == timemlp_sig.parameters["kernel_initializer"].default
    )

    if isinstance(integrate_defaults, dict):
        assert space.integrate_method.constant == integrate_defaults["method"]
        assert space.integrate_steps.constant == integrate_defaults["steps"]
    else:
        assert space.integrate_method.constant == "tsit5"
        assert space.integrate_steps.constant == "adaptive"


def test_fast_profile_samples_speed_oriented_defaults():
    params = FlowMatchingSpace.fast().sample(FakeTrial())
    assert params["fm_dropout"] == 0.0
    assert params["fm_time_embedding_dim"] == 16
    assert params["fm_merge"] == "add"
    assert params["fm_residual"] is False
    assert params["fm_integrate_method"] == "euler"
    assert params["fm_integrate_steps"] == 8
    assert params["fm_norm"] is None


def test_balanced_profile_samples_solver_choices():
    params = FlowMatchingSpace.balanced().sample(FakeTrial())
    assert params["fm_subnet_width"] == 32
    assert params["fm_subnet_depth"] == 1
    assert params["fm_integrate_method"] == "euler"
    assert params["fm_integrate_steps"] == 16
    assert params["fm_merge"] == "add"
    assert params["fm_norm"] is None


def test_quality_profile_samples_quality_range():
    params = FlowMatchingSpace.quality().sample(FakeTrial())
    assert params["fm_subnet_width"] == 96
    assert params["fm_subnet_depth"] == 3
    assert params["fm_time_embedding_dim"] == 32
    assert params["fm_use_optimal_transport"] is True
    assert params["fm_integrate_method"] == "tsit5"
    assert params["fm_integrate_steps"] == 32


def test_preset_aliases_and_validation():
    assert FlowMatchingSpace.preset("default").sample(FakeTrial())[
        "fm_integrate_steps"
    ] == "adaptive"
    assert FlowMatchingSpace.preset("FAST").sample(FakeTrial())[
        "fm_integrate_steps"
    ] == 8
    with pytest.raises(ValueError, match="Unknown FlowMatchingSpace profile"):
        FlowMatchingSpace.preset("unknown")
