"""Tests for FlowMatching search space behavior."""

import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.inference.flow_matching import (
    FlowMatchingSpace,
    _flowmatching_integrate_default,
    _timemlp_default,
)


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

    Neither ``TimeMLP``'s exact parameter set nor
    ``FlowMatching.INTEGRATE_DEFAULT_CONFIG`` are part of BayesFlow's
    public API contract — both have changed across BayesFlow releases
    (e.g. a parameter being renamed/removed, or the class attribute
    disappearing entirely). Production code already handles this via
    ``_timemlp_default``/``_flowmatching_integrate_default``, which fall
    back to safe constants on ``KeyError``/missing attributes. Assert
    against those same helpers (the actual synchronization mechanism
    the search space uses) rather than raw ``inspect.signature``/attribute
    access, so this test validates real BayesFlow defaults when
    introspectable and validates the documented fallback behavior when
    not, instead of crashing on BayesFlow version drift.
    """
    space = FlowMatchingSpace()

    assert space.activation.constant == _timemlp_default("activation", "mish")
    assert space.time_embedding_dim.constant == int(
        _timemlp_default("time_embedding_dim", 32)
    )
    assert space.merge.constant == _timemlp_default("merge", "concat")
    assert space.norm.constant == _timemlp_default("norm", "layer")
    assert space.residual.constant == bool(_timemlp_default("residual", True))
    assert space.spectral_normalization.constant == bool(
        _timemlp_default("spectral_normalization", False)
    )
    assert space.kernel_initializer.constant == _timemlp_default(
        "kernel_initializer", "he_normal"
    )
    assert space.integrate_method.constant == _flowmatching_integrate_default(
        "method", "tsit5"
    )
    assert space.integrate_steps.constant == _flowmatching_integrate_default(
        "steps", "adaptive"
    )


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
