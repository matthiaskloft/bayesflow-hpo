"""Tests for FlowMatching search space behavior."""

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
    assert params["fm_time_embedding_dim"] == 8


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
    assert captured["subnet_kwargs"]["time_embedding_dim"] == 8


def test_time_embedding_dim_is_constant():
    space = FlowMatchingSpace()
    dim = space.time_embedding_dim
    assert dim.name == "fm_time_embedding_dim"
    assert dim.constant == 8
