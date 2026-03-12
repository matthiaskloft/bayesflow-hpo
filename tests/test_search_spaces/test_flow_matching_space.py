"""Tests for FlowMatching search space behavior."""

import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.inference.flow_matching import FlowMatchingSpace


def test_default_sampling_skips_optional_dimensions():
    params = FlowMatchingSpace(include_optional=False).sample(FakeTrial())

    assert "fm_subnet_width" in params
    assert "fm_subnet_depth" in params
    assert "fm_dropout" in params

    # Optional dimensions should be absent
    assert "fm_activation" not in params
    assert "fm_use_ot" not in params
    assert "fm_time_alpha" not in params
    assert "fm_time_resolution" not in params


def test_optional_sampling_includes_optional_dimensions():
    params = FlowMatchingSpace(include_optional=True).sample(FakeTrial())

    assert "fm_use_ot" in params
    assert "fm_time_alpha" in params
    assert "fm_time_resolution" in params
    assert "fm_activation" in params


def test_build_validates_required_keys():
    with pytest.raises(ValueError, match="FlowMatchingSpace.build"):
        FlowMatchingSpace().build({})


def test_build_passes_time_resolution(monkeypatch):
    captured = {}

    class FakeFlowMatching:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "bayesflow_hpo.search_spaces.inference.flow_matching.bf.networks.FlowMatching",
        FakeFlowMatching,
    )

    space = FlowMatchingSpace()
    params = {
        "fm_subnet_width": 64,
        "fm_subnet_depth": 2,
        "fm_dropout": 0.1,
        "fm_time_resolution": 150,
    }
    space.build(params)
    assert captured["integrate_kwargs"] == {"steps": 150}


def test_build_omits_integrate_kwargs_without_time_resolution(monkeypatch):
    captured = {}

    class FakeFlowMatching:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "bayesflow_hpo.search_spaces.inference.flow_matching.bf.networks.FlowMatching",
        FakeFlowMatching,
    )

    space = FlowMatchingSpace()
    params = {
        "fm_subnet_width": 64,
        "fm_subnet_depth": 2,
        "fm_dropout": 0.1,
    }
    space.build(params)
    assert "integrate_kwargs" not in captured


def test_time_resolution_dimension_range():
    space = FlowMatchingSpace()
    dim = space.time_resolution
    assert dim.name == "fm_time_resolution"
    assert dim.low == 50
    assert dim.high == 300
    assert dim.step == 50
    assert dim.default is False
