"""Tests for CouplingFlow search space behavior."""

from conftest import FakeTrial

from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace


def test_default_sampling_skips_optional_dimensions():
    params = CouplingFlowSpace(include_optional=False).sample(FakeTrial())

    assert "cf_depth" in params
    assert "cf_subnet_width" in params
    assert "cf_subnet_depth" in params
    assert "cf_dropout" in params

    # cf_activation is now optional (default=False), falls back to "mish"
    assert "cf_activation" not in params

    assert "cf_transform" not in params
    assert "cf_permutation" not in params
    assert "cf_actnorm" not in params


def test_optional_sampling_includes_optional_dimensions():
    params = CouplingFlowSpace(include_optional=True).sample(FakeTrial())

    assert "cf_transform" in params
    assert "cf_permutation" in params
    assert "cf_actnorm" in params
