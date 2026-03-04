"""Tests for CouplingFlow search space behavior."""

from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace


class FakeTrial:
    def suggest_int(self, name, low, high, step=None, log=False):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


def test_default_sampling_skips_optional_dimensions():
    params = CouplingFlowSpace(include_optional=False).sample(FakeTrial())

    assert "cf_depth" in params
    assert "cf_subnet_width" in params
    assert "cf_subnet_depth" in params
    assert "cf_dropout" in params
    assert "cf_activation" in params

    assert "cf_transform" not in params
    assert "cf_permutation" not in params
    assert "cf_actnorm" not in params


def test_optional_sampling_includes_optional_dimensions():
    params = CouplingFlowSpace(include_optional=True).sample(FakeTrial())

    assert "cf_transform" in params
    assert "cf_permutation" in params
    assert "cf_actnorm" in params
