"""Tests for CouplingFlow search space behavior."""

from conftest import FakeTrial

from bayesflow_hpo.search_spaces.base import BoolDimension
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace


def test_sampling_includes_all_dimensions():
    params = CouplingFlowSpace().sample(FakeTrial())

    # Tuned dimensions
    assert "cf_depth" in params
    assert "cf_subnet_width" in params
    assert "cf_subnet_depth" in params
    assert "cf_dropout" in params

    # Constants (always present)
    assert params["cf_activation"] == "silu"
    assert params["cf_transform"] == "affine"
    assert params["cf_permutation"] == "random"
    assert params["cf_use_actnorm"] is True


def test_user_can_widen_constant_to_tunable():
    space = CouplingFlowSpace(
        use_actnorm=BoolDimension("cf_use_actnorm")
    )
    params = space.sample(FakeTrial())
    # Now sampled via Optuna, not constant
    assert "cf_use_actnorm" in params
    assert "cf_use_actnorm" not in space.constants
