"""Tests for dimension dataclass validation."""

from dataclasses import dataclass, field

import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    BoolDimension,
    DerivedDimension,
    IntDimension,
)


class TestIntDimensionLogStepValidation:
    """IntDimension rejects log=True combined with step != 1."""

    def test_log_true_with_step_raises(self):
        with pytest.raises(ValueError, match="log=True is incompatible with step=4"):
            IntDimension("x", low=1, high=100, log=True, step=4)

    def test_log_true_with_step_2_raises(self):
        with pytest.raises(ValueError, match="log=True is incompatible with step=2"):
            IntDimension("x", low=1, high=100, log=True, step=2)

    def test_log_true_with_step_1_ok(self):
        dim = IntDimension("x", low=1, high=100, log=True, step=1)
        assert dim.log is True
        assert dim.step == 1

    def test_log_true_without_step_ok(self):
        dim = IntDimension("x", low=1, high=100, log=True)
        assert dim.log is True
        assert dim.step is None

    def test_step_without_log_ok(self):
        dim = IntDimension("x", low=1, high=100, step=4)
        assert dim.log is False
        assert dim.step == 4

    def test_constant_with_log_step_skips_validation(self):
        """Constants bypass range validation entirely."""
        dim = IntDimension("x", constant=42)
        assert dim.log is False


@dataclass
class _DummyBoolSpace(BaseSearchSpace):
    """Minimal search space with a single BoolDimension field."""

    flag: BoolDimension = field(default_factory=lambda: BoolDimension("flag"))

    def build(self, params):
        return params


class TestBoolDimension:
    """BoolDimension is discovered and sampled like other dimension types."""

    def test_constant_is_discovered_as_dimension_and_constant(self):
        space = _DummyBoolSpace(flag=BoolDimension("flag", constant=True))
        assert space.dimensions == [BoolDimension("flag", constant=True)]
        assert space.constants == {"flag": True}

    def test_non_constant_samples_via_suggest_categorical(self):
        space = _DummyBoolSpace(flag=BoolDimension("flag"))
        params = space.sample(FakeTrial())
        # FakeTrial.suggest_categorical returns choices[0]
        assert params["flag"] is True


@dataclass
class _BudgetedSpace(BaseSearchSpace):
    batch_size: IntDimension = field(
        default_factory=lambda: IntDimension("batch_size", constant=32)
    )
    epochs: IntDimension = field(
        default_factory=lambda: IntDimension("epochs", constant=10)
    )
    simulation_budget: IntDimension = field(
        default_factory=lambda: IntDimension("simulation_budget", constant=3200)
    )
    num_batches: DerivedDimension = field(
        default_factory=lambda: DerivedDimension(
            "num_batches",
            lambda p: p["simulation_budget"] // (p["batch_size"] * p["epochs"]),
        )
    )

    def build(self, params):
        return params


def test_derived_dimension_runs_after_sampled_and_constant_dimensions():
    params = _BudgetedSpace().sample(FakeTrial())
    assert params["num_batches"] == 10


def test_derived_dimension_is_not_a_constant():
    constants = _BudgetedSpace().constants
    assert constants == {
        "batch_size": 32,
        "epochs": 10,
        "simulation_budget": 3200,
    }


@dataclass
class _DuplicateDimensionSpace(BaseSearchSpace):
    sampled: IntDimension = field(
        default_factory=lambda: IntDimension("shared", low=1, high=2)
    )
    derived: DerivedDimension = field(
        default_factory=lambda: DerivedDimension("shared", lambda p: 3)
    )

    def build(self, params):
        return params


def test_duplicate_dimension_names_are_rejected_before_sampling():
    with pytest.raises(ValueError, match="duplicate dimension names: shared"):
        _DuplicateDimensionSpace().sample(FakeTrial())
