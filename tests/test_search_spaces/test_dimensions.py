"""Tests for dimension dataclass validation."""

from dataclasses import dataclass, field

import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    BoolDimension,
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
