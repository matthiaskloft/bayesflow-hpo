"""Tests for dimension dataclass validation."""

import pytest

from bayesflow_hpo.search_spaces.base import IntDimension


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
