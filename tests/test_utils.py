"""Tests for bayesflow_hpo.utils (loguniform sampling helpers)."""

import numpy as np
import pytest

from bayesflow_hpo.utils import loguniform_float, loguniform_int


class TestLoguniformInt:
    def test_output_within_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            val = loguniform_int(1, 1000, rng=rng)
            assert 1 <= val <= 1000

    def test_deterministic_with_seed(self):
        a = loguniform_int(10, 100, rng=np.random.default_rng(0))
        b = loguniform_int(10, 100, rng=np.random.default_rng(0))
        assert a == b

    def test_returns_int(self):
        val = loguniform_int(1, 100, rng=np.random.default_rng(7))
        assert isinstance(val, int)

    def test_low_equals_high(self):
        val = loguniform_int(5, 5, rng=np.random.default_rng(0))
        assert val == 5

    def test_low_must_be_positive(self):
        with pytest.raises(ValueError, match="low must be positive"):
            loguniform_int(0, 10)

    def test_high_must_ge_low(self):
        with pytest.raises(ValueError, match="high must be >= low"):
            loguniform_int(10, 5)

    def test_alpha_shifts_distribution(self):
        rng_hi = np.random.default_rng(42)
        high_alpha = [
            loguniform_int(1, 1000, alpha=5.0, rng=rng_hi) for _ in range(500)
        ]
        rng_lo = np.random.default_rng(99)
        low_alpha = [
            loguniform_int(1, 1000, alpha=0.2, rng=rng_lo) for _ in range(500)
        ]
        # Higher alpha → U^(1/alpha) closer to 1 → samples shift toward high end
        # Lower alpha → U^(1/alpha) closer to 0 → samples shift toward low end
        assert np.mean(high_alpha) > np.mean(low_alpha)


class TestLoguniformFloat:
    def test_output_within_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            val = loguniform_float(0.001, 1.0, rng=rng)
            assert 0.001 <= val <= 1.0

    def test_returns_float(self):
        val = loguniform_float(1.0, 100.0, rng=np.random.default_rng(7))
        assert isinstance(val, float)

    def test_low_must_be_positive(self):
        with pytest.raises(ValueError, match="low must be positive"):
            loguniform_float(0.0, 1.0)

    def test_high_must_ge_low(self):
        with pytest.raises(ValueError, match="high must be >= low"):
            loguniform_float(10.0, 5.0)

    def test_deterministic_with_seed(self):
        a = loguniform_float(0.01, 1.0, rng=np.random.default_rng(0))
        b = loguniform_float(0.01, 1.0, rng=np.random.default_rng(0))
        assert a == b
