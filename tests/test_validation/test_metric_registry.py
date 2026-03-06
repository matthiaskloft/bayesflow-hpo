"""Tests for the metric registry."""

import numpy as np
import pytest

from bayesflow_hpo.validation.registry import (
    DEFAULT_METRICS,
    get_metric,
    list_metrics,
    make_coverage_metric,
    register_metric,
    resolve_metrics,
)


def test_list_metrics_returns_builtin_names():
    names = list_metrics()
    assert "calibration_error" in names
    assert "coverage" in names
    assert "sbc" in names
    assert "bias" in names
    assert "mae" in names
    assert "correlation" in names


def test_resolve_metrics_returns_callable_dict():
    fns = resolve_metrics(["bias", "mae"])
    assert callable(fns["bias"])
    assert callable(fns["mae"])


def test_get_metric_alias():
    fn = get_metric("cal_error")
    assert fn is get_metric("calibration_error")


def test_unknown_metric_raises():
    with pytest.raises(KeyError, match="nonexistent_metric"):
        get_metric("nonexistent_metric")


def test_register_custom_metric():
    def my_metric(draws, true_values):
        return {"my_val": 42.0}

    register_metric("_test_custom", my_metric, overwrite=True)
    fn = get_metric("_test_custom")
    rng = np.random.default_rng(0)
    result = fn(rng.normal(size=(10, 50)), rng.normal(size=10))
    assert result["my_val"] == 42.0


def test_register_duplicate_raises():
    def dummy(d, t):
        return {}

    register_metric("_test_dup", dummy, overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        register_metric("_test_dup", dummy, overwrite=False)


def test_default_metrics_all_resolvable():
    fns = resolve_metrics(DEFAULT_METRICS)
    assert len(fns) == len(DEFAULT_METRICS)


def test_coverage_metric_two_sided():
    rng = np.random.default_rng(42)
    n_sims, n_samples = 500, 1000
    true_values = rng.normal(size=n_sims)
    draws = rng.normal(loc=true_values[:, None], scale=1.0, size=(n_sims, n_samples))

    fn = make_coverage_metric(levels=[0.9], side="two-sided")
    result = fn(draws, true_values)
    assert "coverage_90" in result
    assert "mean_cal_error" in result
    # With well-calibrated draws, coverage should be close to 0.9
    assert 0.8 < result["coverage_90"] <= 1.0


def test_coverage_metric_left_sided():
    fn = make_coverage_metric(levels=[0.5, 0.9], side="left", prefix="left_")
    rng = np.random.default_rng(7)
    result = fn(rng.normal(size=(200, 500)), rng.normal(size=200))
    assert "left_coverage_50" in result
    assert "left_coverage_90" in result
    assert "left_mean_cal_error" in result


def test_coverage_metric_right_sided():
    fn = make_coverage_metric(levels=[0.5, 0.9], side="right", prefix="right_")
    rng = np.random.default_rng(7)
    result = fn(rng.normal(size=(200, 500)), rng.normal(size=200))
    assert "right_coverage_50" in result
    assert "right_mean_cal_error" in result


def test_coverage_metric_weighted():
    fn = make_coverage_metric(levels=[0.5, 0.9], weights=[1.0, 3.0])
    rng = np.random.default_rng(7)
    result = fn(rng.normal(size=(200, 500)), rng.normal(size=200))
    assert "mean_cal_error" in result


def test_coverage_metric_invalid_side():
    with pytest.raises(ValueError, match="side must be"):
        make_coverage_metric(side="invalid")


def test_coverage_metric_mismatched_weights():
    with pytest.raises(ValueError, match="weights length"):
        make_coverage_metric(levels=[0.5], weights=[1.0, 2.0])


def test_bias_metric_correct():
    rng = np.random.default_rng(0)
    true_values = np.zeros(100)
    draws = rng.normal(loc=0.5, size=(100, 200))  # biased by +0.5
    fn = get_metric("bias")
    result = fn(draws, true_values)
    assert abs(result["bias"] - 0.5) < 0.1


def test_mae_metric_correct():
    true_values = np.zeros(100)
    draws = np.full((100, 200), 1.0)  # all draws = 1.0, true = 0.0
    fn = get_metric("mae")
    result = fn(draws, true_values)
    assert abs(result["mae"] - 1.0) < 1e-9


def test_rmse_metric_smoke():
    rng = np.random.default_rng(123)
    true_values = rng.normal(size=64)
    draws = rng.normal(loc=true_values[:, None], scale=0.5, size=(64, 128))
    fn = get_metric("rmse")
    result = fn(draws, true_values)
    assert "rmse" in result
    assert np.isfinite(result["rmse"])


def test_nrmse_metric_smoke():
    rng = np.random.default_rng(456)
    true_values = rng.normal(size=64)
    draws = rng.normal(loc=true_values[:, None], scale=0.5, size=(64, 128))
    fn = get_metric("nrmse")
    result = fn(draws, true_values)
    assert "nrmse" in result
    assert np.isfinite(result["nrmse"])


def test_correlation_metric_perfect():
    true_values = np.linspace(-2, 2, 100)
    # Posterior means = true values (perfect correlation)
    draws = np.tile(true_values[:, None], (1, 50))
    fn = get_metric("correlation")
    result = fn(draws, true_values)
    assert abs(result["correlation"] - 1.0) < 1e-6


def test_correlation_metric_alias():
    assert get_metric("corr") is get_metric("correlation")


def test_correlation_metric_constant_true_values():
    """Constant true values → correlation should be 0, not NaN."""
    true_values = np.ones(50)
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(50, 100))
    fn = get_metric("correlation")
    result = fn(draws, true_values)
    assert result["correlation"] == 0.0
