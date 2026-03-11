"""Tests for objective extraction and param normalization."""

import numpy as np
import pytest

from bayesflow_hpo.objectives import (
    MIN_PARAM_COUNT,
    MAX_PARAM_COUNT,
    _metric_to_minimize,
    extract_multi_objective_values,
    extract_objective_values,
    normalize_param_count,
)


def test_metric_to_minimize_lower_is_better():
    assert _metric_to_minimize("calibration_error", 0.05) == 0.05
    assert _metric_to_minimize("nrmse", 0.2) == 0.2


def test_metric_to_minimize_higher_is_better():
    assert np.isclose(_metric_to_minimize("correlation", 0.8), 0.2)


def test_extract_multi_mean_mode():
    metrics = {
        "summary": {
            "calibration_error": 0.04,
            "nrmse": 0.10,
        }
    }
    values = extract_multi_objective_values(
        metrics,
        param_count=50_000,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="mean",
    )
    assert len(values) == 2
    assert np.isclose(values[0], np.mean([0.04, 0.10]))
    assert 0.0 < values[1] < 1.0  # param_score


def test_extract_multi_pareto_mode():
    metrics = {
        "summary": {
            "calibration_error": 0.04,
            "nrmse": 0.10,
        }
    }
    values = extract_multi_objective_values(
        metrics,
        param_count=50_000,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="pareto",
    )
    assert len(values) == 3
    assert values[0] == 0.04  # calibration_error
    assert values[1] == 0.10  # nrmse
    assert 0.0 < values[2] < 1.0  # param_score


def test_extract_multi_with_correlation():
    """Correlation is higher-is-better, so objective = 1 - corr."""
    metrics = {
        "summary": {
            "calibration_error": 0.05,
            "nrmse": 0.15,
            "correlation": 0.9,
        }
    }
    values = extract_multi_objective_values(
        metrics,
        param_count=50_000,
        objective_metrics=["calibration_error", "nrmse", "correlation"],
        objective_mode="pareto",
    )
    assert len(values) == 4
    assert values[0] == 0.05
    assert values[1] == 0.15
    assert np.isclose(values[2], 0.1)  # 1 - 0.9


def test_extract_multi_missing_metric_returns_worst():
    """Missing lower-is-better metric defaults to 1.0 (worst)."""
    metrics = {"summary": {"nrmse": 0.1}}
    values = extract_multi_objective_values(
        metrics,
        param_count=50_000,
        objective_metrics=["nrmse", "nonexistent_key"],
        objective_mode="pareto",
    )
    assert values[1] == 1.0  # fallback for missing lower-is-better key


def test_extract_multi_missing_correlation_returns_worst():
    """Missing higher-is-better metric should default to worst (1.0 after inversion)."""
    metrics = {"summary": {"calibration_error": 0.05}}
    values = extract_multi_objective_values(
        metrics,
        param_count=50_000,
        objective_metrics=["calibration_error", "correlation"],
        objective_mode="pareto",
    )
    # correlation missing -> default 0.0 -> _metric_to_minimize: 1.0 - 0.0 = 1.0
    assert values[1] == 1.0


def test_extract_legacy_applies_metric_to_minimize():
    """Legacy extract_objective_values inverts higher-is-better metrics."""
    metrics = {"summary": {"correlation": 0.9}}
    obj_val, _ = extract_objective_values(
        metrics,
        param_count=50_000,
        objective_metric="correlation",
    )
    assert np.isclose(obj_val, 0.1)  # 1 - 0.9


def test_extract_legacy_lower_is_better_unchanged():
    """Legacy path passes lower-is-better metrics through unchanged."""
    metrics = {"summary": {"calibration_error": 0.05}}
    obj_val, _ = extract_objective_values(
        metrics,
        param_count=50_000,
        objective_metric="calibration_error",
    )
    assert obj_val == 0.05


def test_extract_multi_rejects_unknown_mode():
    metrics = {"summary": {"calibration_error": 0.05}}
    with pytest.raises(ValueError, match="Unknown objective_mode"):
        extract_multi_objective_values(
            metrics,
            param_count=50_000,
            objective_metrics=["calibration_error"],
            objective_mode="weighted",
        )


# --- normalize_param_count tests ---


def test_normalize_default_range():
    """Endpoints of the default [1K, 1M] range map to 0.0 and 1.0."""
    assert normalize_param_count(MIN_PARAM_COUNT) == 0.0
    assert normalize_param_count(MAX_PARAM_COUNT) == 1.0


def test_normalize_mid_range():
    """A value inside the range should produce a score in (0, 1)."""
    score = normalize_param_count(100_000)
    assert 0.0 < score < 1.0


def test_normalize_auto_tightens_with_small_max():
    """When max_count < default MAX, min is auto-tightened to max/100."""
    # With max=100K, min auto-tightens to 1K (same as default here).
    # But with max=500K, min auto-tightens to 5K.
    low = normalize_param_count(10_000, max_count=500_000)
    high = normalize_param_count(400_000, max_count=500_000)
    assert low < 0.3, f"Low param count should map near 0, got {low}"
    assert high > 0.8, f"High param count should map near 1, got {high}"


def test_normalize_auto_tighten_spreads_values():
    """Auto-tightening should spread normalized values across [0, 1]."""
    # Without auto-tightening (max=1M default), all values in [70K, 150K]
    # cluster near 0.93-1.0.  With max=200K, they should spread out.
    scores = [
        normalize_param_count(p, max_count=200_000)
        for p in [5_000, 50_000, 100_000, 180_000]
    ]
    spread = max(scores) - min(scores)
    assert spread > 0.5, f"Expected spread > 0.5, got {spread}"


def test_normalize_zero_or_negative_returns_worst():
    assert normalize_param_count(0) == 1.0
    assert normalize_param_count(-10) == 1.0


def test_normalize_explicit_min_skips_auto_tighten():
    """When caller passes a non-default min_count, auto-tightening is skipped."""
    # min=500 is not the default 1000, so no auto-tightening
    score = normalize_param_count(500, min_count=500, max_count=100_000)
    assert score == 0.0  # at the lower bound
