"""Tests for multi-metric objective extraction."""

import numpy as np
import pytest

from bayesflow_hpo.objectives import (
    _metric_to_minimize,
    extract_multi_objective_values,
    extract_objective_values,
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
