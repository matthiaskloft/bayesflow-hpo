"""Tests for validation metric computation and aggregation."""

import numpy as np

from bayesflow_hpo.validation.metrics import (
    aggregate_condition_rows,
    compute_condition_metrics,
)
from bayesflow_hpo.validation.registry import resolve_metrics


def test_compute_condition_metrics_returns_expected_keys():
    rng = np.random.default_rng(7)
    n_sims, n_samples = 40, 200
    true_values = rng.normal(size=n_sims)
    draws = rng.normal(loc=true_values[:, None], scale=0.25, size=(n_sims, n_samples))

    fns = resolve_metrics(["coverage", "bias", "mae"])
    row = compute_condition_metrics(draws, true_values, cond_id=0, metric_fns=fns)

    assert row["id_cond"] == 0
    assert row["n_sims"] == 40
    assert "coverage_90" in row
    assert "mean_cal_error" in row
    assert "bias" in row
    assert "mae" in row


def test_aggregate_condition_rows_averages_correctly():
    rows = [
        {"id_cond": 0, "n_sims": 10, "rmse": 0.2, "bias": 0.01},
        {"id_cond": 1, "n_sims": 10, "rmse": 0.4, "bias": -0.01},
    ]
    summary = aggregate_condition_rows(rows)
    assert abs(summary["rmse"] - 0.3) < 1e-9
    assert abs(summary["bias"] - 0.0) < 1e-9


def test_compute_condition_metrics_with_sbc():
    rng = np.random.default_rng(42)
    n_sims, n_samples = 100, 200
    true_values = rng.normal(size=n_sims)
    draws = rng.normal(loc=true_values[:, None], scale=0.25, size=(n_sims, n_samples))

    fns = resolve_metrics(["sbc"])
    row = compute_condition_metrics(draws, true_values, cond_id=0, metric_fns=fns)

    assert "sbc_ks_pvalue" in row
    assert "sbc_chi2_pvalue" in row
