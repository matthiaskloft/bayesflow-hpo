"""Tests for phase 3 validation metric aggregation."""

import numpy as np
import pandas as pd

from bayesflow_hpo.validation.metrics import aggregate_metrics, compute_batch_metrics


def test_aggregate_metrics_multi_parameter_summary():
    rng = np.random.default_rng(7)
    n_sims = 40
    n_samples = 200

    theta_true = rng.normal(loc=0.0, scale=1.0, size=n_sims)
    phi_true = rng.normal(loc=1.0, scale=1.0, size=n_sims)

    theta_draws = rng.normal(loc=theta_true[:, None], scale=0.25, size=(n_sims, n_samples))
    phi_draws = rng.normal(loc=phi_true[:, None], scale=0.25, size=(n_sims, n_samples))

    theta_metrics = compute_batch_metrics(
        draws=theta_draws,
        true_values=theta_true,
        cond_id=0,
        sim_id_start=0,
        coverage_levels=[0.8, 0.9],
    )
    theta_metrics.insert(0, "param_key", "theta")

    phi_metrics = compute_batch_metrics(
        draws=phi_draws,
        true_values=phi_true,
        cond_id=0,
        sim_id_start=0,
        coverage_levels=[0.8, 0.9],
    )
    phi_metrics.insert(0, "param_key", "phi")

    sim_metrics = pd.concat([theta_metrics, phi_metrics], ignore_index=True)
    result = aggregate_metrics(
        sim_metrics=sim_metrics,
        coverage_levels=[0.8, 0.9],
        n_posterior_samples=n_samples,
    )

    assert "per_parameter" in result
    assert set(result["per_parameter"].keys()) == {"theta", "phi"}
    assert result["summary"]["n_parameters"] == 2
    assert "mean_cal_error_by_param" in result["summary"]
    assert "param_key" in result["condition_metrics"].columns
