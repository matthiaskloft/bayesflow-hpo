"""Metric computation for fixed-validation datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bayesflow_hpo.validation.sbc_tests import compute_sbc_c2st, compute_sbc_uniformity_tests


def compute_batch_metrics(
    draws: np.ndarray,
    true_values: np.ndarray,
    cond_id: int,
    sim_id_start: int,
    coverage_levels: list[float],
) -> pd.DataFrame:
    """Compute per-simulation metrics for a single condition batch."""
    n_sims, _ = draws.shape

    posterior_mean = np.mean(draws, axis=1)
    posterior_median = np.median(draws, axis=1)
    posterior_sd = np.std(draws, axis=1)
    posterior_var = np.var(draws, axis=1, ddof=1)

    errors = posterior_mean - true_values
    sbc_ranks = np.sum(draws < true_values[:, None], axis=1)

    coverage_results: dict[str, Any] = {}
    for level in coverage_levels:
        alpha = 1 - level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        lower = np.quantile(draws, lower_q, axis=1)
        upper = np.quantile(draws, upper_q, axis=1)
        covered = (true_values >= lower) & (true_values <= upper)
        level_int = int(level * 100)
        coverage_results[f"covered_{level_int}"] = covered

    return pd.DataFrame(
        {
            "id_cond": np.full(n_sims, cond_id, dtype=np.int32),
            "id_sim": np.arange(sim_id_start, sim_id_start + n_sims, dtype=np.int32),
            "true_value": true_values,
            "posterior_mean": posterior_mean,
            "posterior_median": posterior_median,
            "posterior_sd": posterior_sd,
            "posterior_var": posterior_var,
            "error": errors,
            "squared_error": errors**2,
            "abs_error": np.abs(errors),
            "sbc_rank": sbc_ranks,
            **coverage_results,
        }
    )


def aggregate_metrics(
    sim_metrics: pd.DataFrame,
    coverage_levels: list[float],
    n_posterior_samples: int,
) -> dict[str, Any]:
    """Aggregate simulation-level metrics into summary metrics."""
    if "param_key" in sim_metrics.columns:
        return _aggregate_multi_parameter_metrics(
            sim_metrics=sim_metrics,
            coverage_levels=coverage_levels,
            n_posterior_samples=n_posterior_samples,
        )

    return _aggregate_single_parameter_metrics(
        sim_metrics=sim_metrics,
        coverage_levels=coverage_levels,
        n_posterior_samples=n_posterior_samples,
    )


def _aggregate_single_parameter_metrics(
    sim_metrics: pd.DataFrame,
    coverage_levels: list[float],
    n_posterior_samples: int,
) -> dict[str, Any]:
    """Aggregate metrics for one inferred parameter."""
    true_values = sim_metrics["true_value"].values
    prior_var = np.var(true_values, ddof=1)
    prior_range = np.max(true_values) - np.min(true_values)

    sim_metrics = sim_metrics.copy()
    sim_metrics["contraction"] = np.clip(
        1 - (sim_metrics["posterior_var"].values / prior_var),
        0,
        1,
    )

    cond_metrics_list: list[dict[str, Any]] = []
    for cond_id in sim_metrics["id_cond"].unique():
        cond_data = sim_metrics[sim_metrics["id_cond"] == cond_id]
        errors = cond_data["error"].values

        cond_row: dict[str, Any] = {
            "id_cond": cond_id,
            "n_sims": len(cond_data),
            "true_value": float(cond_data["true_value"].iloc[0]),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "nrmse": float(np.sqrt(np.mean(errors**2)) / prior_range) if prior_range > 0 else np.nan,
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "mean_posterior_sd": float(cond_data["posterior_sd"].mean()),
            "mean_contraction": float(cond_data["contraction"].mean()),
        }

        cal_errors: list[float] = []
        for level in coverage_levels:
            level_int = int(level * 100)
            coverage = float(cond_data[f"covered_{level_int}"].mean())
            cal_error = abs(coverage - level)
            cond_row[f"coverage_{level_int}"] = coverage
            cond_row[f"cal_error_{level_int}"] = cal_error
            cal_errors.append(cal_error)

        cond_row["mean_cal_error"] = float(np.mean(cal_errors))
        cond_metrics_list.append(cond_row)

    condition_metrics = pd.DataFrame(cond_metrics_list).sort_values("id_cond")

    errors = sim_metrics["error"].values
    squared_errors = sim_metrics["squared_error"].values
    sbc_ranks = sim_metrics["sbc_rank"].values

    summary: dict[str, Any] = {
        "overall_rmse": float(np.sqrt(np.mean(squared_errors))),
        "overall_nrmse": float(np.sqrt(np.mean(squared_errors)) / prior_range) if prior_range > 0 else np.nan,
        "overall_mae": float(np.mean(np.abs(errors))),
        "overall_bias": float(np.mean(errors)),
        "mean_contraction": float(sim_metrics["contraction"].mean()),
        "mean_cal_error": float(condition_metrics["mean_cal_error"].mean()),
        "n_posterior_samples": int(n_posterior_samples),
    }

    for level in coverage_levels:
        level_int = int(level * 100)
        summary[f"coverage_{level_int}"] = float(sim_metrics[f"covered_{level_int}"].mean())

    summary.update(compute_sbc_uniformity_tests(sbc_ranks, n_posterior_samples))
    summary.update(compute_sbc_c2st(sbc_ranks, n_posterior_samples))

    return {
        "condition_metrics": condition_metrics,
        "simulation_metrics": sim_metrics,
        "summary": summary,
    }


def _aggregate_multi_parameter_metrics(
    sim_metrics: pd.DataFrame,
    coverage_levels: list[float],
    n_posterior_samples: int,
) -> dict[str, Any]:
    """Aggregate metrics for multiple inferred parameters."""
    per_parameter: dict[str, dict[str, Any]] = {}
    condition_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for param_key in sim_metrics["param_key"].unique():
        param_data = sim_metrics[sim_metrics["param_key"] == param_key].copy()
        single_result = _aggregate_single_parameter_metrics(
            sim_metrics=param_data,
            coverage_levels=coverage_levels,
            n_posterior_samples=n_posterior_samples,
        )

        condition_df = single_result["condition_metrics"].copy()
        condition_df.insert(0, "param_key", param_key)
        condition_frames.append(condition_df)

        summary_row = dict(single_result["summary"])
        summary_row["param_key"] = param_key
        summary_rows.append(summary_row)

        per_parameter[str(param_key)] = {
            "condition_metrics": condition_df,
            "summary": single_result["summary"],
        }

    summary_df = pd.DataFrame(summary_rows)
    agg_summary: dict[str, Any] = {
        "n_posterior_samples": int(n_posterior_samples),
        "n_parameters": int(summary_df.shape[0]),
        "mean_cal_error": float(summary_df["mean_cal_error"].mean()),
        "overall_rmse": float(summary_df["overall_rmse"].mean()),
        "overall_nrmse": float(summary_df["overall_nrmse"].mean()),
        "overall_mae": float(summary_df["overall_mae"].mean()),
        "overall_bias": float(summary_df["overall_bias"].mean()),
        "mean_contraction": float(summary_df["mean_contraction"].mean()),
        "mean_cal_error_by_param": {
            str(row["param_key"]): float(row["mean_cal_error"])
            for row in summary_rows
        },
    }

    for level in coverage_levels:
        level_int = int(level * 100)
        key = f"coverage_{level_int}"
        agg_summary[key] = float(summary_df[key].mean())

    condition_metrics = pd.concat(condition_frames, ignore_index=True)
    return {
        "condition_metrics": condition_metrics,
        "simulation_metrics": sim_metrics,
        "summary": agg_summary,
        "per_parameter": per_parameter,
        "parameter_summary": summary_df,
    }
