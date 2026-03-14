"""Validation pipeline on fixed ``ValidationDataset``."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from bayesflow_hpo.optimization.cleanup import cleanup_trial
from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn
from bayesflow_hpo.validation.metrics import (
    aggregate_condition_rows,
    compute_condition_metrics,
)
from bayesflow_hpo.validation.registry import DEFAULT_METRICS, resolve_metrics
from bayesflow_hpo.validation.result import ValidationResult


def run_validation_pipeline(
    approximator: Any,
    validation_data: ValidationDataset,
    n_posterior_samples: int = 1000,
    metrics: Sequence[str] | None = None,
) -> ValidationResult:
    """Run metric evaluation on a fixed dataset reused across trials.

    Parameters
    ----------
    approximator
        Trained BayesFlow approximator with a ``.sample()`` method.
    validation_data
        Pre-generated :class:`ValidationDataset`.
    n_posterior_samples
        Number of posterior draws per simulation.
    metrics
        List of metric names to compute (resolved via the registry).
        Defaults to :data:`~bayesflow_hpo.validation.registry.DEFAULT_METRICS`.

    Returns
    -------
    ValidationResult
        Structured result with per-condition and summary tables.
    """
    if metrics is None:
        metrics = list(DEFAULT_METRICS)
    metric_fns = resolve_metrics(list(metrics))

    infer_fn = make_bayesflow_infer_fn(
        approximator=approximator,
        param_keys=validation_data.param_keys,
        data_keys=validation_data.data_keys,
    )

    timing: dict[str, float] = {"inference": 0.0, "metrics": 0.0}
    n_params = len(validation_data.param_keys)
    multi_param = n_params > 1

    # Per-parameter condition rows: {param_key: [row_dicts]}
    param_condition_rows: dict[str, list[dict[str, Any]]] = {}
    if multi_param:
        for pk in validation_data.param_keys:
            param_condition_rows[pk] = []
    else:
        param_condition_rows[validation_data.param_keys[0]] = []

    for cond_id, sim_batch in enumerate(validation_data.simulations):
        # --- Inference ---
        t0 = time.perf_counter()
        draws = infer_fn(sim_batch, n_posterior_samples)
        timing["inference"] += time.perf_counter() - t0

        # --- Metrics per parameter ---
        t1 = time.perf_counter()
        if multi_param:
            if draws.ndim != 3:
                raise ValueError(
                    "Expected posterior draws with shape (n_sims, n_samples, n_params) "
                    "for multi-parameter inference."
                )
            for param_idx, param_key in enumerate(validation_data.param_keys):
                true_values = np.asarray(sim_batch[param_key]).reshape(-1)
                param_draws = np.asarray(draws[:, :, param_idx])
                row = compute_condition_metrics(
                    param_draws, true_values, cond_id, metric_fns,
                )
                param_condition_rows[param_key].append(row)
        else:
            param_key = validation_data.param_keys[0]
            true_values = np.asarray(sim_batch[param_key]).reshape(-1)
            if draws.ndim == 3 and draws.shape[-1] == 1:
                draws = np.squeeze(draws, axis=-1)
            row = compute_condition_metrics(draws, true_values, cond_id, metric_fns)
            param_condition_rows[param_key].append(row)

        timing["metrics"] += time.perf_counter() - t1
        cleanup_trial()

    # --- Assemble result ---
    n_conditions = len(validation_data.simulations)

    if multi_param:
        per_parameter: dict[str, ValidationResult] = {}
        all_condition_rows: list[dict[str, Any]] = []

        for param_key, cond_rows in param_condition_rows.items():
            param_summary = aggregate_condition_rows(cond_rows)
            param_cond_df = pd.DataFrame(cond_rows)
            per_parameter[param_key] = ValidationResult(
                condition_metrics=param_cond_df,
                summary=param_summary,
                n_conditions=n_conditions,
                n_posterior_samples=n_posterior_samples,
                metric_names=list(metrics),
            )
            for row in cond_rows:
                tagged = dict(row, param_key=param_key)
                all_condition_rows.append(tagged)

        condition_df = pd.DataFrame(all_condition_rows)
        # Overall summary: average across per-parameter summaries
        overall_summary: dict[str, float] = {}
        for key in per_parameter[validation_data.param_keys[0]].summary:
            vals = [pr.summary.get(key, float("nan")) for pr in per_parameter.values()]
            overall_summary[key] = float(np.nanmean(vals))

        return ValidationResult(
            condition_metrics=condition_df,
            summary=overall_summary,
            per_parameter=per_parameter,
            timing=timing,
            n_conditions=n_conditions,
            n_posterior_samples=n_posterior_samples,
            metric_names=list(metrics),
        )

    # Single-parameter case
    param_key = validation_data.param_keys[0]
    cond_rows = param_condition_rows[param_key]
    condition_df = pd.DataFrame(cond_rows)
    summary = aggregate_condition_rows(cond_rows)

    return ValidationResult(
        condition_metrics=condition_df,
        summary=summary,
        timing=timing,
        n_conditions=n_conditions,
        n_posterior_samples=n_posterior_samples,
        metric_names=list(metrics),
    )
