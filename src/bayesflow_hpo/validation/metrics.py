"""Metric computation for fixed-validation datasets.

Delegates to the metric registry for actual computation. This module provides
the per-condition dispatch and cross-condition aggregation logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bayesflow_hpo.validation.registry import MetricFn


def compute_condition_metrics(
    draws: np.ndarray,
    true_values: np.ndarray,
    cond_id: int,
    metric_fns: dict[str, MetricFn],
) -> dict[str, Any]:
    """Run all requested metrics on one condition batch.

    Parameters
    ----------
    draws
        Posterior samples, shape ``(n_sims, n_samples)``.
    true_values
        Ground truth, shape ``(n_sims,)``.
    cond_id
        Integer condition identifier.
    metric_fns
        ``{name: fn}`` mapping from
        :func:`~bayesflow_hpo.validation.registry.resolve_metrics`.

    Returns
    -------
    dict
        Flat dict with ``"id_cond"`` plus all metric output keys.
    """
    row: dict[str, Any] = {"id_cond": cond_id, "n_sims": len(true_values)}

    for _name, fn in metric_fns.items():
        row.update(fn(draws, true_values))

    return row


def aggregate_condition_rows(condition_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average numeric values across conditions.

    Non-numeric and identifier columns (``id_cond``, ``n_sims``) are skipped.
    """
    if not condition_rows:
        return {}

    skip_keys = {"id_cond", "n_sims"}
    numeric_keys = [
        k for k in condition_rows[0]
        if k not in skip_keys
        and isinstance(condition_rows[0][k], (int, float))
    ]

    summary: dict[str, float] = {}
    for key in numeric_keys:
        vals = [
            row[key] for row in condition_rows
            if not np.isnan(row.get(key, float("nan")))
        ]
        summary[key] = float(np.mean(vals)) if vals else float("nan")

    return summary
