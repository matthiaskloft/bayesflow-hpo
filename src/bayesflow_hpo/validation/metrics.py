"""Metric computation for fixed-validation datasets.

Delegates to the metric registry for actual computation. This module
provides:

- **Per-condition dispatch**: runs all registered metrics on a single
  condition batch and returns a flat dict.
- **Cross-condition aggregation**: reduces numeric metric values across
  conditions with scalar or per-metric settings, skipping NaNs and identifiers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bayesflow_hpo.validation.registry import MetricFn, canonical_metric_name

Aggregate = str | Mapping[str, str]


class AggregationError(ValueError):
    """A reduction cannot score its inputs and must not become a trial fallback."""


def normalize_aggregate(aggregate: Aggregate) -> Aggregate:
    """Validate reductions and canonicalize metric aliases in a copied mapping."""
    choices = {"mean", "worst", "geometric"}
    if isinstance(aggregate, str):
        if aggregate not in choices:
            raise ValueError(
                f"Unknown aggregate {aggregate!r}; expected {sorted(choices)}."
            )
        return aggregate
    if not isinstance(aggregate, Mapping):
        raise TypeError(
            "aggregate must be a reduction name or a metric-to-reduction mapping."
        )
    normalized: dict[str, str] = {}
    for key, reduction in aggregate.items():
        if not isinstance(key, str):
            raise TypeError("aggregate metric names must be strings.")
        if not isinstance(reduction, str) or reduction not in choices:
            raise ValueError(f"Unknown aggregate {reduction!r} for metric {key!r}.")
        name = canonical_metric_name(key)
        if name in normalized and normalized[name] != reduction:
            raise ValueError(f"Conflicting aggregate settings for metric {name!r}.")
        normalized[name] = reduction
    return normalized


def reduce_metric(values: Sequence[float], key: str, reduction: str) -> float:
    """Reduce raw metric values, omitting NaNs and retaining infinities.

    Geometric reduction uses ``exp(mean(log(x)))`` as documented by
    SciPy's ``scipy.stats.gmean``. It requires strictly positive values;
    no epsilon is added because that would change the metric's scale.
    Arithmetic means follow NumPy's ``nanmean`` convention. Empty or
    all-NaN inputs return NaN. Worst follows the registered raw direction.

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gmean.html
    https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
    """
    from bayesflow_hpo.objectives import _direction_for

    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan")
    if reduction == "geometric":
        if np.any(vals <= 0):
            raise AggregationError(
                f"Geometric aggregation for metric {key!r} requires strictly "
                "positive values; use 'mean' or 'worst' for values <= 0."
            )
        return float(np.exp(np.mean(np.log(vals))))
    if reduction == "worst":
        direction = _direction_for(canonical_metric_name(key))
        reducer = np.min if direction and direction.higher_is_better else np.max
        return float(reducer(vals))
    return float(np.mean(vals))


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


def aggregate_condition_rows(
    condition_rows: list[dict[str, Any]], aggregate: Aggregate = "mean",
) -> dict[str, float]:
    """Reduce numeric values across conditions using scalar or per-key settings.

    Non-numeric and identifier columns (``id_cond``, ``n_sims``) are skipped.
    Unspecified mapping keys use the arithmetic mean. See ``reduce_metric``
    for geometric-domain and missing-value rules.
    """
    aggregate = normalize_aggregate(aggregate)
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
        reduction = (
            aggregate if isinstance(aggregate, str) else aggregate.get(key, "mean")
        )
        summary[key] = reduce_metric(vals, key, reduction)

    return summary
