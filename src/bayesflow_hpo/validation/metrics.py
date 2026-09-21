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

from bayesflow_hpo.validation.registry import (
    MetricFn,
    canonical_metric_name,
    is_diagnostic_metric,
    output_keys_for,
    producer_for_key,
)

Aggregate = str | Mapping[str, str]

#: Reductions accepted by ``aggregate``.
AGGREGATIONS = ("mean", "worst", "geometric")


class AggregationError(ValueError):
    """Base class for a reduction that cannot score its inputs.

    Split into two subclasses because the trial lifecycle has to tell them
    apart, and only one of them is a property of the study. Catching this
    base class is right for a one-shot caller (``validate_once``,
    ``check_pipeline``) that only wants the message preserved.
    """


class AggregationConfigError(AggregationError):
    """The requested reduction is invalid for the configured metrics.

    Every trial in the study would hit it identically, so the objective
    re-raises it rather than recording a fallback score for the whole budget.
    """


class AggregationDomainError(AggregationError):
    """A trial's own values fall outside the reduction's domain.

    Deliberately NOT a configuration error. ``geometric`` rejects values
    ``<= 0``, and ``correlation`` and ``contraction`` -- both in
    ``DEFAULT_METRICS`` -- are legitimately non-positive for an undertrained
    approximator (a negative Pearson correlation; ``1 - var_post/var_prior``
    below zero when the posterior is wider than the prior). Re-raising that
    out of the objective would let one poor trial terminate an otherwise
    healthy study, so it falls through to the ordinary failed-trial path.
    """


def require_pipeline_aggregate(aggregate: Aggregate, *, has_validate_fn: bool) -> None:
    """Reject a non-default *aggregate* alongside a custom ``validate_fn``.

    A custom ``validate_fn`` returns scores that are already reduced, so
    there is no grid left for this package to reduce. Shared by every public
    boundary that accepts both, so the rule and its message stay in one place.

    Parameters
    ----------
    aggregate
        A normalized reduction, as returned by :func:`normalize_aggregate`.
    has_validate_fn
        Whether the caller supplied a custom ``validate_fn``.

    Raises
    ------
    ValueError
        If *aggregate* is anything but the default mean.
    """
    if has_validate_fn and aggregate not in ("mean", {}):
        raise ValueError("aggregate requires the built-in validation pipeline.")


def _resolve_output_key(key: str) -> str:
    """Resolve *key* to the summary column it names, or explain why it cannot.

    A mapping key has to match a column the pipeline actually emits, because
    the reduction is applied by looking that column up. An unmatched key is
    not inert: it silently leaves the metric on the arithmetic mean, which is
    the averaging this option exists to escape, with nothing to show that the
    setting never took effect.

    Raises
    ------
    ValueError
        If *key* is not a summary key. A metric group -- a multi-output
        metric such as ``coverage``, whose own name is never a column -- is
        reported with the outputs the caller can choose from instead.
    """
    name = canonical_metric_name(key)
    producer = producer_for_key(name)
    if producer is None:
        # `output_keys_for` answers `(name,)` for anything it does not know,
        # so it cannot be asked first: every misspelling would look like a
        # single-output metric that happens to emit itself.
        raise ValueError(
            f"aggregate key {key!r} is not a known metric output. Register "
            "the metric first, or check the spelling against the validation "
            "summary."
        )
    outputs = output_keys_for(producer)
    if name in outputs:
        return name
    joined = ", ".join(repr(k) for k in outputs)
    raise ValueError(
        f"aggregate key {key!r} names the metric group {producer!r}, which "
        f"emits no summary column of its own. Use one of: {joined}."
    )


def normalize_aggregate(aggregate: Aggregate) -> Aggregate:
    """Validate reductions and metric keys, returning a copied mapping.

    Mapping keys are resolved through :func:`_resolve_output_key`, so an
    alias is canonicalized and anything that is not an emitted summary
    column is rejected rather than silently ignored.
    """
    if isinstance(aggregate, str):
        if aggregate not in AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregate {aggregate!r}; expected {sorted(AGGREGATIONS)}."
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
        if not isinstance(reduction, str) or reduction not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregate {reduction!r} for metric {key!r}.")
        name = _resolve_output_key(key)
        if name in normalized and normalized[name] != reduction:
            raise ValueError(f"Conflicting aggregate settings for metric {name!r}.")
        normalized[name] = reduction
    return normalized


def worst_reducer(key: str) -> Any:
    """Return the NumPy reducer that picks the worst value of *key*, or None.

    Resolution is in two steps, because ``_direction_for`` returning ``None``
    means two different things. For ``mae`` it means only that the directions
    table carries no entry: the metric is objective-eligible and
    lower-is-better like every other error, so the maximum is its worst
    condition. For ``bias`` or ``coverage_90`` it means the quantity is not
    monotone in quality at all -- signed bias is optimal at zero and coverage
    at its nominal level, so 0.99 is no better a worst case than 0.55 -- and
    there the registry says so independently by marking the producing metric
    ``kind="diagnostic"``.

    Returns
    -------
    numpy.ufunc or None
        ``np.min`` for a registered higher-is-better key, ``np.max`` for any
        other scorable key, and ``None`` when the key has no worst case.
    """
    from bayesflow_hpo.objectives import _direction_for

    name = canonical_metric_name(key)
    direction = _direction_for(name)
    if direction is not None:
        return np.min if direction.higher_is_better else np.max
    producer = producer_for_key(name)
    if producer is not None and is_diagnostic_metric(producer):
        return None
    return np.max


def resolve_reduction(aggregate: Aggregate, key: str) -> tuple[str, bool]:
    """Return the reduction for *key* and whether it was named explicitly.

    A scalar applies to every key and names none of them, so it is never
    explicit; a mapping entry is. :func:`reduce_metric` treats the two
    differently for ``"worst"``.
    """
    if isinstance(aggregate, str):
        return aggregate, False
    reduction = aggregate.get(key)
    if reduction is None:
        return "mean", False
    return reduction, True


def reduce_metric(
    values: Sequence[float], key: str, reduction: str, *, explicit: bool = False,
) -> float:
    """Reduce raw metric values, omitting NaNs and retaining infinities.

    Geometric reduction uses ``exp(mean(log(x)))`` as documented by
    SciPy's ``scipy.stats.gmean``. It requires strictly positive values;
    no epsilon is added because that would change the metric's scale.
    Arithmetic means follow NumPy's ``nanmean`` convention. Empty or
    all-NaN inputs return NaN. Worst follows the registered raw direction.

    Parameters
    ----------
    values
        Raw per-cell values for one summary key.
    key
        The summary key, used to resolve direction for ``"worst"``.
    reduction
        One of :data:`AGGREGATIONS`.
    explicit
        Whether the caller named *key* in an ``aggregate`` mapping rather
        than reaching it through a scalar setting. Only an explicit request
        for ``"worst"`` on a diagnostic-only key raises: the caller asserted
        a worst case that does not exist and has to be told. A scalar sweeps
        up every reported diagnostic as a side effect, so there the key
        keeps the arithmetic mean -- see :func:`worst_reducer`.

    Raises
    ------
    AggregationConfigError
        On an explicit ``"worst"`` for a key with no defined direction.
    AggregationDomainError
        On a ``"geometric"`` reduction over a non-positive value.

    See Also
    --------
    worst_reducer : How ``"worst"`` resolves a key's direction.

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gmean.html
    https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan")
    if reduction == "geometric":
        if np.any(vals <= 0):
            raise AggregationDomainError(
                f"Geometric aggregation for metric {key!r} requires strictly "
                "positive values; use 'mean' or 'worst' for values <= 0."
            )
        return float(np.exp(np.mean(np.log(vals))))
    if reduction == "worst":
        reducer = worst_reducer(key)
        if reducer is None:
            if explicit:
                raise AggregationConfigError(
                    f"Metric {key!r} is diagnostic-only and has no worst "
                    "case: it is optimal at a point rather than at an "
                    "extreme, so no condition is its worst. Use 'mean' or "
                    "'geometric' for it, or name a metric that is eligible "
                    "as an objective."
                )
            # Reached by a scalar `aggregate="worst"`, which sweeps up every
            # reported key and not only the scored ones. Taking the max here
            # would report the BEST condition for a higher-is-better key and
            # a meaningless extreme for a zero-optimum one, so the documented
            # rule is that a scalar reduces the scorable metrics and leaves
            # the diagnostics on the mean they have always had.
            return float(np.mean(vals))
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
    for geometric-domain, direction and missing-value rules.
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
        reduction, explicit = resolve_reduction(aggregate, key)
        summary[key] = reduce_metric(vals, key, reduction, explicit=explicit)

    return summary
