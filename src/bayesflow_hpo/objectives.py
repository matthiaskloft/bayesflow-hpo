"""Objective helpers and inference-time cost computation.

Utility functions for mapping raw validation metrics and model costs into
minimize-is-better objective values for Optuna.

Key concepts:

- **Parameter count normalization**: Maps raw param counts to [0, 1] via
  log-linear scaling so the Pareto front is meaningful when accuracy
  metrics are also in [0, 1].
- **Inference time per dataset**: Reports inference cost as seconds per
  dataset (averaged over conditions), giving an interpretable measure of
  how long one inference call takes.
- **Direction conversion**: every objective is mapped to minimize-is-better
  through :data:`METRIC_DIRECTIONS`. The conversion is per-metric because the
  scales differ -- ``correlation`` is bounded on [-1, 1] and inverts as
  ``1 - value``, while ``log_gamma`` is an unbounded log-ratio and must be
  negated. Getting this wrong is silent: an un-flipped ``log_gamma`` makes the
  search prefer the most miscalibrated model it can find.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

KERAS_AVAILABLE = importlib.util.find_spec("keras") is not None

# Default budget boundaries for parameter-count rejection gate.
MIN_PARAM_COUNT = 1_000
MAX_PARAM_COUNT = 1_000_000

# Penalty values returned for failed / budget-rejected trials.
FAILED_TRIAL_CAL_ERROR = 1.0
# Cost penalty for failed trials.  Must be large enough to dominate
# any legitimate cost score so the sampler avoids these regions.
# 1e6 seconds (~278 hours) safely dominates any real inference time.
FAILED_TRIAL_COST = 1e6


def get_param_count(model: Any) -> int:
    """Count trainable parameters from a model-like object."""
    if not KERAS_AVAILABLE:
        raise ImportError("Keras is required for parameter counting")

    if hasattr(model, "count_params"):
        try:
            return int(model.count_params())
        except ValueError as exc:
            logger.warning("count_params() failed — model may not be built")
            raise ValueError("Model not built: count_params() failed") from exc

    if hasattr(model, "trainable_weights"):
        if len(model.trainable_weights) == 0:
            logger.warning("Model has no trainable weights — may not be built")
            raise ValueError("Model not built: no trainable weights")
        return int(sum(np.prod(w.shape) for w in model.trainable_weights))

    raise TypeError(f"Cannot count parameters for type: {type(model)}")


def _resolve_min_count(min_count: int, max_count: int) -> int:
    """Auto-tighten ``min_count`` to match ``normalize_param_count``.

    When ``max_count`` is below the default ``MAX_PARAM_COUNT`` and
    ``min_count`` is still at the default, the lower bound is
    automatically tightened to ``max_count / 100`` so the normalized
    values spread across the full [0, 1] range. Also clamps non-positive
    values to ``1``.

    Parameters
    ----------
    min_count
        Lower reference as passed by the caller.
    max_count
        Upper reference as passed by the caller.

    Returns
    -------
    int
        The resolved ``min_count`` to use.
    """
    # Auto-tighten min_count when user specified a smaller max_count
    # but left min_count at its default.
    if min_count == MIN_PARAM_COUNT and max_count < MAX_PARAM_COUNT:
        min_count = max(1, max_count // 100)
    if min_count <= 0:
        min_count = 1
    return min_count


def normalize_param_count(
    param_count: int,
    min_count: int = MIN_PARAM_COUNT,
    max_count: int = MAX_PARAM_COUNT,
) -> float:
    """Map raw parameter count to 0--1 via log-linear scaling.

    Uses ``log10(count / min) / log10(max / min)`` so that *min_count*
    maps to 0.0 and *max_count* maps to 1.0.

    When ``max_count`` is below the default ``MAX_PARAM_COUNT`` and
    ``min_count`` is still at the default, the lower bound is
    automatically tightened to ``max_count / 100`` so the normalized
    values spread across the full [0, 1] range.

    Parameters
    ----------
    param_count
        Raw trainable parameter count.
    min_count
        Lower reference (maps to 0.0).  Default ``1_000``.
    max_count
        Upper reference (maps to 1.0).  Default ``1_000_000``.

    Raises
    ------
    ValueError
        If *max_count* <= *min_count* after auto-tightening.
    """
    min_count = _resolve_min_count(min_count, max_count)
    if param_count <= 0:
        return 1.0  # worst score — signals broken or unbuilt model
    if max_count <= min_count:
        raise ValueError(
            f"max_count ({max_count}) must be greater than min_count ({min_count})"
        )
    clamped = max(min(param_count, max_count), min_count)
    return float(np.log10(clamped / min_count) / np.log10(max_count / min_count))


def denormalize_param_count(
    normalized: float,
    min_count: int = MIN_PARAM_COUNT,
    max_count: int = MAX_PARAM_COUNT,
) -> int:
    """Invert :func:`normalize_param_count` back to a raw count.

    Raises
    ------
    ValueError
        If *max_count* <= *min_count*.
    """
    min_count = _resolve_min_count(min_count, max_count)
    if normalized <= 0:
        return 0
    if max_count <= min_count:
        raise ValueError(
            f"max_count ({max_count}) must be greater than min_count ({min_count})"
        )
    log_range = np.log10(max_count / min_count)
    return int(min_count * 10 ** (normalized * log_range))


@dataclass(frozen=True)
class MetricDirection:
    """How one metric maps onto Optuna's minimize-is-better convention.

    A membership set cannot express this. ``contraction`` lives on [0, 1] and
    ``correlation`` on [-1, 1], so ``1 - value`` is a direction flip that
    lands on a sensible scale for both -- but their *worst* values differ
    (1.0 and 2.0), which a single shared rule cannot record. ``log_gamma`` is
    an unbounded log-ratio where ``1 - value`` is meaningless -- the flip has
    to be a negation. Pairing the conversion with the metric, rather than
    deriving it from set membership, is what lets all three coexist.

    Attributes
    ----------
    higher_is_better
        Direction of the raw metric.
    to_minimize
        Raw value -> minimize-is-better objective value.
    worst_objective
        Objective value standing in for a metric missing from a validation
        summary. Already in minimize space, so it is *large*, and ``math.inf``
        for any metric that is unbounded in the losing direction -- only a
        bounded metric has a finite worst case.

        Note this makes a missing metric worse than any reported value *on
        that objective*. In multi-objective mode it does not by itself keep
        such a trial off the Pareto front, since a trial can stay
        non-dominated by excelling on a different objective.
    """

    higher_is_better: bool
    to_minimize: Callable[[float], float]
    worst_raw: float

    @property
    def worst_objective(self) -> float:
        """The worst value in minimize space, derived from :attr:`worst_raw`.

        Derived rather than stored so the two spaces cannot drift apart. They
        did: the penalty injected for a missing metric is a *raw* value, and
        it was a hardcoded 1.0 for every metric, so a missing ``log_gamma``
        became the objective ``-1.0`` -- which beats a genuinely reported
        ``log_gamma`` of 0.5 (objective ``-0.5``). A trial that failed to
        report the metric outranked one that reported a good value.
        """
        return self.to_minimize(self.worst_raw)


#: Direction and minimize-conversion for every metric with a known direction.
#:
#: A metric absent from this table is treated as lower-is-better and passed
#: through unchanged, which is correct for the remaining error-style metrics
#: (``calibration_error``, ``nrmse``, ``rmse``) and is the historical
#: behaviour. Custom metrics registered by a caller land here too.
#:
#: ``log_gamma`` is the entry that motivated replacing the old
#: ``HIGHER_IS_BETTER`` set. Its direction is not inferred: BayesFlow's
#: ``bayesflow.diagnostics.metrics.calibration_log_gamma`` documents the
#: statistic as log(gamma/gamma_null), where gamma_null is the 5th percentile
#: of the null distribution under uniformity of ranks, and states that
#: "log_gamma < 0 implies a rejection of the hypothesis of uniform ranks at
#: the 5% level". Larger is therefore better, and minimizing it searches for
#: the *most* miscalibrated model available. The failure is silent: the study
#: output looks entirely normal.
#:
#: The statistic is from Modrak, M., Moon, A. H., Kim, S., Buerkner, P.,
#: Huurre, N., Faltejskova, K., Gelman, A., & Vehtari, A. (2025).
#: Simulation-based calibration checking for Bayesian computation: The choice
#: of test quantities shapes sensitivity. *Bayesian Analysis, 20*(2),
#: 461-488. https://doi.org/10.1214/23-BA1404
METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "correlation": MetricDirection(
        higher_is_better=True,
        to_minimize=lambda v: 1.0 - v,
        # Pearson correlation runs [-1, 1], not [0, 1]. Using 0.0 here would
        # let a missing value tie a reported correlation of 0 and *beat* every
        # negative one -- a reported -0.5 maps to 1.5, worse than the penalty.
        worst_raw=-1.0,
    ),
    "contraction": MetricDirection(
        higher_is_better=True,
        to_minimize=lambda v: 1.0 - v,
        # Contraction is a variance ratio: 0 = no narrowing, its true worst.
        worst_raw=0.0,
    ),
    "log_gamma": MetricDirection(
        higher_is_better=True,
        # Negation, not ``1 - v``: log_gamma is unbounded in both directions.
        to_minimize=lambda v: -v,
        # Infinite, because log_gamma is unbounded below and therefore no
        # finite constant is provably the worst value. A finite penalty can be
        # *beaten* by a genuinely terrible trial -- with a penalty of 1e3, a
        # real log_gamma of -5000 would score worse than a missing one, so
        # failing to report the metric would look better than reporting a
        # catastrophic value. FAILED_TRIAL_CAL_ERROR = 1.0 is worse still: as
        # a log_gamma objective it means log_gamma = -1, an ordinary value.
        worst_raw=-math.inf,
    ),
    # The error-style metrics. Lower is better and they pass through
    # unchanged, so registering them changes no conversion -- but it records
    # the unit scale explicitly, which is what lets the training-loss fallback
    # tell "this metric is on a [0, 1] lower-is-better scale, so a clamped
    # loss is a sensible proxy" apart from "nothing is known about this
    # metric". Without the distinction an unregistered custom metric silently
    # inherited the proxy and a failed trial could outrank a valid one.
    "calibration_error": MetricDirection(
        higher_is_better=False,
        to_minimize=lambda v: v,
        # ECE is a mean absolute deviation between two probabilities.
        worst_raw=1.0,
    ),
    "nrmse": MetricDirection(
        higher_is_better=False,
        to_minimize=lambda v: v,
        # Range-normalized, so 1.0 is the historical penalty and the scale the
        # training-loss proxy assumes. It can exceed 1 in principle; the value
        # is kept for continuity with FAILED_TRIAL_CAL_ERROR.
        worst_raw=1.0,
    ),
    "rmse": MetricDirection(
        higher_is_better=False,
        to_minimize=lambda v: v,
        worst_raw=1.0,
    ),
    # The SBC tests are lower-is-better and pass through unchanged, but they
    # are listed explicitly so a missing value takes a defined penalty rather
    # than silently borrowing calibration_error's.
    "sbc_ks": MetricDirection(
        higher_is_better=False,
        to_minimize=lambda v: v,
        # A KS statistic is a sup of a CDF difference, so it is bounded by 1.
        worst_raw=1.0,
    ),
    "sbc_chi2": MetricDirection(
        higher_is_better=False,
        to_minimize=lambda v: v,
        # Unlike KS, the raw chi-squared statistic is unbounded above.
        worst_raw=math.inf,
    ),
}

#: Version of the objective-value encoding written into a study.
#:
#: Bumped when a change alters the NUMBERS stored for a trial rather than the
#: code that computes them. Version 2 negates ``log_gamma`` so that every
#: objective is minimize-is-better; version 1 stored it raw. Trials from the
#: two cannot be compared, so a resumed study has to be checked rather than
#: assumed compatible -- see :func:`bayesflow_hpo.api.optimize`.
OBJECTIVE_ENCODING_VERSION = 2

#: Metrics whose stored objective values actually changed at version 2.
#:
#: Recorded explicitly rather than derived from ``higher_is_better``, because
#: that predicate is wrong here. Before this change the conversion was
#: ``1 - value`` for the two names in the old ``HIGHER_IS_BETTER`` set and
#: pass-through for everything else, so ``correlation`` and ``contraction``
#: convert exactly as they always did and their stored values are unchanged.
#: Only ``log_gamma`` moved -- from pass-through to negation. Using
#: ``higher_is_better`` would refuse a resumable ``contraction`` study, which
#: is a real objective metric, on the strength of a property that says nothing
#: about whether its numbers moved.
ENCODING_CHANGED_AT_V2: frozenset[str] = frozenset({"log_gamma"})

#: Mutable set of higher-is-better metric names, kept as a live extension
#: point rather than a derived view.
#:
#: This was public API whose *contents* drove :func:`_metric_to_minimize`, so
#: a consumer could register a custom higher-is-better metric with
#: ``HIGHER_IS_BETTER.add("my_metric")``. Making it a derived frozenset would
#: have silently removed that, so names added here still take the historical
#: ``1 - value`` conversion. :data:`METRIC_DIRECTIONS` takes precedence, and
#: is the right place to register anything whose scale is not [0, 1] --
#: see :func:`register_metric_direction`.
HIGHER_IS_BETTER: set[str] = {
    name for name, d in METRIC_DIRECTIONS.items() if d.higher_is_better
}


def register_metric_direction(
    name: str,
    *,
    higher_is_better: bool,
    worst_raw: float,
    to_minimize: Callable[[float], float] | None = None,
) -> None:
    """Register the direction of a custom metric.

    Preferred over mutating :data:`HIGHER_IS_BETTER`, because it also fixes
    the conversion and the worst-case value. ``to_minimize`` defaults to
    ``1 - value`` for higher-is-better metrics and the identity otherwise,
    which is only right for a metric bounded on [0, 1]; pass an explicit
    conversion (typically negation) for an unbounded one.
    """
    if to_minimize is None:
        to_minimize = (lambda v: 1.0 - v) if higher_is_better else (lambda v: v)
    METRIC_DIRECTIONS[name] = MetricDirection(
        higher_is_better=higher_is_better,
        to_minimize=to_minimize,
        worst_raw=worst_raw,
    )
    if higher_is_better:
        HIGHER_IS_BETTER.add(name)
    else:
        HIGHER_IS_BETTER.discard(name)


def _direction_for(key: str) -> MetricDirection | None:
    """Resolve a metric's direction, honouring both compatibility mutations.

    :data:`HIGHER_IS_BETTER` used to control conversion by its contents, so a
    consumer could both *add* a custom higher-is-better metric and *remove* a
    built-in to make it pass through. Consulting :data:`METRIC_DIRECTIONS`
    unconditionally would have supported only the first: after
    ``HIGHER_IS_BETTER.discard("contraction")`` the conversion would still
    invert. A discarded higher-is-better entry is therefore treated as
    removed.
    """
    direction = METRIC_DIRECTIONS.get(key)
    if direction is not None:
        if direction.higher_is_better and key not in HIGHER_IS_BETTER:
            # Removed from the legacy set: honour the removal.
            return None
        if not direction.higher_is_better and key in HIGHER_IS_BETTER:
            # ADDED to the legacy set for a metric the table calls
            # lower-is-better. Before `calibration_error`, `nrmse` and `rmse`
            # were registered explicitly, adding one of those names selected
            # the historical `1 - value` conversion; registering them must not
            # quietly take that away from a consumer who reinterprets one.
            return MetricDirection(
                higher_is_better=True,
                to_minimize=lambda v: 1.0 - v,
                worst_raw=0.0,
            )
        return direction
    if key in HIGHER_IS_BETTER:
        # Added to the legacy set only: historical `1 - value` behaviour.
        return MetricDirection(
            higher_is_better=True,
            to_minimize=lambda v: 1.0 - v,
            worst_raw=0.0,
        )
    return None


def _metric_to_minimize(key: str, value: float) -> float:
    """Convert a raw metric value to a minimize-is-better scalar."""
    direction = _direction_for(key)
    if direction is None:
        return value
    return direction.to_minimize(value)


def worst_raw_value(key: str) -> float:
    """Raw-space value representing the worst case for *key*.

    This is what a *penalty injected before conversion* must use. Injecting a
    minimize-space value there is the bug this function exists to prevent: a
    flat raw penalty of 1.0 for ``log_gamma`` converts to ``-1.0``, which
    beats a genuinely reported ``log_gamma`` of 0.5.
    """
    direction = _direction_for(key)
    if direction is not None:
        return direction.worst_raw
    return FAILED_TRIAL_CAL_ERROR


def worst_objective_value(key: str) -> float:
    """Minimize-space value standing in for a metric missing from a summary.

    Returned already converted, so callers must not pass it through
    :func:`_metric_to_minimize` again.
    """
    return _metric_to_minimize(key, worst_raw_value(key))


def compute_inference_time_per_dataset(
    inference_time: float,
    n_datasets: int,
) -> float:
    """Compute average inference time in seconds per dataset.

    Parameters
    ----------
    inference_time
        Total pure inference seconds across all validation datasets.
    n_datasets
        Number of datasets (conditions) inferred on.

    Returns
    -------
    float
        Seconds per dataset (``inference_time / max(n_datasets, 1)``).
    """
    return inference_time / max(n_datasets, 1)


def extract_objective_values(
    metrics: dict[str, Any],
    cost_score: float,
    objective_metric: str = "calibration_error",
) -> tuple[float, float]:
    """Extract ``(objective_value, cost_score)``.

    Parameters
    ----------
    metrics
        Nested dict with at least ``{"summary": {objective_metric: value}}``.
    cost_score
        Pre-computed cost objective (minimize-is-better).  Typically
        ``inference_time_s`` (seconds per dataset) or
        ``normalized_param_count``.
    objective_metric
        Key to look up inside the summary dict.
    """
    summary = metrics.get("summary", metrics)
    if objective_metric not in summary:
        logger.warning(
            "Metric key %r not found in validation summary. "
            "Available keys: %s. Falling back to 'calibration_error' or 1.0.",
            objective_metric, list(summary.keys()),
        )
    if objective_metric in summary:
        objective_value = _metric_to_minimize(
            objective_metric, float(summary[objective_metric])
        )
    else:
        # No cross-metric substitution. This used to fall back to
        # `calibration_error`, which assumes the missing metric shares its
        # direction AND its scale -- and absence from METRIC_DIRECTIONS
        # establishes neither. A missing `custom_rmse` would take
        # calibration_error's 0.05 while a genuinely reported custom RMSE can
        # be 100, so the missing metric wins. For `log_gamma` it was worse
        # still: 0.05 negates to -0.05, beating every real value.
        objective_value = worst_objective_value(objective_metric)
    return objective_value, cost_score


def extract_multi_objective_values(
    metrics: dict[str, Any],
    cost_score: float,
    objective_metrics: list[str],
    objective_mode: str = "mean",
) -> tuple[float, ...]:
    """Extract objective values for multi-metric optimization.

    Parameters
    ----------
    metrics
        Nested dict with at least ``{"summary": {...}}``.
    cost_score
        Pre-computed cost objective (minimize-is-better).  Typically
        ``inference_time_s`` (seconds per dataset) or
        ``normalized_param_count``.
    objective_metrics
        List of metric keys to optimize.
    objective_mode
        ``"mean"`` — return ``(mean_of_metrics, cost_score)`` (2 values).
        ``"pareto"`` — return ``(*metric_values, cost_score)``
        (one value per metric + cost).
    """
    if objective_mode not in ("mean", "pareto"):
        raise ValueError(
            f"Unknown objective_mode: {objective_mode!r}. "
            f"Expected 'mean' or 'pareto'."
        )

    summary = metrics.get("summary", metrics)

    raw_values: list[float] = []
    for key in objective_metrics:
        if key not in summary:
            logger.warning(
                "Metric key %r not found in validation summary. "
                "Available keys: %s. Using worst-case default.",
                key, list(summary.keys()),
            )
        if key in summary:
            raw_values.append(_metric_to_minimize(key, float(summary[key])))
        else:
            # Already in minimize space -- do not convert it again.
            raw_values.append(worst_objective_value(key))

    if objective_mode == "pareto":
        return tuple(raw_values) + (cost_score,)

    # "mean" mode — arithmetic mean of all metric values
    mean_val = float(np.mean(raw_values))
    return (mean_val, cost_score)


def mean_objective_score(values: list[float] | tuple[float, ...]) -> float:
    """Reduce a multi-objective values tuple to a single ranking score.

    Averages all elements except the last (assumed to be the cost
    score), matching the ``(*metric_values, cost_score)`` shape
    returned by :func:`extract_multi_objective_values` in both
    ``"pareto"`` mode (multiple metrics + cost) and ``"mean"`` mode
    (single metric + cost). Falls back to the sole element when only
    one value is given.

    Parameters
    ----------
    values
        Objective values tuple, e.g. ``(metric_1, ..., metric_n,
        cost_score)``.

    Returns
    -------
    float
        Mean of all elements except the last, or the single element
        when ``len(values) == 1``.
    """
    if len(values) > 1:
        return float(np.mean(values[:-1]))
    return float(values[0])
