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
  scales differ -- ``correlation`` is bounded on [0, 1] and inverts as
  ``1 - value``, while ``log_gamma`` is an unbounded log-ratio and must be
  negated. Getting this wrong is silent: an un-flipped ``log_gamma`` makes the
  search prefer the most miscalibrated model it can find.
"""

from __future__ import annotations

import importlib.util
import logging
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

    A membership set cannot express this. ``correlation`` and ``contraction``
    live on [0, 1], so ``1 - value`` is both a direction flip and a sensible
    scale. ``log_gamma`` is an unbounded log-ratio where ``1 - value`` is
    meaningless -- the flip has to be a negation. Pairing the conversion with
    the metric, rather than deriving it from set membership, is what lets the
    two coexist.

    Attributes
    ----------
    higher_is_better
        Direction of the raw metric.
    to_minimize
        Raw value -> minimize-is-better objective value.
    worst_objective
        Objective value standing in for a metric missing from a validation
        summary. Already in minimize space, so it is *large*.
    """

    higher_is_better: bool
    to_minimize: Callable[[float], float]
    worst_objective: float


#: Direction and minimize-conversion for every metric with a known direction.
#:
#: A metric absent from this table is treated as lower-is-better and passed
#: through unchanged. That is correct for the error-style metrics
#: (``calibration_error``, ``nrmse``, ``rmse``, ``sbc_ks``, ``sbc_chi2``) and
#: is the historical behaviour.
#:
#: ``log_gamma`` is the entry that motivated replacing the old
#: ``HIGHER_IS_BETTER`` set. BayesFlow's ``calibration_log_gamma`` returns
#: log(gamma/gamma_null) and documents ``log_gamma < 0`` as rejecting the
#: hypothesis of uniform ranks at the 5% level -- so larger is better, and
#: minimizing it searches for the *most* miscalibrated model available. The
#: failure is silent: the study output looks entirely normal.
METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "correlation": MetricDirection(
        higher_is_better=True,
        to_minimize=lambda v: 1.0 - v,
        worst_objective=1.0,
    ),
    "contraction": MetricDirection(
        higher_is_better=True,
        to_minimize=lambda v: 1.0 - v,
        worst_objective=1.0,
    ),
    "log_gamma": MetricDirection(
        higher_is_better=True,
        # Negation, not ``1 - v``: log_gamma is unbounded in both directions.
        to_minimize=lambda v: -v,
        # A missing log_gamma must look terrible, and terrible here is a large
        # POSITIVE objective. FAILED_TRIAL_CAL_ERROR = 1.0 would correspond to
        # log_gamma = -1, an ordinary value that would not deter the sampler.
        worst_objective=1e3,
    ),
}

#: Backwards-compatible view of :data:`METRIC_DIRECTIONS`, kept because it was
#: public. Prefer the table, which also carries the conversion.
HIGHER_IS_BETTER = frozenset(
    name for name, d in METRIC_DIRECTIONS.items() if d.higher_is_better
)


def _metric_to_minimize(key: str, value: float) -> float:
    """Convert a raw metric value to a minimize-is-better scalar."""
    direction = METRIC_DIRECTIONS.get(key)
    if direction is None:
        return value
    return direction.to_minimize(value)


def worst_objective_value(key: str) -> float:
    """Minimize-space value standing in for a metric missing from a summary.

    Returned already converted, so callers must not pass it through
    :func:`_metric_to_minimize` again.
    """
    direction = METRIC_DIRECTIONS.get(key)
    if direction is None:
        return FAILED_TRIAL_CAL_ERROR
    return direction.worst_objective


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
    elif (
        objective_metric not in METRIC_DIRECTIONS
        and "calibration_error" in summary
    ):
        # Historical fallback, now restricted to metrics that share
        # calibration_error's direction and scale. Substituting it for a
        # missing `log_gamma` would be actively harmful: a good
        # calibration_error of 0.05 becomes an objective of -0.05, which under
        # minimization looks *better* than any real log_gamma, so a trial that
        # failed to report the metric would outrank every trial that did.
        objective_value = _metric_to_minimize(
            objective_metric, float(summary["calibration_error"])
        )
    else:
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
