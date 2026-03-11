"""Objective helpers and inference-time cost normalization."""

from __future__ import annotations

import importlib.util
import logging
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
# When sim_time_per_sim is unavailable, the fallback is raw inference
# seconds, which can be in the hundreds — 1e6 safely dominates.
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
    """
    # Auto-tighten min_count when user specified a smaller max_count
    # but left min_count at its default.
    if min_count == MIN_PARAM_COUNT and max_count < MAX_PARAM_COUNT:
        min_count = max(1, max_count // 100)
    if param_count <= 0:
        return 1.0  # worst score — signals broken or unbuilt model
    if min_count <= 0:
        min_count = 1
    if max_count <= min_count:
        return 0.0
    clamped = max(min(param_count, max_count), min_count)
    return float(np.log10(clamped / min_count) / np.log10(max_count / min_count))


def denormalize_param_count(
    normalized: float,
    min_count: int = MIN_PARAM_COUNT,
    max_count: int = MAX_PARAM_COUNT,
) -> int:
    """Invert :func:`normalize_param_count` back to a raw count."""
    if normalized <= 0:
        return 0
    if min_count <= 0:
        min_count = 1
    log_range = np.log10(max_count / min_count)
    return int(min_count * 10 ** (normalized * log_range))


# Metrics where higher is better — the objective value is ``1 - metric``.
HIGHER_IS_BETTER = {"correlation"}


def _metric_to_minimize(key: str, value: float) -> float:
    """Convert a raw metric value to a minimize-is-better scalar."""
    if key in HIGHER_IS_BETTER:
        return 1.0 - value
    return value


def compute_inference_time_ratio(
    inference_time: float,
    sim_time_per_sim: float | None,
    n_sims: int,
) -> float:
    """Compute the inference-to-simulation time ratio.

    Returns ``inference_time / (sim_time_per_sim * n_sims)``.  When
    simulation timing is unavailable, falls back to raw inference
    seconds so the objective is still meaningful (just not normalized).

    Parameters
    ----------
    inference_time
        Total wall-clock inference seconds across all validation sims.
    sim_time_per_sim
        Average seconds to simulate one observation (from
        :class:`~bayesflow_hpo.validation.data.ValidationDataset`).
    n_sims
        Total number of simulations that were inferred on.
    """
    if sim_time_per_sim is not None and sim_time_per_sim > 0:
        total_sim_time = sim_time_per_sim * max(n_sims, 1)
        return inference_time / total_sim_time
    # Fallback: raw seconds (still minimize-is-better).
    return inference_time


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
        ``inference_time_ratio`` or ``normalized_param_count``.
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
    default = 0.0 if objective_metric in HIGHER_IS_BETTER else 1.0
    raw_value = float(
        summary.get(
            objective_metric,
            summary.get("calibration_error", default),
        )
    )
    objective_value = _metric_to_minimize(objective_metric, raw_value)
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
        ``inference_time_ratio`` or ``normalized_param_count``.
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
        # For higher-is-better metrics, default to 0.0 so that
        # _metric_to_minimize inverts it to 1.0 (worst).
        default = 0.0 if key in HIGHER_IS_BETTER else 1.0
        val = float(summary.get(key, default))
        raw_values.append(_metric_to_minimize(key, val))

    if objective_mode == "pareto":
        return tuple(raw_values) + (cost_score,)

    # "mean" mode — arithmetic mean of all metric values
    mean_val = float(np.mean(raw_values))
    return (mean_val, cost_score)
