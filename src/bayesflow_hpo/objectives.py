"""Objective helpers and parameter normalization."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

KERAS_AVAILABLE = importlib.util.find_spec("keras") is not None

PARAM_COUNT_LOG_SCALE = 6.0
FAILED_TRIAL_CAL_ERROR = 1.0
FAILED_TRIAL_PARAM_SCORE = 1.5


def get_param_count(model: Any) -> int:
    """Count trainable parameters from a model-like object."""
    if not KERAS_AVAILABLE:
        raise ImportError("Keras is required for parameter counting")

    if hasattr(model, "count_params"):
        try:
            return int(model.count_params())
        except ValueError:
            return -1

    if hasattr(model, "trainable_weights"):
        if len(model.trainable_weights) == 0:
            return -1
        return int(sum(np.prod(w.shape) for w in model.trainable_weights))

    raise TypeError(f"Cannot count parameters for type: {type(model)}")


def normalize_param_count(param_count: int) -> float:
    """Map raw parameter count to ~0-1 log scale."""
    if param_count <= 0:
        return 0.0
    return float(np.log10(param_count) / PARAM_COUNT_LOG_SCALE)


def denormalize_param_count(normalized: float) -> int:
    """Invert normalized parameter count back to raw count."""
    if normalized <= 0:
        return 0
    return int(10 ** (normalized * PARAM_COUNT_LOG_SCALE))


def extract_objective_values(
    metrics: dict[str, Any],
    param_count: int,
    objective_metric: str = "calibration_error",
) -> tuple[float, float]:
    """Extract (objective_value, normalized_param_score)."""
    summary = metrics.get("summary", metrics)
    # Try the requested metric, fall back to mean_cal_error, then 1.0
    objective_value = float(
        summary.get(objective_metric, summary.get("mean_cal_error", 1.0))
    )
    param_score = normalize_param_count(param_count)
    return objective_value, param_score
