"""Utility helpers used across bayesflow_hpo."""

from __future__ import annotations

import numpy as np


def loguniform_int(
    low: int,
    high: int,
    alpha: float = 1.0,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> int:
    """Sample an integer from a generalized log-uniform distribution."""
    if low <= 0:
        raise ValueError(f"low must be positive, got {low}")
    if high < low:
        raise ValueError(f"high must be >= low, got low={low}, high={high}")
    random = rng if rng is not None else np.random
    log_low = np.log(low)
    log_high = np.log(high)
    u = random.uniform(0, 1) ** (1.0 / alpha)
    log_val = log_low + u * (log_high - log_low)
    return int(np.round(np.exp(log_val)))


def loguniform_float(
    low: float,
    high: float,
    alpha: float = 1.0,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> float:
    """Sample a float from a generalized log-uniform distribution."""
    if low <= 0:
        raise ValueError(f"low must be positive, got {low}")
    if high < low:
        raise ValueError(f"high must be >= low, got low={low}, high={high}")
    random = rng if rng is not None else np.random
    log_low = np.log(low)
    log_high = np.log(high)
    u = random.uniform(0, 1) ** (1.0 / alpha)
    log_val = log_low + u * (log_high - log_low)
    return float(np.exp(log_val))
