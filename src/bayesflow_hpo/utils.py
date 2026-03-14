"""Utility helpers used across bayesflow_hpo."""

from __future__ import annotations

import numpy as np


def loguniform_int(
    low: int,
    high: int,
    alpha: float = 1.0,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> int:
    """Sample an integer from a generalized log-uniform distribution.

    The standard log-uniform is ``alpha=1``.  ``alpha > 1`` shifts
    probability mass toward the upper end of the range; ``alpha < 1``
    shifts it toward the lower end.  The transform is ``U^(1/alpha)``
    where ``U ~ Uniform(0, 1)``.

    Parameters
    ----------
    low
        Inclusive lower bound (must be positive).
    high
        Inclusive upper bound.
    alpha
        Shape parameter controlling the skew (default 1.0 = standard
        log-uniform).
    rng
        Optional NumPy random generator or RandomState.  When ``None``,
        uses the global ``np.random`` module (non-deterministic across runs).
    """
    if low <= 0:
        raise ValueError(f"low must be positive, got {low}")
    if high < low:
        raise ValueError(f"high must be >= low, got low={low}, high={high}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    random = rng if rng is not None else np.random
    log_low = np.log(low)
    log_high = np.log(high)
    # Power transform of uniform sample controls skew.
    u = random.uniform(0, 1) ** (1.0 / alpha)
    log_val = log_low + u * (log_high - log_low)
    # Clamp to [low, high] to guard against rounding beyond bounds.
    return int(np.clip(np.round(np.exp(log_val)), low, high))


def loguniform_float(
    low: float,
    high: float,
    alpha: float = 1.0,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> float:
    """Sample a float from a generalized log-uniform distribution.

    Same distribution as :func:`loguniform_int` but returns a
    continuous float without rounding.

    Parameters
    ----------
    low
        Inclusive lower bound (must be positive).
    high
        Inclusive upper bound.
    alpha
        Shape parameter controlling the skew (default 1.0).
    rng
        Optional NumPy random generator or RandomState.  When ``None``,
        uses the global ``np.random`` module (non-deterministic across runs).
    """
    if low <= 0:
        raise ValueError(f"low must be positive, got {low}")
    if high < low:
        raise ValueError(f"high must be >= low, got low={low}, high={high}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    random = rng if rng is not None else np.random
    log_low = np.log(low)
    log_high = np.log(high)
    u = random.uniform(0, 1) ** (1.0 / alpha)
    log_val = log_low + u * (log_high - log_low)
    return float(np.exp(log_val))
