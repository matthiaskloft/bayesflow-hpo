"""Metric registry for validation pipeline.

Maps string names to callable metric functions. Built-in metrics wrap
BayesFlow diagnostics; additional metrics (SBC, coverage, bias, MAE) are
provided natively. Users can register custom metrics via
:func:`register_metric`.

Metric function signature
-------------------------
``(draws: ndarray[n, s], true_values: ndarray[n]) -> dict``
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], dict[str, float]]

_REGISTRY: dict[str, MetricFn] = {}
_ALIASES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_metric(
    name: str,
    fn: MetricFn,
    aliases: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a metric function under *name* (and optional aliases)."""
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"Metric '{name}' is already registered. "
            "Use overwrite=True to replace."
        )
    _REGISTRY[name] = fn
    if aliases:
        for alias in aliases:
            _ALIASES[alias] = name


def get_metric(name: str) -> MetricFn:
    """Look up a metric by name or alias."""
    canonical = _ALIASES.get(name, name)
    if canonical not in _REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Available: {list_metrics()}")
    return _REGISTRY[canonical]


def resolve_metrics(names: list[str]) -> dict[str, MetricFn]:
    """Resolve a list of metric names to a ``{name: fn}`` dict."""
    return {n: get_metric(n) for n in names}


def list_metrics() -> list[str]:
    """Return sorted list of registered metric names (not aliases)."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# BayesFlow diagnostic wrappers
# ---------------------------------------------------------------------------


def _reshape_for_bf(
    draws: np.ndarray, true_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape (n_sims, n_samples) → BF format (n_sims, n_samples, 1)."""
    return draws[:, :, np.newaxis], true_values[:, np.newaxis]


def _bf_calibration_error(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    import bayesflow as bf

    estimates, targets = _reshape_for_bf(draws, true_values)
    result = bf.diagnostics.calibration_error(estimates=estimates, targets=targets)
    return {"calibration_error": float(np.mean(result["values"]))}


def _bf_rmse(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    import bayesflow as bf

    rmse_fn = getattr(bf.diagnostics, "root_mean_squared_error", None)
    if rmse_fn is not None:
        estimates, targets = _reshape_for_bf(draws, true_values)
        result = rmse_fn(estimates=estimates, targets=targets)
        return {"rmse": float(np.mean(result["values"]))}

    posterior_mean = np.mean(draws, axis=1)
    rmse = float(np.sqrt(np.mean((posterior_mean - true_values) ** 2)))
    return {"rmse": rmse}


def _bf_nrmse(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    import bayesflow as bf

    rmse_fn = getattr(bf.diagnostics, "root_mean_squared_error", None)
    if rmse_fn is not None:
        estimates, targets = _reshape_for_bf(draws, true_values)
        result = rmse_fn(estimates=estimates, targets=targets, normalize="range")
        return {"nrmse": float(np.mean(result["values"]))}

    posterior_mean = np.mean(draws, axis=1)
    rmse = float(np.sqrt(np.mean((posterior_mean - true_values) ** 2)))
    value_range = float(np.max(true_values) - np.min(true_values))
    denom = value_range if value_range > 0 else 1.0
    return {"nrmse": rmse / denom}


def _bf_contraction(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    import bayesflow as bf

    estimates, targets = _reshape_for_bf(draws, true_values)
    result = bf.diagnostics.posterior_contraction(
        estimates=estimates, targets=targets,
    )
    return {"contraction": float(np.mean(result["values"]))}


def _bf_z_score(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    import bayesflow as bf

    estimates, targets = _reshape_for_bf(draws, true_values)
    result = bf.diagnostics.posterior_z_score(
        estimates=estimates, targets=targets,
    )
    vals = result["values"].flatten()
    return {
        "mean_abs_z_score": float(np.mean(np.abs(vals))),
        "mean_z_score": float(np.mean(vals)),
    }


def _bf_log_gamma(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    import bayesflow as bf

    estimates, targets = _reshape_for_bf(draws, true_values)
    result = bf.diagnostics.calibration_log_gamma(
        estimates=estimates, targets=targets,
    )
    return {"log_gamma": float(np.mean(result["values"]))}


# ---------------------------------------------------------------------------
# Native metrics
# ---------------------------------------------------------------------------


def _sbc_metric(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """SBC rank uniformity tests (KS, chi-squared, C2ST)."""
    from bayesflow_hpo.validation.sbc_tests import (
        compute_sbc_c2st,
        compute_sbc_uniformity_tests,
    )

    n_posterior_samples = draws.shape[1]
    ranks = np.sum(draws < true_values[:, None], axis=1)
    result: dict[str, float] = {}
    result.update(compute_sbc_uniformity_tests(ranks, n_posterior_samples))
    result.update(compute_sbc_c2st(ranks, n_posterior_samples))
    return result


def _bias_metric(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    posterior_mean = np.mean(draws, axis=1)
    return {"bias": float(np.mean(posterior_mean - true_values))}


def _mae_metric(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    posterior_mean = np.mean(draws, axis=1)
    return {"mae": float(np.mean(np.abs(posterior_mean - true_values)))}


def _correlation_metric(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """Pearson correlation between posterior means and true values."""
    posterior_mean = np.mean(draws, axis=1)
    if np.std(true_values) < 1e-12 or np.std(posterior_mean) < 1e-12:
        return {"correlation": 0.0}
    corr = float(np.corrcoef(posterior_mean, true_values)[0, 1])
    if np.isnan(corr):
        corr = 0.0
    return {"correlation": corr}


# ---------------------------------------------------------------------------
# SBC rank-based coverage
# ---------------------------------------------------------------------------

DEFAULT_COVERAGE_LEVELS = [0.5, 0.8, 0.9, 0.95, 0.99]


def make_coverage_metric(
    levels: list[float] | None = None,
    side: str = "two-sided",
    weights: list[float] | None = None,
    prefix: str = "",
) -> MetricFn:
    """Factory for SBC rank-based coverage metrics.

    Parameters
    ----------
    levels
        Nominal coverage levels (default: ``[0.5, 0.8, 0.9, 0.95, 0.99]``).
    side
        ``"two-sided"`` (standard calibration), ``"left"`` (efficiency),
        or ``"right"`` (futility).
    weights
        Per-level weights for the weighted mean calibration error.
        If ``None``, uniform weights are used.
    prefix
        Key prefix for output dict (e.g., ``"left_"``).
    """
    if levels is None:
        levels = list(DEFAULT_COVERAGE_LEVELS)
    if weights is not None and len(weights) != len(levels):
        raise ValueError(
            f"weights length ({len(weights)}) must match "
            f"levels length ({len(levels)})"
        )
    valid_sides = ("two-sided", "left", "right")
    if side not in valid_sides:
        raise ValueError(
            f"side must be one of {valid_sides}, got '{side}'"
        )

    def metric_fn(draws: np.ndarray, true_values: np.ndarray) -> dict[str, float]:
        n_sims, n_samples = draws.shape
        ranks = np.sum(draws < true_values[:, None], axis=1)
        normalized_ranks = ranks / (n_samples + 1)

        result: dict[str, float] = {}
        cal_errors: list[float] = []

        for level in levels:
            level_int = int(level * 100)

            if side == "two-sided":
                alpha = 1 - level
                lo = alpha / 2
                hi = 1 - alpha / 2
                in_interval = (normalized_ranks >= lo) & (normalized_ranks <= hi)
            elif side == "left":
                in_interval = normalized_ranks <= level
            else:  # right
                in_interval = normalized_ranks >= 1 - level

            empirical = float(np.mean(in_interval))
            cal_error = abs(empirical - level)
            result[f"{prefix}coverage_{level_int}"] = empirical
            cal_errors.append(cal_error)

        if weights is not None:
            w = np.asarray(weights, dtype=float)
            weighted = np.sum(w * np.asarray(cal_errors))
            result[f"{prefix}mean_cal_error"] = float(weighted / np.sum(w))
        else:
            result[f"{prefix}mean_cal_error"] = float(np.mean(cal_errors))

        return result

    return metric_fn


def _coverage_two_sided(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    return make_coverage_metric(side="two-sided")(draws, true_values)


def _mean_cal_error(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """SBC rank-based mean absolute coverage error across standard levels.

    Averages |empirical − nominal| at [0.5, 0.8, 0.9, 0.95, 0.99]. This is
    more comprehensive than ``calibration_error`` (BayesFlow ECE) because it
    tests the full posterior via SBC ranks at multiple credible-interval levels.
    """
    result = make_coverage_metric(side="two-sided")(draws, true_values)
    return {"mean_cal_error": result["mean_cal_error"]}


def _coverage_left(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    fn = make_coverage_metric(side="left", prefix="left_")
    return fn(draws, true_values)


def _coverage_right(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    fn = make_coverage_metric(side="right", prefix="right_")
    return fn(draws, true_values)


# ---------------------------------------------------------------------------
# Register built-in metrics
# ---------------------------------------------------------------------------

DEFAULT_METRICS = [
    "mean_cal_error", "nrmse", "correlation", "coverage", "rmse", "contraction", "sbc",
]

# BF wrappers
register_metric("calibration_error", _bf_calibration_error, aliases=["cal_error"])
register_metric("rmse", _bf_rmse)
register_metric("nrmse", _bf_nrmse)
register_metric("contraction", _bf_contraction)
register_metric("z_score", _bf_z_score)
register_metric("log_gamma", _bf_log_gamma)

# Native metrics
register_metric("mean_cal_error", _mean_cal_error)
register_metric("sbc", _sbc_metric)
register_metric("coverage", _coverage_two_sided, aliases=["coverage_two_sided"])
register_metric("coverage_left", _coverage_left)
register_metric("coverage_right", _coverage_right)
register_metric("bias", _bias_metric)
register_metric("mae", _mae_metric)
register_metric("correlation", _correlation_metric, aliases=["corr"])
