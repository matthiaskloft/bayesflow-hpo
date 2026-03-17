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
_DESCRIPTIONS: dict[str, str] = {}  # canonical name → one-line description


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_metric(
    name: str,
    fn: MetricFn,
    aliases: list[str] | None = None,
    overwrite: bool = False,
    description: str | None = None,
) -> None:
    """Register a metric function under *name* (and optional aliases).

    Parameters
    ----------
    name
        Canonical name used to look up the metric (e.g. ``"nrmse"``).
    fn
        Callable with signature
        ``(draws: ndarray[n, s], true_values: ndarray[n]) -> dict``.
    aliases
        Alternative names that resolve to *name*
        (e.g. ``["cal_error"]`` for ``"calibration_error"``).
    overwrite
        If ``True``, silently replace an existing metric with the same
        *name*.  Default ``False`` (raises :class:`ValueError`).
    description
        One-line human-readable summary shown by
        :func:`describe_metrics`.  ``None`` (default) leaves any
        existing description unchanged on overwrite.

    Raises
    ------
    ValueError
        If *name* is already registered and *overwrite* is ``False``.

    See Also
    --------
    describe_metrics : Discover all registered metrics.
    get_metric : Look up a single metric by name or alias.
    """
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"Metric '{name}' is already registered. "
            "Use overwrite=True to replace."
        )
    _REGISTRY[name] = fn
    if description is not None:
        _DESCRIPTIONS[name] = description
    if aliases:
        for alias in aliases:
            _ALIASES[alias] = name


def get_metric(name: str) -> MetricFn:
    """Look up a metric by name or alias.

    Parameters
    ----------
    name
        Canonical name or alias (e.g. ``"nrmse"`` or ``"corr"``).

    Returns
    -------
    MetricFn
        The registered callable.

    Raises
    ------
    KeyError
        If *name* is not a registered metric or alias.
    """
    canonical = _ALIASES.get(name, name)
    if canonical not in _REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Available: {list_metrics()}")
    return _REGISTRY[canonical]


def resolve_metrics(names: list[str]) -> dict[str, MetricFn]:
    """Resolve a list of metric names to a ``{name: fn}`` dict.

    Parameters
    ----------
    names
        Metric names or aliases to resolve.

    Returns
    -------
    dict[str, MetricFn]
        Mapping from the *input* names to their callables.

    Raises
    ------
    KeyError
        If any name in *names* is unknown.
    """
    return {n: get_metric(n) for n in names}


def list_metrics() -> list[str]:
    """Return sorted canonical names of all registered metrics.

    Aliases are excluded; use :func:`describe_metrics` for a full
    listing that includes aliases and descriptions.

    Returns
    -------
    list[str]
        Sorted metric names.
    """
    return sorted(_REGISTRY)


def describe_metrics() -> list[dict[str, str]]:
    """Return a description of every registered metric.

    Each entry contains ``name``, ``aliases`` (comma-separated), and
    ``description``.  Useful for discovering which strings are valid
    for ``objective_metrics`` in :func:`~bayesflow_hpo.optimize`.

    Returns
    -------
    list of dict
        One dict per metric with keys ``"name"``, ``"aliases"``,
        ``"description"``.

    Examples
    --------
    >>> from bayesflow_hpo import describe_metrics
    >>> for m in describe_metrics():
    ...     print(f"{m['name']:20s} {m['description']}")
    """
    inverse_aliases: dict[str, list[str]] = {}
    for alias, canonical in _ALIASES.items():
        inverse_aliases.setdefault(canonical, []).append(alias)

    rows: list[dict[str, str]] = []
    for name in sorted(_REGISTRY):
        rows.append({
            "name": name,
            "aliases": ", ".join(sorted(inverse_aliases.get(name, []))),
            "description": _DESCRIPTIONS.get(name, ""),
        })
    return rows


# ---------------------------------------------------------------------------
# BayesFlow diagnostic wrappers
# ---------------------------------------------------------------------------


def _reshape_for_bf(
    draws: np.ndarray, true_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape validation arrays to BayesFlow diagnostic format.

    BayesFlow diagnostics expect ``(n_sims, n_samples, n_params)`` for
    estimates and ``(n_sims, n_params)`` for targets.  Since validation
    runs per-parameter, the last axis is always 1.
    """
    return draws[:, :, np.newaxis], true_values[:, np.newaxis]


def _bf_calibration_error(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """Expected Calibration Error (ECE) via BayesFlow diagnostics."""
    import bayesflow as bf

    estimates, targets = _reshape_for_bf(draws, true_values)
    result = bf.diagnostics.calibration_error(estimates=estimates, targets=targets)
    return {"calibration_error": float(np.mean(result["values"]))}


def _bf_rmse(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """Root Mean Squared Error of posterior means vs true values.

    Uses BayesFlow's ``root_mean_squared_error`` when available,
    otherwise falls back to a manual NumPy implementation.
    """
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
    """Normalized RMSE (range-normalized) of posterior means.

    Divides RMSE by the range of true values so the metric is
    comparable across parameters with different scales.
    """
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
    """Posterior contraction: how much the posterior narrows vs the prior.

    Values near 1 indicate strong learning; near 0 indicates the
    posterior is as wide as the prior.
    """
    import bayesflow as bf

    estimates, targets = _reshape_for_bf(draws, true_values)
    result = bf.diagnostics.posterior_contraction(
        estimates=estimates, targets=targets,
    )
    return {"contraction": float(np.mean(result["values"]))}


def _bf_z_score(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """Posterior z-score: (true - posterior_mean) / posterior_std.

    Returns both the mean z-score (bias indicator) and mean absolute
    z-score (overall calibration indicator).
    """
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
    """Log-gamma calibration diagnostic from BayesFlow."""
    import bayesflow as bf

    estimates, targets = _reshape_for_bf(draws, true_values)
    result = bf.diagnostics.calibration_log_gamma(
        estimates=estimates, targets=targets,
    )
    return {"log_gamma": float(np.mean(result["values"]))}


# ---------------------------------------------------------------------------
# Native metrics
# ---------------------------------------------------------------------------


def _sbc_ranks(
    draws: np.ndarray, true_values: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Compute SBC ranks shared by all SBC metrics."""
    n_posterior_samples = draws.shape[1]
    ranks = np.sum(draws < true_values[:, None], axis=1)
    return ranks, n_posterior_samples


def _sbc_ks_metric(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """SBC rank uniformity via the Kolmogorov-Smirnov test.

    Returns the KS statistic (minimize → 0 = perfectly uniform ranks).
    """
    from bayesflow_hpo.validation.sbc_tests import compute_sbc_uniformity_tests

    ranks, n_posterior_samples = _sbc_ranks(draws, true_values)
    full = compute_sbc_uniformity_tests(ranks, n_posterior_samples)
    return {"sbc_ks": full["sbc_ks_stat"]}


def _sbc_chi2_metric(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """SBC rank uniformity via the chi-squared goodness-of-fit test.

    Returns the chi-squared statistic (minimize → 0 = perfectly uniform
    ranks).  NaN when expected bin counts are below 5.
    """
    from bayesflow_hpo.validation.sbc_tests import compute_sbc_uniformity_tests

    ranks, n_posterior_samples = _sbc_ranks(draws, true_values)
    full = compute_sbc_uniformity_tests(ranks, n_posterior_samples)
    return {"sbc_chi2": full["sbc_chi2_stat"]}


def _sbc_c2st_metric(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """SBC rank uniformity via the Classifier Two-Sample Test (C2ST).

    Returns ``accuracy - 0.5`` (minimize → 0 = classifier cannot
    distinguish observed ranks from uniform).  Requires scikit-learn.
    """
    from bayesflow_hpo.validation.sbc_tests import compute_sbc_c2st

    ranks, n_posterior_samples = _sbc_ranks(draws, true_values)
    result = compute_sbc_c2st(ranks, n_posterior_samples)
    acc = result["sbc_c2st_accuracy"]
    if np.isnan(acc):
        return {"sbc_c2st": np.nan}
    return {"sbc_c2st": acc - 0.5}




def _bias_metric(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """Mean signed error (posterior mean − true value).

    Positive bias means the model systematically overestimates;
    negative means it underestimates.
    """
    posterior_mean = np.mean(draws, axis=1)
    return {"bias": float(np.mean(posterior_mean - true_values))}


def _mae_metric(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """Mean Absolute Error of posterior means vs true values."""
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

DEFAULT_COVERAGE_LEVELS = [0.9, 0.95, 0.975, 0.99]


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
            level_int = round(level * 100)

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
    """Two-sided SBC rank coverage at standard credible-interval levels."""
    return make_coverage_metric(side="two-sided")(draws, true_values)


def _coverage_left(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """Left-sided coverage — useful for assessing statistical efficiency."""
    fn = make_coverage_metric(side="left", prefix="left_")
    return fn(draws, true_values)


def _coverage_right(
    draws: np.ndarray, true_values: np.ndarray,
) -> dict[str, float]:
    """Right-sided coverage — useful for assessing futility/conservatism."""
    fn = make_coverage_metric(side="right", prefix="right_")
    return fn(draws, true_values)


# ---------------------------------------------------------------------------
# Register built-in metrics
# ---------------------------------------------------------------------------

DEFAULT_METRICS = [
    "calibration_error", "nrmse", "correlation", "coverage", "rmse", "contraction",
]

# BF wrappers
register_metric(
    "calibration_error", _bf_calibration_error,
    aliases=["cal_error"],
    description="Expected Calibration Error (ECE) via BayesFlow diagnostics.",
)
register_metric(
    "rmse", _bf_rmse,
    description="Root Mean Squared Error of posterior means vs true values.",
)
register_metric(
    "nrmse", _bf_nrmse,
    description="Range-normalized RMSE (comparable across parameter scales).",
)
register_metric(
    "contraction", _bf_contraction,
    description="Posterior contraction (1 = strong learning, 0 = no narrowing).",
)
register_metric(
    "z_score", _bf_z_score,
    description="Posterior z-score (mean and mean-absolute; bias + calibration).",
)
register_metric(
    "log_gamma", _bf_log_gamma,
    description="Log-gamma calibration diagnostic from BayesFlow.",
)

# Native metrics — SBC rank uniformity
register_metric(
    "sbc_ks", _sbc_ks_metric,
    description="SBC KS statistic (minimize → 0 = uniform ranks).",
)
register_metric(
    "sbc_chi2", _sbc_chi2_metric,
    description="SBC chi-squared statistic (minimize → 0 = uniform ranks).",
)
register_metric(
    "sbc_c2st", _sbc_c2st_metric,
    description="SBC C2ST deviation (accuracy − 0.5; minimize → 0). Requires sklearn.",
)
register_metric(
    "coverage", _coverage_two_sided,
    aliases=["coverage_two_sided"],
    description="Two-sided SBC rank coverage at standard credible-interval levels.",
)
register_metric(
    "coverage_left", _coverage_left,
    description="Left-sided coverage (statistical efficiency).",
)
register_metric(
    "coverage_right", _coverage_right,
    description="Right-sided coverage (futility / conservatism).",
)
register_metric(
    "bias", _bias_metric,
    description="Mean signed error (positive=overestimate, negative=underestimate).",
)
register_metric(
    "mae", _mae_metric,
    description="Mean Absolute Error of posterior means vs true values.",
)
register_metric(
    "correlation", _correlation_metric,
    aliases=["corr"],
    description="Pearson correlation between posterior means and true values.",
)
