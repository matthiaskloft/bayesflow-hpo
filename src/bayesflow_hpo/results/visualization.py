"""Result visualizations for HPO studies.

Matplotlib-based plots for analyzing HPO results:

- **Pareto front**: first objective vs actual parameter count.
- **Optimization history**: convergence curve with running best.
- **Metric scatter**: two metrics against each other with 2D Pareto.
- **Metric panels**: per-metric vs param count subplots.
- **Parameter importance**: Optuna's fANOVA-based importance ranking.

All plot functions accept an optional ``ax`` parameter for embedding
in user-created figure layouts and return the axes for further
customization.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna

from bayesflow_hpo.results.extraction import _objective_column_names

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trained_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Return completed, non-rejected trials with valid objective values."""
    return [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.values is not None
        and "rejected_reason" not in t.user_attrs
    ]


def _pareto_front_2d(
    xs: list[float],
    ys: list[float],
) -> list[int]:
    """Return indices of non-dominated points (both objectives minimized).

    Uses a sweep-line algorithm: sort by *x* ascending, then keep only
    points with strictly decreasing *y*.  Points with equal *x* are
    resolved by keeping the one with smallest *y*.  O(n log n) time.
    """
    if not xs:
        return []
    order = sorted(range(len(xs)), key=lambda i: (xs[i], ys[i]))
    front: list[int] = [order[0]]
    best_y = ys[order[0]]
    for idx in order[1:]:
        if ys[idx] < best_y:
            front.append(idx)
            best_y = ys[idx]
    return front


def _format_param_count(y: float, _pos: Any) -> str:
    """Format parameter counts with K/M suffixes."""
    if y >= 1e6:
        return f"{y / 1e6:.4g}M"
    if y >= 1e3:
        return f"{y / 1e3:.4g}K"
    return f"{y:.4g}"


def _param_count_formatter() -> plt.FuncFormatter:
    """Y-axis formatter that shows K/M suffixes for parameter counts."""
    return plt.FuncFormatter(_format_param_count)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_pareto_front(
    study: optuna.Study,
    ax: Any | None = None,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Any:
    """Plot first objective versus actual parameter count.

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Created if *None*.
    xlabel : str, optional
        Override for the x-axis label. Auto-derived from study metric names
        when *None*.
    ylabel : str, optional
        Override for the y-axis label. Defaults to ``"Parameter count"``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    obj_cols = _objective_column_names(study)
    if len(obj_cols) < 2:
        ax.text(0.5, 0.5, "Single-objective study",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    trained = [
        t for t in _trained_trials(study)
        if t.user_attrs.get("param_count", 0) > 0
    ]
    if not trained:
        ax.text(0.5, 0.5, "No trained trials",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    obj_values = [t.values[0] for t in trained]
    param_counts = [t.user_attrs["param_count"] for t in trained]
    ax.scatter(obj_values, param_counts, alpha=0.4, label="Trials")

    # Pareto front from study (study-objective level)
    pareto = [
        t for t in study.best_trials
        if t.values is not None
        and "rejected_reason" not in t.user_attrs
        and t.user_attrs.get("param_count", 0) > 0
    ]
    if pareto:
        p_obj = [t.values[0] for t in pareto]
        p_params = [t.user_attrs["param_count"] for t in pareto]
        ax.scatter(p_obj, p_params, c="red", s=90, marker="*", label="Pareto")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_param_count_formatter())
    ax.set_xlabel(xlabel or obj_cols[0])
    ax.set_ylabel(ylabel or "Parameter count")
    ax.set_title("Pareto front")
    ax.legend()
    return ax


def plot_optimization_history(
    study: optuna.Study,
    ax: Any | None = None,
) -> Any:
    """Plot optimization convergence (first objective vs trial number).

    Shows individual trial values as a scatter and a step-line for the
    running best.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    trained = _trained_trials(study)
    if not trained:
        ax.text(0.5, 0.5, "No trained trials",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    # Sort by trial number to get chronological order
    trained.sort(key=lambda t: t.number)
    numbers = [t.number for t in trained]
    values = [t.values[0] for t in trained]

    ax.scatter(numbers, values, alpha=0.4, label="Trials")

    # Best-so-far step line
    best_so_far = []
    running_best = float("inf")
    for v in values:
        running_best = min(running_best, v)
        best_so_far.append(running_best)
    ax.step(numbers, best_so_far, where="post", color="red", label="Best so far")

    obj_cols = _objective_column_names(study)
    ax.set_xlabel("Trial")
    ax.set_ylabel(obj_cols[0])
    ax.set_title("Optimization history")
    ax.legend()
    return ax


def plot_metric_scatter(
    study: optuna.Study,
    x_metric: str,
    y_metric: str,
    ax: Any | None = None,
    *,
    show_iso_lines: bool | None = None,
) -> Any:
    """Scatter plot of two per-trial metrics with 2D Pareto front.

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study.
    x_metric, y_metric : str
        Metric names stored in ``trial.user_attrs`` (e.g.
        ``"calibration_error"``, ``"nrmse"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    show_iso_lines : bool, optional
        Draw iso-mean contour lines. Auto-detected when *None*: enabled if
        the first objective name starts with ``"mean("``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    trained = _trained_trials(study)

    # Collect metric values
    xs, ys = [], []
    for t in trained:
        xv = t.user_attrs.get(x_metric)
        yv = t.user_attrs.get(y_metric)
        if xv is None or yv is None:
            continue
        xs.append(float(xv))
        ys.append(float(yv))

    if not xs:
        logger.warning(
            "No trials have both %s and %s in user_attrs",
            x_metric, y_metric,
        )
        ax.text(0.5, 0.5, "No metric data",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    means = (xs_arr + ys_arr) / 2

    # Color by mean value
    sc = ax.scatter(xs_arr, ys_arr, c=means, cmap="viridis_r", alpha=0.6)
    plt.colorbar(sc, ax=ax, label=f"mean({x_metric}, {y_metric})")

    # 2D Pareto front (both minimized)
    front_idx = _pareto_front_2d(xs, ys)
    if front_idx:
        fx = xs_arr[front_idx]
        fy = ys_arr[front_idx]
        # Sort by x for the step line
        sort_order = np.argsort(fx)
        fx, fy = fx[sort_order], fy[sort_order]
        ax.step(fx, fy, where="post", color="red", linewidth=1.5, zorder=3)
        ax.scatter(fx, fy, c="red", s=90, marker="*", zorder=4, label="Pareto")

    # Iso-mean lines
    if show_iso_lines is None:
        obj_cols = _objective_column_names(study)
        show_iso_lines = obj_cols[0].startswith("mean(")
    if show_iso_lines and len(means) > 1:
        best_mean = float(np.min(means))
        median_mean = float(np.median(means))
        worst_mean = float(np.max(means))
        x_range = np.array([float(xs_arr.min()), float(xs_arr.max())])
        for level, ls in [(best_mean, "-"), (median_mean, "--"), (worst_mean, ":")]:
            # y = 2*level - x  (iso-mean contour for mean = (x+y)/2)
            y_line = 2 * level - x_range
            ax.plot(
                x_range, y_line,
                color="grey", linestyle=ls, alpha=0.5, linewidth=0.8,
            )
        ax.plot([], [], color="grey", linestyle="--", alpha=0.5, label="Iso-mean")

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{x_metric} vs {y_metric}")
    ax.legend()
    return ax


def plot_metric_panels(
    study: optuna.Study,
    metrics: list[str] | None = None,
    axes: Any | None = None,
) -> Any:
    """Per-metric vs parameter count subplots with 2D Pareto fronts.

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study.
    metrics : list of str, optional
        Metric names from ``trial.user_attrs``. Auto-detected when *None*.
    axes : array of matplotlib.axes.Axes, optional
        Pre-created axes (length must match *metrics*).
    """
    trained = [
        t for t in _trained_trials(study)
        if t.user_attrs.get("param_count", 0) > 0
    ]

    # Auto-detect metrics from user_attrs
    if metrics is None:
        if not trained:
            metrics = []
        else:
            all_keys: set[str] = set()
            for t in trained:
                all_keys.update(t.user_attrs.keys())
            # Keep only numeric metric-like keys, exclude bookkeeping attrs
            exclude = {"param_count", "rejected_reason", "param_budget"}
            metrics = sorted(
                k for k in all_keys - exclude
                if any(
                    isinstance(t.user_attrs.get(k), (int, float))
                    for t in trained
                )
            )

    n = len(metrics)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No metrics found",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
        axes = axes[0]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        mvs, pcs = [], []
        for t in trained:
            mv = t.user_attrs.get(metric)
            if mv is None:
                continue
            mvs.append(float(mv))
            pcs.append(t.user_attrs["param_count"])

        if not mvs:
            ax.text(0.5, 0.5, f"No data for {metric}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        ax.scatter(pcs, mvs, alpha=0.4)

        # 2D Pareto: minimize metric AND param_count
        front_idx = _pareto_front_2d(pcs, mvs)
        if front_idx:
            pcs_arr = np.asarray(pcs)
            mvs_arr = np.asarray(mvs)
            fx = pcs_arr[front_idx]
            fy = mvs_arr[front_idx]
            sort_order = np.argsort(fx)
            fx, fy = fx[sort_order], fy[sort_order]
            ax.step(fx, fy, where="post", color="red", linewidth=1.5, zorder=3)
            ax.scatter(fx, fy, c="red", s=90, marker="*", zorder=4)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(_param_count_formatter())
        ax.set_xlabel("Parameter count")
        ax.set_ylabel(metric)
        ax.set_title(metric)

    return axes


def plot_param_importance(
    study: optuna.Study,
    ax: Any | None = None,
    top_k: int = 10,
    *,
    target_name: str | None = None,
) -> Any:
    """Plot Optuna parameter importances.

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    top_k : int
        Maximum number of parameters to show.
    target_name : str, optional
        Target a specific metric stored in ``trial.user_attrs`` instead of
        the first objective value. Useful for decomposing importance by
        individual metrics (e.g. ``"calibration_error"``).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    try:
        if target_name is not None:
            def target(t: optuna.trial.FrozenTrial) -> float:
                if "rejected_reason" in t.user_attrs:
                    return float("inf")
                return t.user_attrs.get(target_name, float("inf"))
        elif len(study.directions) > 1:
            def target(t: optuna.trial.FrozenTrial) -> float:
                return t.values[0] if t.values else float("inf")
        else:
            target = None

        importance = optuna.importance.get_param_importances(
            study, target=target,
        )
    except Exception:
        ax.text(
            0.5, 0.5, "Importance unavailable",
            ha="center", va="center", transform=ax.transAxes,
        )
        return ax

    params = list(importance.keys())[:top_k]
    values = [importance[p] for p in params]

    y_pos = np.arange(len(params))
    ax.barh(y_pos, values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    title = (
        f"Parameter importance ({target_name})"
        if target_name
        else "Parameter importance"
    )
    ax.set_title(title)
    return ax
