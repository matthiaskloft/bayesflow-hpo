"""Result visualizations for HPO studies."""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna

from bayesflow_hpo.results import _colors as c
from bayesflow_hpo.results.extraction import _objective_column_names

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trained_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Return completed, non-rejected trials."""
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

    Sorts by *x* ascending, then sweeps for monotonically decreasing *y*.
    Points with equal x are resolved by keeping the one with the smallest y.
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


def _valid_nobj_trials(
    study: optuna.Study,
    n: int,
) -> list[optuna.trial.FrozenTrial]:
    """Return trained trials with *n* valid (non-NaN) objective values."""
    return [
        t for t in _trained_trials(study)
        if t.values
        and len(t.values) >= n
        and all(v is not None and not math.isnan(v) for v in t.values[:n])
    ]


def _draw_pareto_overlay(
    ax: Any,
    xs: list[float],
    ys: list[float],
    *,
    draw_step: bool = True,
    draw_markers: bool = True,
    label: str | None = "Pareto",
) -> None:
    """Draw 2D Pareto front step line and/or markers on *ax*.

    Computes the non-dominated front from *xs* / *ys*, then overlays
    accent-colored elements.  Use *draw_step* and *draw_markers* to
    control which elements are drawn.  Set *label* to ``None`` to
    suppress the legend entry (useful when the caller draws its own
    Pareto markers separately).
    """
    front_idx = _pareto_front_2d(xs, ys)
    if not front_idx:
        return
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    fx = xs_arr[front_idx]
    fy = ys_arr[front_idx]
    sort_order = np.argsort(fx)
    fx, fy = fx[sort_order], fy[sort_order]
    if draw_step:
        ax.step(fx, fy, where="post", color=c.ACCENT, linewidth=1.5,
                zorder=3)
    if draw_markers:
        ax.scatter(fx, fy, c=c.ACCENT, s=c.PARETO_SIZE,
                   marker=c.PARETO_MARKER, zorder=4, label=label)


def _normalize_to_sizes(
    values: np.ndarray,
    size_min: float = 20.0,
    size_max: float = 200.0,
) -> np.ndarray:
    """Map *values* linearly to marker sizes in [*size_min*, *size_max*]."""
    v_min, v_max = values.min(), values.max()
    if v_max > v_min:
        return size_min + (size_max - size_min) * (values - v_min) / (v_max - v_min)
    return np.full_like(values, (size_min + size_max) / 2)


def _get_metric_user_attrs(
    study: optuna.Study,
) -> list[str]:
    """Return sorted list of numeric user-attr metric keys."""
    trained = _trained_trials(study)
    if not trained:
        return []
    all_keys: set[str] = set()
    for t in trained:
        all_keys.update(t.user_attrs.keys())
    exclude = {"param_count", "rejected_reason", "param_budget"}
    return sorted(
        k for k in all_keys - exclude
        if any(isinstance(t.user_attrs.get(k), (int, float)) for t in trained)
    )


def _setup_grid(
    n_panels: int,
    axes: Any | None,
    *,
    max_cols: int = 3,
    figsize: tuple[float, float] | None = None,
    panel_size: tuple[float, float] = (5.0, 4.5),
) -> tuple[plt.Figure | None, np.ndarray]:
    """Create or validate a subplot grid for multi-panel plots.

    Parameters
    ----------
    n_panels : int
        Number of panels to draw.
    axes : array-like or None
        Pre-allocated axes (embedded mode) or *None* (standalone mode).
    max_cols : int
        Maximum columns per row (standalone mode only).
    figsize : tuple, optional
        Explicit figure size. Auto-computed from *panel_size* when *None*.
    panel_size : tuple
        ``(width, height)`` per panel for auto-computed figsize.

    Returns
    -------
    tuple of (fig_or_None, axes_1d)
        *fig_or_None* is the ``Figure`` in standalone mode, ``None`` in
        embedded mode.  *axes_1d* is a 1D array of exactly *n_panels*
        ``Axes`` objects.
    """
    if axes is not None:
        flat = np.asarray(axes, dtype=object).ravel()
        if flat.size < n_panels:
            raise ValueError(
                f"Expected at least {n_panels} axes, got {flat.size}."
            )
        return None, flat[:n_panels]
    if n_panels <= 0:
        fig, ax = plt.subplots(figsize=figsize or (5, 4))
        return fig, np.array([ax])
    n_cols = min(max_cols, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    if figsize is None:
        figsize = (panel_size[0] * n_cols, panel_size[1] * n_rows)
    fig, ax_grid = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat = ax_grid.flatten()
    for idx in range(n_panels, len(flat)):
        flat[idx].set_visible(False)
    return fig, flat[:n_panels]


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_pareto_front(
    study: optuna.Study,
    axes: Any | None = None,
    *,
    third_dim: str = "color",
    max_cols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot pairwise 2D Pareto projections of study objectives.

    For each pair ``(i, j)`` of objectives, plots ``objective[i]`` on X
    versus ``objective[j]`` on Y.  When the study has 3+ objectives, the
    omitted objective for each pair is encoded via *third_dim*.

    Two Pareto layers are drawn per panel:

    1. A 2D non-dominated step line computed per-projection.
    2. Study-level Pareto markers (``study.best_trials``) — these are
       Pareto-optimal in the full N-dimensional objective space but may
       not be non-dominated in the 2D projection.

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study with 2+ objectives.
    axes : array of matplotlib.axes.Axes, optional
        Pre-allocated axes (one per pair).  When *None*, a figure is
        created automatically (standalone mode).
    third_dim : ``"color"`` | ``"size"`` | ``"none"``
        Encoding for the omitted objective in 3+ objective studies.
        Ignored for 2-objective studies (no omitted dimension).
    max_cols : int
        Maximum columns per row in standalone mode.
    figsize : tuple, optional
        Explicit figure size.  Auto-computed when *None*.

    Returns
    -------
    matplotlib.figure.Figure (standalone) or ndarray of Axes (embedded)
    """
    obj_cols = _objective_column_names(study)
    n_obj = len(obj_cols)

    if n_obj < 2:
        fig, ax_arr = _setup_grid(1, axes, max_cols=max_cols, figsize=figsize)
        ax_arr[0].text(0.5, 0.5, "Need at least 2 objectives",
                       ha="center", va="center", transform=ax_arr[0].transAxes)
        return fig if fig is not None else ax_arr

    # All pairwise objective combinations
    pairs = [(i, j) for i in range(n_obj) for j in range(i + 1, n_obj)]

    fig, ax_arr = _setup_grid(len(pairs), axes, max_cols=max_cols,
                              figsize=figsize)

    trials = _valid_nobj_trials(study, n_obj)
    pareto_numbers = {t.number for t in study.best_trials}

    for panel_idx, (i, j) in enumerate(pairs):
        ax = ax_arr[panel_idx]
        if not trials:
            ax.text(0.5, 0.5, "No valid trials",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{obj_cols[i]} vs {obj_cols[j]}")
            continue

        xi = [t.values[i] for t in trials]
        xj = [t.values[j] for t in trials]

        # Determine omitted dimension (only for 3+ objectives).
        # When >1 dimension is omitted (4+ objectives), encode the first.
        all_indices = set(range(n_obj))
        omitted_indices = sorted(all_indices - {i, j})
        omitted_idx = omitted_indices[0] if omitted_indices else None

        # Scatter trials with optional 3rd-dimension encoding
        if omitted_idx is not None and third_dim == "color":
            omitted_vals = np.array([t.values[omitted_idx] for t in trials])
            sc = ax.scatter(xi, xj, c=omitted_vals, cmap="viridis_r",
                            alpha=0.6, label="Trials")
            plt.colorbar(sc, ax=ax, label=obj_cols[omitted_idx])
        elif omitted_idx is not None and third_dim == "size":
            omitted_vals = np.array([t.values[omitted_idx] for t in trials])
            sizes = _normalize_to_sizes(omitted_vals)
            ax.scatter(xi, xj, s=sizes, color=c.PRIMARY,
                       alpha=c.ALPHA_TRIAL, label="Trials")
        else:
            ax.scatter(xi, xj, color=c.PRIMARY, alpha=c.ALPHA_TRIAL,
                       label="Trials")

        # Layer 1: 2D non-dominated step line per projection
        _draw_pareto_overlay(ax, xi, xj, draw_step=True, draw_markers=False)

        # Layer 2: study-level Pareto markers (N-D optimal)
        pareto_in = [t for t in trials if t.number in pareto_numbers]
        if pareto_in:
            px = [t.values[i] for t in pareto_in]
            py = [t.values[j] for t in pareto_in]
            ax.scatter(px, py, c=c.ACCENT, s=c.PARETO_SIZE,
                       marker=c.PARETO_MARKER, zorder=4, label="Pareto")

        ax.set_xlabel(obj_cols[i])
        ax.set_ylabel(obj_cols[j])
        ax.set_title(f"{obj_cols[i]} vs {obj_cols[j]}")
        ax.legend(loc="best", framealpha=0.8, fontsize="small")

    return fig if fig is not None else ax_arr


def plot_optimization_history(
    study: optuna.Study,
    axes: Any | None = None,
    *,
    max_cols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot per-objective optimization convergence as best-so-far step lines.

    Creates one panel per objective.  Each panel shows a step line of the
    running best value (respecting the objective's direction: ``min`` for
    ``MINIMIZE``, ``max`` for ``MAXIMIZE``).

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study.
    axes : array of matplotlib.axes.Axes, optional
        Pre-allocated axes (one per objective).  When *None*, a figure is
        created automatically (standalone mode).
    max_cols : int
        Maximum columns per row in standalone mode.
    figsize : tuple, optional
        Explicit figure size.  Auto-computed when *None*.

    Returns
    -------
    matplotlib.figure.Figure (standalone) or ndarray of Axes (embedded)
    """
    obj_cols = _objective_column_names(study)
    n_obj = len(obj_cols)

    fig, ax_arr = _setup_grid(n_obj, axes, max_cols=max_cols, figsize=figsize)

    trained = _trained_trials(study)
    trained.sort(key=lambda t: t.number)

    for obj_idx in range(n_obj):
        ax = ax_arr[obj_idx]

        if not trained:
            ax.text(0.5, 0.5, "No trained trials",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(obj_cols[obj_idx])
            continue

        numbers = [t.number for t in trained]
        values = [t.values[obj_idx] for t in trained]

        # Direction-aware best-so-far
        direction = study.directions[obj_idx]
        best_func = (max if direction == optuna.study.StudyDirection.MAXIMIZE
                     else min)
        best_so_far = list(itertools.accumulate(values, best_func))

        ax.step(numbers, best_so_far, where="post", color=c.BEST_LINE,
                label="Best so far")

        ax.set_xlabel("Trial")
        ax.set_ylabel(obj_cols[obj_idx])
        ax.set_title(obj_cols[obj_idx])

    return fig if fig is not None else ax_arr


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

    _draw_pareto_overlay(ax, xs, ys)

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
                color=c.SECONDARY, linestyle=ls, alpha=0.5, linewidth=0.8,
            )
        ax.plot([], [], color=c.SECONDARY, linestyle="--", alpha=0.5,
                label="Iso-mean")

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{x_metric} vs {y_metric}")
    ax.legend(loc="best", framealpha=0.8, fontsize="small")
    return ax


def plot_metric_panels(
    study: optuna.Study,
    metrics: list[str] | None = None,
    axes: Any | None = None,
    *,
    max_cols: int = 3,
    figsize: tuple[float, float] | None = None,
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
    max_cols : int
        Maximum columns per row in standalone mode.
    figsize : tuple, optional
        Explicit figure size.  Auto-computed when *None*.
    """
    trained = [
        t for t in _trained_trials(study)
        if t.user_attrs.get("param_count", 0) > 0
    ]

    # Auto-detect metrics from user_attrs
    if metrics is None:
        metrics = _get_metric_user_attrs(study) if trained else []

    n = len(metrics)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No metrics found",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    fig, ax_arr = _setup_grid(n, axes, max_cols=max_cols,
                              figsize=figsize, panel_size=(5.0, 5.0))

    for i, metric in enumerate(metrics):
        ax = ax_arr[i]
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

        ax.scatter(pcs, mvs, color=c.PRIMARY, alpha=c.ALPHA_TRIAL)

        # 2D Pareto: minimize metric AND param_count
        _draw_pareto_overlay(ax, pcs, mvs)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(_param_count_formatter())
        ax.set_xlabel("Parameter count")
        ax.set_ylabel(metric)
        ax.set_title(metric)

    return fig if fig is not None else ax_arr


def plot_param_importance(
    study: optuna.Study,
    axes: Any | None = None,
    top_k: int = 10,
    *,
    max_cols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> Any | None:
    """Plot per-objective parameter importance bar charts.

    Creates one bar chart per objective showing the top-*k* most
    important hyperparameters according to Optuna's fANOVA importance
    evaluator.

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study.
    axes : array of matplotlib.axes.Axes, optional
        Pre-allocated axes (one per objective).  When *None*, a figure is
        created automatically (standalone mode).
    top_k : int
        Maximum number of parameters to show per panel.
    max_cols : int
        Maximum columns per row in standalone mode.
    figsize : tuple, optional
        Explicit figure size.  Auto-computed when *None*.

    Returns
    -------
    matplotlib.figure.Figure (standalone), ndarray of Axes (embedded),
    or *None* if **all** panels failed (signals the orchestrator to drop
    the importance row).
    """
    obj_cols = _objective_column_names(study)
    n_obj = len(obj_cols)

    fig, ax_arr = _setup_grid(n_obj, axes, max_cols=max_cols, figsize=figsize)

    all_failed = True
    for obj_idx in range(n_obj):
        ax = ax_arr[obj_idx]
        try:
            # Per-objective target callable; default arg captures loop var
            target = lambda t, idx=obj_idx: (  # noqa: E731
                t.values[idx] if t.values and len(t.values) > idx
                else float("inf")
            )
            importance = optuna.importance.get_param_importances(
                study, target=target,
            )
        except Exception:
            ax.text(0.5, 0.5, "Importance unavailable",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(obj_cols[obj_idx])
            continue

        all_failed = False
        params = list(importance.keys())[:top_k]
        values = [importance[p] for p in params]

        y_pos = np.arange(len(params))
        ax.barh(y_pos, values, align="center", color=c.PRIMARY)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Importance ({obj_cols[obj_idx]})")

    if all_failed:
        if fig is not None:
            plt.close(fig)
        return None

    return fig if fig is not None else ax_arr


# ---------------------------------------------------------------------------
# 3-objective plot functions
# ---------------------------------------------------------------------------

def plot_pareto_3d(
    study: optuna.Study,
    ax: Any | None = None,
    *,
    cost_display: str = "color",
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
) -> Any:
    """3D scatter of all 3 objectives with Pareto front highlighted.

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study with 3 objectives.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        3D axes to draw on. Created if *None*.
    cost_display : ``"color"`` or ``"size"``
        How to display ``param_count`` user attr on the scatter:
        ``"color"`` maps log-param_count to a ``viridis`` colormap;
        ``"size"`` maps log-param_count to marker size (20--200).
    xlabel, ylabel, zlabel : str, optional
        Axis label overrides. Auto-derived from study metric names.
    """
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    obj_cols = _objective_column_names(study)
    if len(obj_cols) < 3:
        # text2D required for Axes3D (text() expects x, y, z, s)
        ax.text2D(
            0.5, 0.5, f"Need 3 objectives, got {len(obj_cols)}",
            ha="center", va="center", transform=ax.transAxes,
        )
        return ax

    trials = _valid_nobj_trials(study, 3)
    if not trials:
        ax.text2D(
            0.5, 0.5, "No trained trials with 3 valid objectives",
            ha="center", va="center", transform=ax.transAxes,
        )
        return ax

    v0 = [t.values[0] for t in trials]
    v1 = [t.values[1] for t in trials]
    v2 = [t.values[2] for t in trials]

    # param_count for cost_display
    param_counts = np.array(
        [t.user_attrs.get("param_count", 1) for t in trials], dtype=float,
    )
    log_pc = np.log1p(param_counts)

    if cost_display == "size":
        sizes = _normalize_to_sizes(log_pc)
        ax.scatter(
            v0, v1, v2,
            s=sizes, color=c.PRIMARY, alpha=c.ALPHA_TRIAL, label="Trials",
        )
    else:
        sc = ax.scatter(
            v0, v1, v2,
            c=log_pc, cmap="viridis", alpha=c.ALPHA_TRIAL, label="Trials",
        )
        ax.figure.colorbar(sc, ax=ax, label="log(param_count)", shrink=0.6,
                           pad=0.1)

    # Highlight Pareto front
    pareto_numbers = {t.number for t in study.best_trials}
    pareto_trials = [t for t in trials if t.number in pareto_numbers]
    if pareto_trials:
        pv0 = [t.values[0] for t in pareto_trials]
        pv1 = [t.values[1] for t in pareto_trials]
        pv2 = [t.values[2] for t in pareto_trials]
        ax.scatter(
            pv0, pv1, pv2,
            c=c.ACCENT, s=c.PARETO_SIZE, marker=c.PARETO_MARKER,
            label="Pareto", zorder=5,
        )

    ax.set_xlabel(xlabel or obj_cols[0])
    ax.set_ylabel(ylabel or obj_cols[1])
    ax.set_zlabel(zlabel or obj_cols[2])
    ax.set_title("3D Pareto front")
    ax.legend(loc="upper left")
    return ax


def plot_pareto_projections(
    study: optuna.Study,
    axes: Any | None = None,
    *,
    cost_display: str = "color",
    max_cols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Paired 2D Pareto projections for 3-objective studies.

    For each pair ``(i, j)`` in ``[(0,1), (0,2), (1,2)]``, shows a 2D
    scatter of ``objective[i]`` vs ``objective[j]``, with the omitted
    third dimension encoded via *cost_display*.

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study with 2 or 3 objectives.
    axes : array of matplotlib.axes.Axes, optional
        Pre-created axes array. Length must match number of pairs.
    cost_display : ``"color"`` or ``"size"``
        How to show the omitted third objective dimension.
    max_cols : int
        Maximum columns per row in standalone mode.
    figsize : tuple, optional
        Explicit figure size.  Auto-computed when *None*.
    """
    obj_cols = _objective_column_names(study)
    n_obj = len(obj_cols)

    if n_obj < 2:
        fig, ax_arr = _setup_grid(1, axes, max_cols=max_cols, figsize=figsize,
                                  panel_size=(6.0, 5.0))
        ax_arr[0].text(0.5, 0.5, "Need at least 2 objectives",
                       ha="center", va="center", transform=ax_arr[0].transAxes)
        return ax_arr

    # Build pairs: for 2-obj just one pair, for 3-obj three pairs
    if n_obj >= 3:
        pairs = [(0, 1), (0, 2), (1, 2)]
    else:
        pairs = [(0, 1)]

    fig, ax_arr = _setup_grid(len(pairs), axes, max_cols=max_cols,
                              figsize=figsize, panel_size=(6.0, 5.0))

    trials = _valid_nobj_trials(study, n_obj)
    pareto_numbers = {t.number for t in study.best_trials}

    for idx, (i, j) in enumerate(pairs):
        ax = ax_arr[idx]
        if not trials:
            ax.text(0.5, 0.5, "No valid trials",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        xi = [t.values[i] for t in trials]
        xj = [t.values[j] for t in trials]

        # Omitted dimension for 3-obj studies
        omitted_idx = ({0, 1, 2} - {i, j}).pop() if n_obj >= 3 else None

        if omitted_idx is not None:
            omitted_vals = np.array([t.values[omitted_idx] for t in trials])
            omitted_name = obj_cols[omitted_idx]
            if cost_display == "size":
                sizes = _normalize_to_sizes(omitted_vals)
                ax.scatter(xi, xj, s=sizes, color=c.PRIMARY,
                           alpha=c.ALPHA_TRIAL, label="Trials")
            else:
                sc = ax.scatter(xi, xj, c=omitted_vals, cmap="viridis",
                                alpha=0.6, label="Trials")
                plt.colorbar(sc, ax=ax, label=omitted_name)
        else:
            ax.scatter(xi, xj, color=c.PRIMARY, alpha=c.ALPHA_TRIAL,
                       label="Trials")

        # 2D Pareto step line (markers drawn separately from study Pareto)
        _draw_pareto_overlay(ax, xi, xj, draw_markers=False)

        # Highlight study-level Pareto trials
        pareto_in = [t for t in trials if t.number in pareto_numbers]
        if pareto_in:
            px = [t.values[i] for t in pareto_in]
            py = [t.values[j] for t in pareto_in]
            ax.scatter(px, py, c=c.ACCENT, s=c.PARETO_SIZE,
                       marker=c.PARETO_MARKER, zorder=4, label="Pareto")

        ax.set_xlabel(obj_cols[i])
        ax.set_ylabel(obj_cols[j])
        # Apply param_count formatter on axes showing param-related metrics
        if "param" in obj_cols[i].lower():
            ax.xaxis.set_major_formatter(_param_count_formatter())
        if "param" in obj_cols[j].lower():
            ax.yaxis.set_major_formatter(_param_count_formatter())
        ax.set_title(f"{obj_cols[i]} vs {obj_cols[j]}")
        ax.legend(loc="best", framealpha=0.8, fontsize="small")

    return ax_arr


def plot_parallel_coordinates(
    study: optuna.Study,
    ax: Any | None = None,
    *,
    top_k: int = 20,
    select_by: int = 0,
    metric_order: list[str] | None = None,
) -> Any:
    """Parallel coordinates plot of objectives for top-k trials.

    Each vertical axis represents one objective, normalized to [0, 1] for
    visual comparability. Lines are colored by the *select_by* objective.

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Created if *None*.
    top_k : int
        Number of best trials to display.
    select_by : int
        Objective index used for sorting and coloring.
    metric_order : list of str, optional
        Custom axis order. Defaults to ``_objective_column_names(study)``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    obj_cols = metric_order or _objective_column_names(study)
    n_axes = len(obj_cols)

    trained = _trained_trials(study)
    if not trained:
        ax.text(0.5, 0.5, "No trained trials",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    # Sort by select_by objective, take top_k
    trained = [
        t for t in trained
        if t.values and len(t.values) > select_by
    ]
    trained.sort(key=lambda t: t.values[select_by])
    trained = trained[:top_k]

    if not trained or n_axes < 2:
        ax.text(0.5, 0.5, "Insufficient data",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    # Build data matrix (trials × axes)
    data = np.array([
        [t.values[k] if k < len(t.values) else float("nan")
         for k in range(n_axes)]
        for t in trained
    ])

    # Invert and log-transform the cost metric (last axis) so that
    # "better" (lower cost) maps to higher normalized values, consistent
    # with the other axes where lower values are better and map to the
    # bottom of the plot.
    cost_col = n_axes - 1
    display_labels = list(obj_cols)
    data[:, cost_col] = np.log1p(np.abs(data[:, cost_col]))
    data[:, cost_col] = -data[:, cost_col]
    display_labels[cost_col] = f"-log({obj_cols[cost_col]})"

    # Normalize each axis to [0, 1]
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    normed = (data - col_min) / col_range

    # Color by select_by objective (use original data for non-cost axes)
    color_vals = data[:, select_by]
    val_min, val_max = color_vals.min(), color_vals.max()
    if val_max > val_min:
        normed_colors = (color_vals - val_min) / (val_max - val_min)
    else:
        normed_colors = np.zeros_like(color_vals)

    cmap = plt.cm.viridis_r
    x_ticks = np.arange(n_axes)

    for row_idx in range(len(trained)):
        color = cmap(normed_colors[row_idx])
        lw = 2.0 if row_idx < 3 else 0.8
        ax.plot(x_ticks, normed[row_idx], color=color, alpha=0.7,
                linewidth=lw)

    # Draw vertical axis lines and labels
    for i in range(n_axes):
        ax.axvline(i, color=c.SECONDARY, linewidth=0.5, alpha=0.5)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(display_labels, rotation=15, ha="right")
    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel coordinates")

    # Add tick labels showing actual range per axis
    for i in range(n_axes):
        ax.annotate(
            f"{col_min[i]:.3g}", (i, -0.05),
            ha="center", fontsize=7, color=c.SECONDARY,
            annotation_clip=False,
        )
        ax.annotate(
            f"{col_max[i]:.3g}", (i, 1.05),
            ha="center", fontsize=7, color=c.SECONDARY,
            annotation_clip=False,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=val_min, vmax=val_max),
    )
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, label=obj_cols[select_by])

    return ax


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def plot_study(
    study: optuna.Study,
    *,
    third_dim: str = "color",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Adaptive multi-row study overview figure.

    Produces a 3-row grid of sub-plots:

    - **Row 0 — Pareto**: all pairwise objective projections
    - **Row 1 — History**: per-objective best-so-far step lines
    - **Row 2 — Importance**: per-objective parameter importance
      (dropped if all panels fail)

    Parameters
    ----------
    study : optuna.Study
        Completed HPO study with 2 or 3 objectives.
    third_dim : ``"color"`` | ``"size"`` | ``"none"``
        Encoding for the omitted objective in Pareto projections
        (3-objective studies only; ignored for 2 objectives).
    figsize : tuple, optional
        Explicit ``(width, height)``.  Auto-computed as
        ``(5 * n_cols, 4.5 * n_rows)`` when *None*.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the study has fewer than 2 or more than 3 objectives.
        For >3 objectives, use the standalone functions directly
        (``plot_pareto_front``, ``plot_optimization_history``,
        ``plot_param_importance``) which support arbitrary counts
        via ``max_cols``.
    """
    from matplotlib.gridspec import GridSpec

    n_obj = len(study.directions)

    if n_obj < 2:
        raise ValueError(
            f"plot_study requires at least 2 objectives, got {n_obj}. "
            f"Use individual plot functions for single-objective studies."
        )
    if n_obj > 3:
        raise ValueError(
            f"plot_study supports 2-3 objectives, got {n_obj}. "
            f"For >3 objectives, use plot_pareto_front(), "
            f"plot_optimization_history(), and plot_param_importance() "
            f"directly with max_cols wrapping."
        )

    # Layout: n_pairs Pareto panels, n_obj history + importance panels
    n_pairs = n_obj * (n_obj - 1) // 2  # 1 for 2-obj, 3 for 3-obj
    n_cols = max(n_pairs, n_obj)
    n_rows = 3  # may shrink to 2 if importance fails

    if figsize is None:
        figsize = (5 * n_cols, 4.5 * n_rows)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig)

    # --- Row 0: Pareto projections ---
    # Center-span when fewer panels than columns (e.g. 1 pair in 2-col grid)
    if n_pairs < n_cols:
        # Single panel spanning all columns
        pareto_axes = np.array([fig.add_subplot(gs[0, :])])
    else:
        pareto_axes = np.array([fig.add_subplot(gs[0, col]) for col in range(n_cols)])
    plot_pareto_front(study, axes=pareto_axes, third_dim=third_dim)

    # --- Row 1: Optimization history (one per objective) ---
    history_axes = np.array([fig.add_subplot(gs[1, col]) for col in range(n_obj)])
    # Pad with hidden axes if n_obj < n_cols
    for col in range(n_obj, n_cols):
        ax = fig.add_subplot(gs[1, col])
        ax.set_visible(False)
    plot_optimization_history(study, axes=history_axes)

    # --- Row 2: Parameter importance (one per objective, may fail) ---
    importance_axes = np.array([fig.add_subplot(gs[2, col]) for col in range(n_obj)])
    row2_padding: list[Any] = []
    for col in range(n_obj, n_cols):
        ax = fig.add_subplot(gs[2, col])
        ax.set_visible(False)
        row2_padding.append(ax)
    result = plot_param_importance(study, axes=importance_axes)

    if result is None:
        # All importance panels failed — remove row 2 and shrink figure
        for ax in importance_axes:
            ax.remove()
        for ax in row2_padding:
            ax.remove()
        fig.set_size_inches(figsize[0], figsize[1] * 2 / 3)

    fig.tight_layout()
    return fig


