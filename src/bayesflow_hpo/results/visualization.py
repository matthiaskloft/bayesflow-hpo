"""Result visualizations for HPO studies."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna

from bayesflow_hpo.results.extraction import _objective_column_names


def plot_pareto_front(study: optuna.Study, ax: Any | None = None) -> Any:
    """Plot calibration error versus actual parameter count.

    Uses the ``param_count`` user attribute (actual trainable parameters)
    rather than the normalized objective value, which depends on the
    min/max normalization constants and can be misleading.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    obj_cols = _objective_column_names(study)
    if len(obj_cols) < 2:
        ax.text(0.5, 0.5, "Single-objective study",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    # Collect trained trials with actual param counts
    trained = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.values is not None
        and "rejected_reason" not in t.user_attrs
        and t.user_attrs.get("param_count", 0) > 0
    ]
    if not trained:
        ax.text(0.5, 0.5, "No trained trials",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    cal_errors = [t.values[0] for t in trained]
    param_counts = [t.user_attrs["param_count"] for t in trained]
    ax.scatter(cal_errors, param_counts, alpha=0.4, label="Trials")

    # Pareto front
    pareto = [
        t for t in study.best_trials
        if t.values is not None
        and "rejected_reason" not in t.user_attrs
        and t.user_attrs.get("param_count", 0) > 0
    ]
    if pareto:
        p_cal = [t.values[0] for t in pareto]
        p_params = [t.user_attrs["param_count"] for t in pareto]
        ax.scatter(p_cal, p_params, c="red", s=90, marker="*", label="Pareto")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda y, _: (
                f"{y / 1e6:.4g}M"
                if y >= 1e6
                else f"{y / 1e3:.4g}K"
                if y >= 1e3
                else f"{y:.4g}"
            )
        )
    )
    ax.set_xlabel("Calibration error")
    ax.set_ylabel("Parameter count")
    ax.set_title("Pareto front")
    ax.legend()
    return ax


def plot_param_importance(
    study: optuna.Study,
    ax: Any | None = None,
    top_k: int = 10,
) -> Any:
    """Plot Optuna parameter importances (single-objective-style)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        ax.text(
            0.5,
            0.5,
            "Importance unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
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
    ax.set_title("Parameter importance")
    return ax
