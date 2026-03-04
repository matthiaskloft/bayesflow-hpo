"""Result visualizations for HPO studies."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna

from bayesflow_hpo.objectives import denormalize_param_count
from bayesflow_hpo.results.extraction import trials_to_dataframe


def plot_pareto_front(study: optuna.Study, ax: Any | None = None) -> Any:
    """Plot calibration error versus parameter count."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    trials_df = trials_to_dataframe(study)
    if "objective_0" not in trials_df.columns or "objective_1" not in trials_df.columns:
        ax.text(0.5, 0.5, "Single-objective study", ha="center", va="center", transform=ax.transAxes)
        return ax

    denorm_params = trials_df["objective_1"].apply(denormalize_param_count)
    ax.scatter(trials_df["objective_0"], denorm_params, alpha=0.4, label="Trials")

    pareto = study.best_trials
    pareto_cal = [t.values[0] for t in pareto if t.values is not None]
    pareto_params = [denormalize_param_count(t.values[1]) for t in pareto if t.values is not None]
    if pareto_cal:
        ax.scatter(pareto_cal, pareto_params, c="red", s=90, marker="*", label="Pareto")

    ax.set_xlabel("Calibration error")
    ax.set_ylabel("Parameter count")
    ax.set_title("Pareto front")
    ax.legend()
    return ax


def plot_param_importance(study: optuna.Study, ax: Any | None = None, top_k: int = 10) -> Any:
    """Plot Optuna parameter importances (single-objective-style)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        ax.text(0.5, 0.5, "Importance unavailable", ha="center", va="center", transform=ax.transAxes)
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
