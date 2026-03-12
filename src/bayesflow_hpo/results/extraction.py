"""Extract tabular/Pareto results from Optuna studies.

Post-optimization analysis: convert completed trials into DataFrames,
extract Pareto-optimal trials, and generate human-readable summaries.

Design decision: budget-rejected trials are excluded from results by
default because their penalty objective values are not meaningful for
analysis.  Set ``trained_only=False`` in :func:`trials_to_dataframe`
to include them.
"""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd


def _fmt_param_count(count: int | float) -> str:
    """Format a raw parameter count as a human-readable string (e.g. ``"1.5M"``)."""
    count = int(count)
    if count >= 1_000_000:
        return f"{count / 1e6:.2f}M"
    if count >= 1_000:
        return f"{count / 1e3:.1f}K"
    return str(count)


def get_pareto_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Return Pareto-optimal trials from a multi-objective study."""
    return study.best_trials


def _objective_column_names(study: optuna.Study) -> list[str]:
    """Return objective column names, using ``study.metric_names`` when set.

    Falls back to ``"objective"`` (single) or ``"objective_0"``, … (multi)
    when metric names are unavailable (Optuna < 4.x or not configured).
    """
    metric_names: list[str] | None = getattr(study, "metric_names", None) or getattr(
        study, "_metric_names", None
    )
    n_objectives = len(study.directions)
    if metric_names and len(metric_names) == n_objectives:
        return list(metric_names)
    if n_objectives == 1:
        return ["objective"]
    return [f"objective_{i}" for i in range(n_objectives)]


# User attributes surfaced as columns by default in the results table.
DEFAULT_RESULT_ATTRS = [
    "param_count",
    "training_time_s",
    "inference_time_s",
    "inference_time_ratio",
    "calibration_error",
    "mean_cal_error",
    "nrmse",
    "correlation",
    "rmse",
    "contraction",
    "coverage_90",
    "coverage_95",
    "training_error",
    "rejected_reason",
]


def trials_to_dataframe(
    study: optuna.Study,
    trained_only: bool = True,
    include_pruned: bool = False,
    extra_attrs: list[str] | None = None,
) -> pd.DataFrame:
    """Convert study trials to a DataFrame.

    Objective columns are named after ``study.metric_names`` when set,
    otherwise ``"objective"`` (single-objective) or ``"objective_0"``,
    ``"objective_1"``, … (multi-objective).

    Parameters
    ----------
    study
        Optuna study.
    trained_only
        If ``True`` (default), exclude budget-rejected trials (those with
        a ``rejected_reason`` user attribute). Set to ``False`` to include
        all completed trials.
    include_pruned
        Whether to include pruned trials.
    extra_attrs
        Additional trial user-attribute keys to include as columns
        (beyond :data:`DEFAULT_RESULT_ATTRS`).
    """
    obj_cols = _objective_column_names(study)
    attr_keys = list(DEFAULT_RESULT_ATTRS)
    if extra_attrs:
        attr_keys.extend(k for k in extra_attrs if k not in attr_keys)

    records: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if trial.values is None:
                continue
            if trained_only and "rejected_reason" in trial.user_attrs:
                continue
            rec: dict[str, Any] = {"trial_number": trial.number, **trial.params}
            for col, val in zip(obj_cols, trial.values):
                rec[col] = val
            for attr_key in attr_keys:
                if attr_key in trial.user_attrs:
                    rec[attr_key] = trial.user_attrs[attr_key]
            records.append(rec)
        elif include_pruned and trial.state == optuna.trial.TrialState.PRUNED:
            records.append(
                {"trial_number": trial.number, "pruned": True, **trial.params}
            )
    return pd.DataFrame(records)


def summarize_study(
    study: optuna.Study,
    select_by: int = 0,
    top_k: int = 5,
) -> str:
    """Return a human-readable summary of an HPO study.

    Prints an overview of completed/pruned/failed trial counts, the Pareto
    front (for multi-objective studies), the single best trial selected by
    one objective, and a leaderboard of the top-k trials.

    Parameters
    ----------
    study
        Optuna study to summarize.
    select_by
        Index of the objective used to pick the "best" Pareto trial
        (default 0, typically the calibration error).
    top_k
        Number of top trials to show in the leaderboard.

    Returns
    -------
    str
        Formatted summary string (also printed to stdout).
    """
    obj_cols = _objective_column_names(study)
    n_objectives = len(study.directions)

    # --- trial counts ---
    states = {s: 0 for s in optuna.trial.TrialState}
    for t in study.trials:
        states[t.state] += 1

    n_complete = states[optuna.trial.TrialState.COMPLETE]
    n_pruned = states[optuna.trial.TrialState.PRUNED]
    n_failed = states[optuna.trial.TrialState.FAIL]
    n_trained = sum(
        1
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and "rejected_reason" not in t.user_attrs
        and t.values is not None
    )
    n_rejected = n_complete - n_trained

    lines: list[str] = []
    lines.append(f"Study: {study.study_name}")
    lines.append("=" * 60)
    lines.append(
        f"Trials: {len(study.trials)} total | "
        f"{n_trained} trained | {n_rejected} budget-rejected | "
        f"{n_pruned} pruned | {n_failed} failed"
    )
    lines.append(f"Objectives: {', '.join(obj_cols)}")
    lines.append("")

    # helper: format a single trial
    def _fmt_trial(trial: optuna.trial.FrozenTrial) -> list[str]:
        out: list[str] = []
        out.append(f"  Trial #{trial.number}")

        # Show all objective values with their column names.
        if trial.values:
            for col, val in zip(obj_cols, trial.values):
                out.append(f"    {col:25s}: {val:.4f}")

        # Show actual param count from user attrs (more reliable than
        # denormalizing the objective, which depends on min/max constants).
        raw_params = trial.user_attrs.get("param_count")
        if raw_params is not None and raw_params > 0:
            label = _fmt_param_count(raw_params)
            out.append(f"    {'Param count':25s}: {label}")

        # Key user attributes (skip those already shown above)
        attr_display = [
            ("training_time_s", "Training time (s)"),
            ("inference_time_s", "Inference time (s)"),
            ("inference_time_ratio", "Inference/sim ratio"),
            ("nrmse", "NRMSE"),
            ("correlation", "Correlation"),
            ("contraction", "Contraction"),
            ("coverage_90", "Coverage 90%"),
            ("coverage_95", "Coverage 95%"),
        ]
        for attr_key, label in attr_display:
            if attr_key in trial.user_attrs:
                v = trial.user_attrs[attr_key]
                if isinstance(v, float):
                    out.append(f"    {label:25s}: {v:.4f}")
                else:
                    out.append(f"    {label:25s}: {v}")
        return out

    # --- Pareto front (multi-objective) ---
    if n_objectives > 1 and n_trained > 0:
        # Filter to actually-trained Pareto trials (exclude budget-rejected
        # ones that got penalty scores).
        pareto = [
            t
            for t in study.best_trials
            if t.values is not None
            and "rejected_reason" not in t.user_attrs
        ]
        lines.append(f"Pareto front: {len(pareto)} trials")
        lines.append("-" * 60)
        if pareto:
            best = min(pareto, key=lambda t: t.values[select_by])
            lines.append(f"Best by {obj_cols[select_by]}:")
            lines.extend(_fmt_trial(best))
            lines.append("")

    # --- single-objective best ---
    elif n_objectives == 1 and n_trained > 0:
        best_trial = study.best_trial
        lines.append("Best trial:")
        lines.append("-" * 60)
        lines.extend(_fmt_trial(best_trial))
        lines.append("")

    # --- top-k leaderboard ---
    trained_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.values is not None
        and "rejected_reason" not in t.user_attrs
    ]
    if trained_trials:
        trained_trials.sort(key=lambda t: t.values[select_by])
        show = trained_trials[: min(top_k, len(trained_trials))]
        lines.append(f"Top {len(show)} trials (by {obj_cols[select_by]}):")
        lines.append("-" * 60)
        for t in show:
            parts = [f"#{t.number:>4d}"]
            # All objective values
            for col, val in zip(obj_cols, t.values):
                parts.append(f"{col}: {val:.4f}")
            # Actual param count from user attrs
            raw_params = t.user_attrs.get("param_count")
            if raw_params is not None and raw_params > 0:
                parts.append(f"params: {_fmt_param_count(raw_params)}")
            # Key metrics from user attrs (only if not already an objective)
            obj_set = set(obj_cols)
            for attr_key, label in [("nrmse", "nrmse"), ("correlation", "corr")]:
                if attr_key not in obj_set:
                    val = t.user_attrs.get(attr_key)
                    if val is not None:
                        parts.append(f"{label}: {val:.4f}")
            lines.append("  " + "  |  ".join(parts))
        lines.append("")

    # --- hyperparameters of best trial ---
    if n_trained > 0:
        ref = (
            min(
                [
                    t
                    for t in study.best_trials
                    if t.values is not None
                    and "rejected_reason" not in t.user_attrs
                ],
                key=lambda t: t.values[select_by],
            )
            if n_objectives > 1
            else study.best_trial
        )
        lines.append(f"Hyperparameters (trial #{ref.number}):")
        lines.append("-" * 60)
        for k, v in sorted(ref.params.items()):
            lines.append(f"  {k:35s}: {v}")

    summary = "\n".join(lines)
    print(summary)
    return summary
