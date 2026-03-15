"""Extract tabular/pareto results from Optuna studies."""

from __future__ import annotations

import math
from typing import Any

import optuna
import pandas as pd

# Patterns used by _round_value() to decide how to format floats.
_LR_PATTERNS = ("lr", "learning_rate", "initial_lr")
_DROPOUT_PATTERNS = ("dropout",)
_INT_PATTERNS = ("dim", "width", "depth", "units", "heads", "layers")
_TIME_PATTERNS = ("time",)


def _round_value(key: str, value: Any) -> Any:
    """Round a hyperparameter value based on its key name.

    Parameters
    ----------
    key
        Hyperparameter or metric name (e.g. ``"initial_lr"``, ``"ds_dropout"``).
    value
        The raw value to format.

    Returns
    -------
    Any
        Rounded value: scientific notation string for learning rates,
        2-decimal float for dropouts, int for architectural dims,
        1-decimal float for times, 4-decimal float for other floats,
        or the original value for non-floats.
    """
    if not isinstance(value, float):
        return value
    if math.isnan(value) or math.isinf(value):
        return value
    low = key.lower()
    if any(p in low for p in _LR_PATTERNS):
        return f"{value:.2e}"
    if any(p in low for p in _DROPOUT_PATTERNS):
        return round(value, 2)
    if any(p in low for p in _INT_PATTERNS):
        return int(value)
    if any(p in low for p in _TIME_PATTERNS):
        return round(value, 1)
    return round(value, 4)


def _fmt_param_count(count: int | float) -> str:
    """Format a raw parameter count as a human-readable string."""
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
    """Return objective column names, using study.metric_names when set."""
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


def _get_trained_trials(
    study: optuna.Study,
    trained_only: bool = True,
) -> list[optuna.trial.FrozenTrial]:
    """Return completed trials, optionally filtering out budget-rejected ones."""
    trials = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.values is None:
            continue
        if trained_only and "rejected_reason" in t.user_attrs:
            continue
        trials.append(t)
    return trials


def trial_table(
    study: optuna.Study,
    top_k: int | None = None,
    select_by: int = 0,
    metrics: list[str] | None = None,
    trained_only: bool = True,
) -> pd.DataFrame:
    """Return a ranked DataFrame of the best trials.

    Unlike :func:`trials_to_dataframe` (which returns raw, unranked data),
    this function sorts trials by a chosen objective, applies top-k
    filtering, and rounds all float values for clean display.  The result
    is ready for ``.to_csv()`` or ``.to_markdown()`` export.

    Parameters
    ----------
    study
        Optuna study.
    top_k
        Maximum number of trials to include.  ``None`` (default) returns
        all trained trials.
    select_by
        Index of the objective used to rank trials (default 0).
    metrics
        Extra user-attribute keys to include as columns (e.g.
        ``["nrmse", "correlation"]``).
    trained_only
        If ``True`` (default), exclude budget-rejected trials.

    Returns
    -------
    pd.DataFrame
        Ranked table with columns: ``rank``, ``trial``, objective columns,
        ``param_count``, hyperparameters, and any requested metrics.
    """
    obj_cols = _objective_column_names(study)
    trials = _get_trained_trials(study, trained_only=trained_only)

    if not trials:
        return pd.DataFrame()

    # Sort by the chosen objective.
    trials.sort(key=lambda t: t.values[select_by])

    if top_k is not None:
        trials = trials[:top_k]

    records: list[dict[str, Any]] = []
    for rank, trial in enumerate(trials, 1):
        rec: dict[str, Any] = {"rank": rank, "trial": trial.number}

        # Objective values.
        for col, val in zip(obj_cols, trial.values):
            rec[col] = _round_value(col, val)

        # Param count.
        raw_params = trial.user_attrs.get("param_count")
        if raw_params is not None:
            rec["param_count"] = _fmt_param_count(raw_params)

        # Hyperparameters.
        for k, v in sorted(trial.params.items()):
            rec[k] = _round_value(k, v)

        # Extra metrics.
        if metrics:
            for m in metrics:
                if m in trial.user_attrs:
                    rec[m] = _round_value(m, trial.user_attrs[m])

        records.append(rec)

    return pd.DataFrame(records)


def _find_trial(
    study: optuna.Study,
    trial_number: int,
) -> optuna.trial.FrozenTrial:
    """Look up a trial by number, raising ValueError if not found."""
    for t in study.trials:
        if t.number == trial_number:
            return t
    raise ValueError(
        f"Trial #{trial_number} not found in study '{study.study_name}'"
    )


def best_config(
    study: optuna.Study,
    trial_number: int | None = None,
    select_by: int = 0,
) -> dict[str, Any]:
    """Return the hyperparameter config of a trial, with rounded values.

    Parameters
    ----------
    study
        Optuna study.
    trial_number
        Specific trial to retrieve.  If ``None`` (default), the best
        trained trial by *select_by* objective is used.
    select_by
        Index of the objective used to pick the best trial when
        *trial_number* is ``None``.

    Returns
    -------
    dict[str, Any]
        Hyperparameter name → rounded value mapping.

    Raises
    ------
    ValueError
        If *trial_number* does not exist in the study, or if the study
        has no trained trials.
    """
    if trial_number is not None:
        trial = _find_trial(study, trial_number)
    else:
        trained = _get_trained_trials(study)
        if not trained:
            raise ValueError("Study has no trained trials")
        trial = min(trained, key=lambda t: t.values[select_by])

    config = {k: _round_value(k, v) for k, v in sorted(trial.params.items())}

    # Print a formatted block.
    header = f"Hyperparameters (trial #{trial.number})"
    print(header)
    print("-" * len(header))
    for k, v in config.items():
        print(f"  {k:35s}: {v}")

    return config


def compare_trials(
    study: optuna.Study,
    trial_numbers: list[int],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compare 2–5 trials side by side.

    Returns a DataFrame where rows are hyperparameters, objectives,
    and optional metrics, and columns are the requested trial numbers.

    Parameters
    ----------
    study
        Optuna study.
    trial_numbers
        List of 2–5 trial numbers to compare.
    metrics
        Extra user-attribute keys to include as rows.

    Returns
    -------
    pd.DataFrame
        Comparison table with trial numbers as columns.

    Raises
    ------
    ValueError
        If fewer than 2 or more than 5 trial numbers are given,
        or if a trial number is not found.
    """
    if len(trial_numbers) < 2:
        raise ValueError("Need at least 2 trials to compare")
    if len(trial_numbers) > 5:
        raise ValueError("At most 5 trials can be compared")

    obj_cols = _objective_column_names(study)
    trials = [_find_trial(study, n) for n in trial_numbers]

    # Collect all hyperparameter keys across all trials.
    all_param_keys: list[str] = []
    seen: set[str] = set()
    for t in trials:
        for k in sorted(t.params):
            if k not in seen:
                all_param_keys.append(k)
                seen.add(k)

    # Build row labels: objectives, then param_count, then hyperparams, then metrics.
    row_labels: list[str] = list(obj_cols) + ["param_count"] + all_param_keys
    if metrics:
        row_labels.extend(m for m in metrics if m not in seen)

    data: dict[str, list[Any]] = {}
    for t in trials:
        col: list[Any] = []
        # Objectives.
        for i, name in enumerate(obj_cols):
            val = t.values[i] if t.values and i < len(t.values) else None
            col.append(_round_value(name, val) if val is not None else None)
        # Param count.
        raw = t.user_attrs.get("param_count")
        col.append(_fmt_param_count(raw) if raw is not None else None)
        # Hyperparameters.
        for k in all_param_keys:
            val = t.params.get(k)
            col.append(_round_value(k, val) if val is not None else None)
        # Metrics.
        if metrics:
            for m in metrics:
                if m not in seen:
                    val = t.user_attrs.get(m)
                    col.append(_round_value(m, val) if val is not None else None)
        data[f"trial_{t.number}"] = col

    return pd.DataFrame(data, index=row_labels)


def summarize_study(
    study: optuna.Study,
    select_by: int = 0,
) -> str:
    """Return a compact summary of an HPO study.

    Prints trial counts, objectives, and the best trial's scores.
    For detailed results use :func:`trial_table`; for hyperparameters
    use :func:`best_config`.

    Parameters
    ----------
    study
        Optuna study to summarize.
    select_by
        Index of the objective used to pick the "best" trial
        (default 0, typically the calibration error).

    Returns
    -------
    str
        Formatted summary string (also printed to stdout).
    """
    obj_cols = _objective_column_names(study)
    n_objectives = len(study.directions)
    trained = _get_trained_trials(study)
    n_trained = len(trained)

    states = {s: 0 for s in optuna.trial.TrialState}
    for t in study.trials:
        states[t.state] += 1
    n_complete = states[optuna.trial.TrialState.COMPLETE]
    n_pruned = states[optuna.trial.TrialState.PRUNED]
    n_failed = states[optuna.trial.TrialState.FAIL]
    n_rejected = n_complete - n_trained

    lines: list[str] = [
        f"Study: {study.study_name}",
        "=" * 60,
        (
            f"Trials: {len(study.trials)} total | "
            f"{n_trained} trained | {n_rejected} rejected | "
            f"{n_pruned} pruned | {n_failed} failed"
        ),
        f"Objectives: {', '.join(obj_cols)}",
        "",
    ]

    if n_trained > 0:
        best = min(trained, key=lambda t: t.values[select_by])

        # Pareto info (multi-objective only).
        if n_objectives > 1:
            pareto = [
                t
                for t in study.best_trials
                if t.values is not None
                and "rejected_reason" not in t.user_attrs
            ]
            lines.append(f"Pareto front: {len(pareto)} trials")

        lines.append(f"Best trial: #{best.number}")
        for col, val in zip(obj_cols, best.values):
            lines.append(f"  {col:30s}: {_round_value(col, val)}")
        raw_params = best.user_attrs.get("param_count")
        if raw_params is not None:
            lines.append(
                f"  {'param_count':30s}: {_fmt_param_count(raw_params)}"
            )
        lines.append("")
        lines.append(
            "Use trial_table() for detailed results, "
            "best_config() for hyperparameters."
        )

    summary = "\n".join(lines)
    print(summary)
    return summary
