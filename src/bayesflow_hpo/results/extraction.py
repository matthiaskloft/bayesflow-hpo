"""Extract tabular/pareto results from Optuna studies."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from bayesflow_hpo._display import DisplayDataFrame
from bayesflow_hpo.optimization.pruning_strategies import _non_dominated_sort

logger = logging.getLogger(__name__)

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


def _display_col_name(col: str) -> str:
    """Append a unit suffix to time-related column names for display."""
    if any(p in col.lower() for p in _TIME_PATTERNS) and not col.endswith("(s)"):
        return f"{col} (s)"
    return col


def _fmt_param_count(count: int | float) -> str:
    """Format a raw parameter count as a human-readable string."""
    count = int(count)
    if count >= 1_000_000:
        return f"{count / 1e6:.2f}M"
    if count >= 1_000:
        return f"{count / 1e3:.1f}K"
    return str(count)


def _validate_select_by(study: optuna.Study, select_by: int) -> None:
    """Raise ``ValueError`` if *select_by* is out of range for *study*."""
    n = len(study.directions)
    if not (0 <= select_by < n):
        raise ValueError(
            f"select_by={select_by} is out of range for a study with "
            f"{n} objective(s) (valid: 0..{n - 1})"
        )


def get_pareto_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Return Pareto-optimal trials from a multi-objective study."""
    return study.best_trials


# ---------------------------------------------------------------------------
# Lexicographic-Pareto trial selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionResult:
    """Metadata returned alongside the best trial from :func:`select_best_trial`.

    Parameters
    ----------
    thresholds_met : dict[str, bool]
        For each priority metric, whether any trial met the threshold.
    pareto_front : list[optuna.trial.FrozenTrial]
        The Pareto-optimal trials from Phase 2.
    n_candidates_per_step : list[int]
        Number of surviving candidates after each satisficing step.
    """

    thresholds_met: dict[str, bool]
    pareto_front: list[optuna.trial.FrozenTrial]
    n_candidates_per_step: list[int]


def _to_minimization_array(
    trials: list[optuna.trial.FrozenTrial],
    objective_indices: list[int],
    directions: list[optuna.study.StudyDirection],
) -> np.ndarray:
    """Build an (N, M) objective matrix where all columns are minimized.

    Maximize objectives are negated so that ``_non_dominated_sort``
    (which assumes minimization) and ascending ranking work correctly.
    """
    values = np.array(
        [[t.values[i] for i in objective_indices] for t in trials]
    )
    for col, idx in enumerate(objective_indices):
        if directions[idx] == optuna.study.StudyDirection.MAXIMIZE:
            values[:, col] *= -1
    return values


def _pareto_front_trials(
    trials: list[optuna.trial.FrozenTrial],
    objective_indices: list[int],
    directions: list[optuna.study.StudyDirection],
) -> list[optuna.trial.FrozenTrial]:
    """Return Pareto-optimal trials over the given objective indices.

    Maximize objectives are negated internally so that
    ``_non_dominated_sort`` (Deb et al., 2002) can assume minimization.
    """
    if not trials:
        return []
    if len(objective_indices) == 1:
        idx = objective_indices[0]
        if directions[idx] == optuna.study.StudyDirection.MAXIMIZE:
            best = max(trials, key=lambda t: t.values[idx])
        else:
            best = min(trials, key=lambda t: t.values[idx])
        return [best]

    objectives = _to_minimization_array(trials, objective_indices, directions)
    fronts = _non_dominated_sort(objectives)
    return [trials[i] for i in fronts[0]]


def _mean_rank_trial(
    trials: list[optuna.trial.FrozenTrial],
    objective_indices: list[int],
    directions: list[optuna.study.StudyDirection],
) -> optuna.trial.FrozenTrial:
    """Return the trial with the lowest mean rank across *objective_indices*.

    Ranking is ascending for minimize objectives (lower is better) and
    descending for maximize objectives (higher is better), via negation.
    """
    n = len(trials)
    if n == 1:
        return trials[0]

    values = _to_minimization_array(trials, objective_indices, directions)
    # scipy-free ranking: argsort of argsort gives 0-based ranks.
    ranks = np.zeros_like(values)
    for col in range(values.shape[1]):
        order = np.argsort(values[:, col])
        ranks[order, col] = np.arange(n)
    mean_ranks = ranks.mean(axis=1)
    return trials[int(np.argmin(mean_ranks))]


def _get_metric_value(
    trial: optuna.trial.FrozenTrial,
    metric: str,
    obj_names: list[str],
) -> float | None:
    """Look up a metric value from trial objectives or user_attrs."""
    if metric in obj_names:
        idx = obj_names.index(metric)
        if trial.values and idx < len(trial.values):
            return trial.values[idx]
        return None
    return trial.user_attrs.get(metric)


def select_best_trial(
    study: optuna.Study,
    priorities: list[tuple[str, float] | tuple[str, float, str]],
) -> tuple[optuna.trial.FrozenTrial, SelectionResult]:
    """Select the best trial using lexicographic-Pareto selection.

    Two-phase algorithm:

    **Phase 1 — Satisficing:** Walk priority metrics in order. For each
    ``(metric, threshold)`` pair, filter candidates to those meeting the
    threshold. If no candidate meets it, warn and promote that metric (and
    all subsequent ones) to Phase 2.

    **Phase 2 — Pareto selection:** Among survivors, compute the Pareto
    front over remaining study objectives and return the trial with the
    lowest mean rank.

    Parameters
    ----------
    study
        Optuna study with at least one trained (non-rejected) trial.
    priorities
        Ordered list of ``(metric, threshold)`` or
        ``(metric, threshold, direction)`` tuples. *direction* is
        ``"below"`` (value <= threshold) or ``"above"`` (value >= threshold).
        For study objectives the direction is inferred from
        ``study.directions``; for user_attrs it must be explicit.

    Returns
    -------
    tuple[optuna.trial.FrozenTrial, SelectionResult]
        The best trial and selection metadata.

    Raises
    ------
    ValueError
        If the study has no trained trials, a metric name is not found,
        or a user_attr metric lacks an explicit direction.
    """
    trained = _get_trained_trials(study)
    if not trained:
        raise ValueError("Study has no trained trials")

    obj_names = _objective_column_names(study)
    n_objectives = len(study.directions)

    # Parse priorities and resolve directions.
    parsed: list[tuple[str, float, str]] = []
    for entry in priorities:
        if len(entry) == 3:
            metric, threshold, direction = entry
            if direction not in ("below", "above"):
                raise ValueError(
                    f"direction must be 'below' or 'above', got {direction!r}"
                )
        elif len(entry) == 2:
            metric, threshold = entry
            direction = None
        else:
            raise ValueError(f"Expected 2- or 3-tuple, got {entry!r}")

        # Resolve metric source and direction.
        if metric in obj_names:
            idx = obj_names.index(metric)
            if direction is None:
                d = study.directions[idx]
                direction = (
                    "below"
                    if d == optuna.study.StudyDirection.MINIMIZE
                    else "above"
                )
        else:
            # Must be a user_attr — check it exists in at least one trial.
            if not any(metric in t.user_attrs for t in trained):
                available = sorted(
                    set(obj_names)
                    | {k for t in trained for k in t.user_attrs}
                )
                raise ValueError(
                    f"Metric {metric!r} not found in any trial. "
                    f"Available: {available}"
                )
            if direction is None:
                raise ValueError(
                    f"Metric {metric!r} is a user_attr, not a study objective. "
                    f"Specify direction explicitly as 'below' or 'above'."
                )
        parsed.append((metric, threshold, direction))

    # Phase 1 — Satisficing.
    candidates = list(trained)
    thresholds_met: dict[str, bool] = {}
    n_candidates_per_step: list[int] = [len(candidates)]
    promoted_from: int | None = None  # index where promotion starts

    for i, (metric, threshold, direction) in enumerate(parsed):
        if direction == "below":
            filtered = [
                t for t in candidates
                if (v := _get_metric_value(t, metric, obj_names)) is not None
                and v <= threshold
            ]
        else:
            filtered = [
                t for t in candidates
                if (v := _get_metric_value(t, metric, obj_names)) is not None
                and v >= threshold
            ]

        if filtered:
            candidates = filtered
            thresholds_met[metric] = True
        else:
            logger.warning(
                "No trial meets threshold %s %s %.4g; "
                "promoting remaining priorities to Pareto selection.",
                metric,
                "<=" if direction == "below" else ">=",
                threshold,
            )
            thresholds_met[metric] = False
            for remaining_metric, _, _ in parsed[i + 1:]:
                thresholds_met[remaining_metric] = False
            promoted_from = i
            break

        n_candidates_per_step.append(len(candidates))

    # Determine which study objectives go to Phase 2.
    # "Remaining" = objectives with no threshold in priorities, or promoted.
    priority_obj_names = {m for m, _, _ in parsed}
    if promoted_from is not None:
        promoted_names = {m for m, _, _ in parsed[promoted_from:]}
    else:
        promoted_names = set()

    remaining_indices: list[int] = []
    for idx in range(n_objectives):
        name = obj_names[idx]
        if name not in priority_obj_names or name in promoted_names:
            remaining_indices.append(idx)

    # If all objectives had thresholds and all were met, use all objectives.
    if not remaining_indices:
        remaining_indices = list(range(n_objectives))

    # Phase 2 — Pareto selection.
    dirs = list(study.directions)
    pareto = _pareto_front_trials(candidates, remaining_indices, dirs)
    best = _mean_rank_trial(candidates, remaining_indices, dirs)

    if len(candidates) == 1:
        unmet = [m for m, met in thresholds_met.items() if not met]
        if unmet:
            logger.warning(
                "Single surviving trial #%d does not meet thresholds for: %s",
                candidates[0].number,
                ", ".join(unmet),
            )

    return best, SelectionResult(
        thresholds_met=thresholds_met,
        pareto_front=pareto,
        n_candidates_per_step=n_candidates_per_step,
    )


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
    include_ranks: bool = True,
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
    include_ranks
        If ``True`` (default), append rank columns derived from objective
        values. For single-objective studies this adds ``"rank"``. For
        multi-objective studies this adds one column per objective,
        e.g. ``"rank_sbc_c2st"``, and ``"rank"`` as the rank for the
        first objective.
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
    df = pd.DataFrame(records)

    if include_ranks and not df.empty:
        if len(obj_cols) == 1:
            col = obj_cols[0]
            if col in df.columns:
                df["rank"] = df[col].rank(method="min", ascending=True)
        else:
            for col in obj_cols:
                if col in df.columns:
                    rank_col = f"rank_{col}"
                    df[rank_col] = df[col].rank(method="min", ascending=True)

            first_rank_col = f"rank_{obj_cols[0]}"
            if first_rank_col in df.columns:
                df["rank"] = df[first_rank_col]

        if "rank" in df.columns:
            df["rank"] = df["rank"].astype("Int64")
        for col in obj_cols:
            rank_col = f"rank_{col}"
            if rank_col in df.columns:
                df[rank_col] = df[rank_col].astype("Int64")

    return df


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
    _validate_select_by(study, select_by)
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

    return DisplayDataFrame(records)


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
    priorities: list[tuple[str, float] | tuple[str, float, str]] | None = None,
) -> dict[str, Any]:
    """Return the hyperparameter config of a trial, with rounded values.

    Parameters
    ----------
    study
        Optuna study.
    trial_number
        Specific trial to retrieve.  If ``None`` (default), the best
        trained trial by *select_by* or *priorities* is used.
    select_by
        Index of the objective used to pick the best trial when
        *trial_number* is ``None`` and *priorities* is ``None``.
    priorities
        Lexicographic-Pareto priorities for :func:`select_best_trial`.
        Mutually exclusive with a non-default *select_by*.

    Returns
    -------
    dict[str, Any]
        Hyperparameter name → rounded value mapping.

    Raises
    ------
    ValueError
        If *trial_number* does not exist in the study, if the study
        has no trained trials, or if both *select_by* and *priorities*
        are specified.
    """
    if priorities is not None and select_by != 0:
        raise ValueError(
            "Cannot specify both 'priorities' and a non-default 'select_by'. "
            "Use one or the other."
        )

    if trial_number is not None:
        trial = _find_trial(study, trial_number)
    elif priorities is not None:
        trial, _result = select_best_trial(study, priorities)
    else:
        _validate_select_by(study, select_by)
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
    _validate_select_by(study, select_by)
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
            display_col = _display_col_name(col)
            lines.append(f"  {display_col:30s}: {_round_value(col, val)}")
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
