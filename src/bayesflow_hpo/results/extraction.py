"""Extract tabular/pareto results from Optuna studies."""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

import numpy as np
import optuna
import pandas as pd

from bayesflow_hpo._display import DisplayDataFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority type alias and SelectionResult
# ---------------------------------------------------------------------------

PrioritySpec = tuple[str, float] | tuple[str, float, str]
"""A satisficing priority: ``(metric, threshold)`` or
``(metric, threshold, "below" | "above")``.

For study objectives the direction is inferred from ``study.directions``
when the 2-tuple form is used.  For user attributes that are not study
objectives the 3-tuple form with an explicit ``"below"`` or ``"above"``
direction is required.
"""


@dataclasses.dataclass(frozen=True)
class SelectionResult:
    """Diagnostic output from :func:`select_best_trial`.

    Attributes
    ----------
    thresholds_met : dict[str, bool]
        Maps each priority metric name to whether its threshold was
        satisfied by at least one candidate.
    pareto_front : list[optuna.trial.FrozenTrial]
        The Pareto-optimal trials from Phase 2.
    n_candidates_per_step : list[int]
        Number of surviving candidates after each satisficing filter step.
    """

    thresholds_met: dict[str, bool]
    pareto_front: list[optuna.trial.FrozenTrial]
    n_candidates_per_step: list[int]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_select_by(study: optuna.Study, select_by: int) -> None:
    """Raise ``ValueError`` if *select_by* is out of range."""
    n = len(study.directions)
    if not (0 <= select_by < n):
        raise ValueError(
            f"select_by={select_by} is out of range for a study with "
            f"{n} objective(s) (must be 0 <= select_by < {n})"
        )


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

    # Sort by the chosen objective (ascending for minimize, descending
    # for maximize so that rank 1 = best).
    ascending = study.directions[select_by] == optuna.study.StudyDirection.MINIMIZE
    trials.sort(key=lambda t: t.values[select_by], reverse=not ascending)

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
    priorities: list[PrioritySpec] | None = None,
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
        *trial_number* is ``None`` and *priorities* is ``None``.
    priorities
        Optional list of satisficing priorities for lexicographic-Pareto
        selection.  Each element is a 2-tuple ``(metric, threshold)``
        (direction inferred from study) or 3-tuple
        ``(metric, threshold, "below" | "above")``.  When provided,
        :func:`select_best_trial` is used instead of simple
        single-objective selection.  Mutually exclusive with a non-default
        *select_by*.

    Returns
    -------
    dict[str, Any]
        Hyperparameter name → rounded value mapping.

    Raises
    ------
    ValueError
        If *trial_number* does not exist in the study, if the study
        has no trained trials, or if both *priorities* and a non-default
        *select_by* are provided.
    """
    if priorities is not None and select_by != 0:
        raise ValueError(
            "Cannot specify both 'priorities' and a non-default "
            "'select_by'; they are mutually exclusive."
        )

    if trial_number is not None:
        trial = _find_trial(study, trial_number)
    elif priorities is not None:
        trial, _ = select_best_trial(study, priorities)
    else:
        _validate_select_by(study, select_by)
        trained = _get_trained_trials(study)
        if not trained:
            raise ValueError("Study has no trained trials")
        direction = study.directions[select_by]
        use_min = direction == optuna.study.StudyDirection.MINIMIZE
        trial = (min if use_min else max)(
            trained, key=lambda t: t.values[select_by]
        )

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
        direction = study.directions[select_by]
        use_min = direction == optuna.study.StudyDirection.MINIMIZE
        best = (min if use_min else max)(
            trained, key=lambda t: t.values[select_by]
        )

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


# ---------------------------------------------------------------------------
# Lexicographic-Pareto trial selection
# ---------------------------------------------------------------------------


def _resolve_priority(
    priority: PrioritySpec,
    obj_cols: list[str],
    directions: list[optuna.study.StudyDirection],
) -> tuple[str, float, bool]:
    """Validate and resolve a priority tuple.

    Returns ``(metric, threshold, is_below)`` where *is_below* is ``True``
    when candidates must have value ≤ threshold.
    """
    if len(priority) == 2:
        metric, threshold = priority
        if metric not in obj_cols:
            raise ValueError(
                f"Priority metric {metric!r} is not a study objective "
                f"({obj_cols}); specify an explicit direction as a 3-tuple "
                f'e.g. ({metric!r}, {threshold}, "below")'
            )
        idx = obj_cols.index(metric)
        is_below = directions[idx] == optuna.study.StudyDirection.MINIMIZE
    elif len(priority) == 3:
        metric, threshold, direction = priority
        if direction not in ("below", "above"):
            raise ValueError(
                f"Direction must be 'below' or 'above', got {direction!r}"
            )
        is_below = direction == "below"
    else:
        raise ValueError(
            f"Priority must be a 2- or 3-tuple, got length {len(priority)}"
        )
    return metric, float(threshold), is_below


def _get_trial_metric(
    trial: optuna.trial.FrozenTrial,
    metric: str,
    obj_cols: list[str],
) -> float | None:
    """Extract a metric value from a trial.

    Looks up *metric* first in ``trial.values`` (if it matches a study
    objective name), then in ``trial.user_attrs``.  Returns ``None`` if
    the metric is not available on this trial.
    """
    if metric in obj_cols and trial.values is not None:
        idx = obj_cols.index(metric)
        if idx < len(trial.values):
            return trial.values[idx]
    if metric in trial.user_attrs:
        val = trial.user_attrs[metric]
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _pareto_front_indices(
    values: np.ndarray,
    minimize: list[bool],
) -> list[int]:
    """Return row indices of non-dominated solutions.

    Uses the dominance relation from NSGA-II (Deb et al., 2002).
    Objectives are flipped so that all are treated as minimize.

    Parameters
    ----------
    values
        ``(N, M)`` array of objective values.
    minimize
        Per-objective flag; ``True`` means minimize, ``False`` means
        maximize.

    Returns
    -------
    list[int]
        Indices of Pareto-optimal rows.
    """
    n, m = values.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    # Single objective: just return the index of the best value.
    if m == 1:
        col = values[:, 0]
        best = int(np.argmin(col)) if minimize[0] else int(np.argmax(col))
        return [best]

    transformed = values.copy()
    for j, is_min in enumerate(minimize):
        if not is_min:
            transformed[:, j] = -transformed[:, j]

    front: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if np.all(transformed[j] <= transformed[i]) and np.any(
                transformed[j] < transformed[i]
            ):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def _available_metric_names(
    trials: list[optuna.trial.FrozenTrial],
    obj_cols: list[str],
) -> list[str]:
    """Collect all metric names available across trials."""
    names = list(obj_cols)
    seen = set(names)
    for t in trials:
        for k in sorted(t.user_attrs):
            if k not in seen and isinstance(t.user_attrs[k], (int, float)):
                names.append(k)
                seen.add(k)
    return names


def select_best_trial(
    study: optuna.Study,
    priorities: list[PrioritySpec],
) -> tuple[optuna.trial.FrozenTrial, SelectionResult]:
    """Select the best trial using lexicographic-Pareto selection.

    Two-phase algorithm designed for multi-objective HPO studies:

    **Phase 1 — Satisficing:** Walk through priority metrics in order.
    For each ``(metric, threshold)``, filter candidates to those meeting
    the threshold.  If no candidate meets a threshold, warn and promote
    that metric and all subsequent ones to Phase 2.

    **Phase 2 — Pareto selection:** Among surviving candidates, compute
    the Pareto front over remaining *study objectives* (objectives that
    had no threshold or were promoted).  Return the trial with the
    lowest mean rank across those objectives.

    Direction inference: for metrics matching study objective names,
    direction is inferred from ``study.directions``.  For user attributes
    the 3-tuple form ``(metric, threshold, "below" | "above")`` is
    required.

    Parameters
    ----------
    study
        Optuna study (single- or multi-objective).
    priorities
        Ordered list of satisficing criteria.  Each element is either a
        2-tuple ``(metric, threshold)`` where direction is inferred from
        the study, or a 3-tuple ``(metric, threshold, "below" | "above")``
        for explicit direction.

    Returns
    -------
    tuple[optuna.trial.FrozenTrial, SelectionResult]
        The selected trial and diagnostic metadata.

    Raises
    ------
    ValueError
        If the study has no trained trials, a metric name is not found
        in any trial, or a 2-tuple priority names a non-objective metric.

    References
    ----------
    Pareto dominance follows the non-dominated sorting relation from
    NSGA-II (Deb et al., 2002).
    """
    obj_cols = _objective_column_names(study)
    directions = list(study.directions)
    candidates = _get_trained_trials(study)

    if not candidates:
        raise ValueError("Study has no trained trials")

    if not priorities:
        raise ValueError("priorities must be a non-empty list")

    # -- Validate all priorities upfront --
    resolved: list[tuple[str, float, bool]] = []
    for p in priorities:
        metric, threshold, is_below = _resolve_priority(p, obj_cols, directions)
        # Verify the metric exists on at least one trial.
        found = any(
            _get_trial_metric(t, metric, obj_cols) is not None for t in candidates
        )
        if not found:
            available = _available_metric_names(candidates, obj_cols)
            raise ValueError(
                f"Priority metric {metric!r} not found in any trial. "
                f"Available metrics: {available}"
            )
        resolved.append((metric, threshold, is_below))

    # -- Phase 1: Satisficing filter --
    thresholds_met: dict[str, bool] = {}
    n_candidates_per_step: list[int] = [len(candidates)]
    promoted_from: int | None = None  # index where promotion starts

    for i, (metric, threshold, is_below) in enumerate(resolved):
        filtered = []
        for t in candidates:
            val = _get_trial_metric(t, metric, obj_cols)
            if val is None:
                continue
            if is_below and val <= threshold:
                filtered.append(t)
            elif not is_below and val >= threshold:
                filtered.append(t)

        if filtered:
            thresholds_met[metric] = True
            candidates = filtered
        else:
            thresholds_met[metric] = False
            promoted_from = i
            logger.warning(
                "No trial meets threshold for %r (threshold=%.4g); "
                "promoting to Pareto selection.",
                metric,
                threshold,
            )
            # Mark all subsequent as not met too.
            for j in range(i + 1, len(resolved)):
                thresholds_met[resolved[j][0]] = False
            break

        n_candidates_per_step.append(len(candidates))

    # -- Phase 2: Pareto selection over remaining study objectives --
    # "Remaining" = study objectives with no threshold OR promoted.
    priority_metrics = {r[0] for r in resolved}
    promoted_metrics = set()
    if promoted_from is not None:
        promoted_metrics = {resolved[j][0] for j in range(promoted_from, len(resolved))}

    # Remaining study objectives for Pareto: those not in priorities,
    # or those that were promoted.
    remaining_obj_indices: list[int] = []
    for idx, col in enumerate(obj_cols):
        if col not in priority_metrics or col in promoted_metrics:
            remaining_obj_indices.append(idx)

    # If all thresholds met and no remaining objectives, use all study objectives.
    if not remaining_obj_indices:
        remaining_obj_indices = list(range(len(obj_cols)))

    # Single candidate → done.
    if len(candidates) == 1:
        return candidates[0], SelectionResult(
            thresholds_met=thresholds_met,
            pareto_front=list(candidates),
            n_candidates_per_step=n_candidates_per_step,
        )

    # Build objective matrix for remaining objectives.
    obj_matrix = np.array(
        [[t.values[idx] for idx in remaining_obj_indices] for t in candidates]
    )
    minimize_flags = [
        directions[idx] == optuna.study.StudyDirection.MINIMIZE
        for idx in remaining_obj_indices
    ]

    # Compute Pareto front.
    front_indices = _pareto_front_indices(obj_matrix, minimize_flags)
    pareto_trials = [candidates[i] for i in front_indices]

    # Tiebreak: lowest mean rank across remaining objectives
    # (rank computed among ALL Phase 1 survivors).
    n_candidates = len(candidates)
    trial_numbers = np.array([t.number for t in candidates])
    ranks = np.zeros((n_candidates, len(remaining_obj_indices)))
    for col_j in range(len(remaining_obj_indices)):
        col_vals = obj_matrix[:, col_j]
        # Use lexsort for stable ranking: secondary key = trial number,
        # primary key = objective value.  This ensures tied objective
        # values receive consistent rank ordering across platforms.
        primary = col_vals if minimize_flags[col_j] else -col_vals
        order = np.lexsort((trial_numbers, primary))
        rank_arr = np.empty(n_candidates, dtype=float)
        rank_arr[order] = np.arange(1, n_candidates + 1, dtype=float)
        ranks[:, col_j] = rank_arr

    mean_ranks = ranks.mean(axis=1)

    # Pick the candidate with lowest mean rank, breaking ties by
    # trial number for determinism.
    min_rank = mean_ranks.min()
    tied_indices = np.where(mean_ranks == min_rank)[0]
    best_idx = int(min(tied_indices, key=lambda i: candidates[i].number))
    best_trial = candidates[best_idx]

    return best_trial, SelectionResult(
        thresholds_met=thresholds_met,
        pareto_front=pareto_trials,
        n_candidates_per_step=n_candidates_per_step,
    )
