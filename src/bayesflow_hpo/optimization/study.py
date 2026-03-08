"""Optuna study creation/resume helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState

logger = logging.getLogger(__name__)

# Default SQLite storage file used by :func:`create_study`.
DEFAULT_STORAGE = "sqlite:///bayesflow_hpo.db"


def _budget_constraints_func(trial: optuna.trial.FrozenTrial) -> list[float]:
    """Optuna constraints_func: returns >0 when a trial violated a budget.

    This teaches the TPE sampler to avoid infeasible regions of the search
    space, even during startup trials.
    """
    if "rejected_reason" in trial.user_attrs:
        return [1.0]
    return [0.0]


def _mean_ranking_key(trial: optuna.trial.FrozenTrial) -> float:
    """Rank by the mean of objective values (excluding param_score).

    Falls back to the first objective value when multi-objective values
    are not available.
    """
    if trial.values and len(trial.values) > 1:
        # All values except the last (param_score) — lower is better.
        metric_values = trial.values[:-1]
        return float(np.mean(metric_values))
    if trial.values:
        return float(trial.values[0])
    return float("inf")


def create_study(
    study_name: str = "bayesflow_hpo",
    directions: list[str] | None = None,
    metric_names: list[str] | None = None,
    storage: str | None = DEFAULT_STORAGE,
    load_if_exists: bool = True,
    sampler: Any | None = None,
    pruner: Any | None = None,
    warm_start_from: optuna.Study | None = None,
    warm_start_top_k: int = 25,
    budget_aware: bool = True,
) -> optuna.Study:
    """Create or resume an Optuna study.

    Parameters
    ----------
    study_name
        Optuna study name (default ``"bayesflow_hpo"``).
    directions
        Optimization directions.  Default ``["minimize", "minimize"]``.
        The caller is responsible for passing the correct number of
        directions matching the objective shape.
    metric_names
        Human-readable names for each objective.
    storage
        Optuna storage URL.  Default ``"sqlite:///bayesflow_hpo.db"``
        for automatic persistence and crash recovery.  Pass ``None``
        for in-memory only.
    load_if_exists
        Resume a study with the same name if it already exists.
    sampler
        Optuna sampler.  Default ``TPESampler(seed=42,
        multivariate=True, n_startup_trials=25)``.
    pruner
        Optuna pruner.  Default ``MedianPruner(n_startup_trials=5,
        n_warmup_steps=1, interval_steps=1)``.  For single-objective
        studies, pruning uses this pruner via ``trial.should_prune()``.
        For multi-objective studies (the default), the Optuna pruner is
        **not** used; instead,
        :class:`~bayesflow_hpo.optimization.validation_callback.PeriodicValidationCallback`
        applies a custom median-based strategy.
    warm_start_from
        Optional source study to seed initial trials from.
    warm_start_top_k
        Number of best trials to copy from the source study.
    budget_aware
        Whether to attach a constraints function that marks
        budget-rejected trials as infeasible for the sampler.
    """
    if directions is None:
        directions = ["minimize", "minimize"]

    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
            n_startup_trials=25,
            warn_independent_sampling=False,
            constraints_func=_budget_constraints_func if budget_aware else None,
        )

    if pruner is None:
        # The step counter is managed by PeriodicValidationCallback, so
        # we set warmup/interval to 1 and let the callback decide timing.
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1,
        )

    create_kwargs: dict[str, Any] = dict(
        study_name=study_name,
        directions=directions,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )
    # metric_names was added in Optuna >=4.x — pass it only when supported.
    import inspect

    if "metric_names" in inspect.signature(optuna.create_study).parameters:
        create_kwargs["metric_names"] = metric_names
    study = optuna.create_study(**create_kwargs)
    # Stash metric names as a fallback for result extraction.
    if metric_names and not getattr(study, "metric_names", None):
        study._metric_names = metric_names  # type: ignore[attr-defined]
    if warm_start_from is not None and len(study.trials) == 0:
        warm_start_study(
            target_study=study,
            source_study=warm_start_from,
            top_k=warm_start_top_k,
        )
    return study


def resume_study(study_name: str, storage: str) -> optuna.Study:
    """Resume a persisted study."""
    return create_study(study_name=study_name, storage=storage, load_if_exists=True)


def warm_start_study(
    target_study: optuna.Study,
    source_study: optuna.Study,
    top_k: int = 25,
) -> int:
    """Seed *target_study* with best completed trials from *source_study*.

    Trials are ranked by the arithmetic mean of their objective values
    (excluding param_score), falling back to the first objective when
    only a single value is available.

    Parameters
    ----------
    target_study
        Study to seed.
    source_study
        Study to copy trials from.
    top_k
        Maximum number of trials to copy.

    Returns
    -------
    int
        Number of trials actually added.
    """
    complete_trials = [
        trial
        for trial in source_study.trials
        if trial.state == TrialState.COMPLETE and trial.values is not None
    ]
    if not complete_trials:
        return 0

    ranked = sorted(complete_trials, key=_mean_ranking_key)

    added = 0
    for trial in ranked[: max(0, int(top_k))]:
        seeded_trial = optuna.trial.create_trial(
            params=trial.params,
            distributions=trial.distributions,
            values=trial.values,
            intermediate_values=trial.intermediate_values,
            user_attrs=trial.user_attrs,
            system_attrs=trial.system_attrs,
            state=TrialState.COMPLETE,
        )
        target_study.add_trial(seeded_trial)
        added += 1

    return added


def count_trained_trials(study: optuna.Study) -> int:
    """Count completed trials that were not rejected by budget checks."""
    return sum(
        1
        for t in study.trials
        if t.state == TrialState.COMPLETE
        and "rejected_reason" not in t.user_attrs
    )


def _count_pruned(study: optuna.Study, since_trial: int = 0) -> int:
    """Count trials pruned by intermediate validation."""
    return sum(
        1
        for t in study.trials
        if t.number >= since_trial and t.state == TrialState.PRUNED
    )


def _count_failure_reasons(
    study: optuna.Study,
    since_trial: int = 0,
) -> dict[str, int]:
    """Count training errors and rejection reasons for recent trials."""
    counts: dict[str, int] = {}
    for t in study.trials:
        if t.number < since_trial:
            continue
        reason = t.user_attrs.get("rejected_reason")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
        error = t.user_attrs.get("training_error")
        if error:
            # Group identical error messages.
            key = f"error: {error[:80]}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _best_objective_so_far(
    study: optuna.Study,
    select_by: int = 0,
) -> float | None:
    """Return the best value for the selected objective across trained trials."""
    best = None
    for t in study.trials:
        if (
            t.state == TrialState.COMPLETE
            and t.values is not None
            and "rejected_reason" not in t.user_attrs
        ):
            val = t.values[select_by]
            if best is None or val < best:
                best = val
    return best


def optimize_until(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], tuple[float, ...]],
    n_trained: int,
    *,
    max_total_trials: int | None = None,
    show_progress_bar: bool = True,
) -> None:
    """Run trials until *n_trained* have actually trained (not budget-rejected).

    Parameters
    ----------
    study
        The Optuna study to optimize.
    objective
        The objective callable.
    n_trained
        Target number of trials that pass budget checks and complete training.
    max_total_trials
        Hard cap on total trials (including rejected). Defaults to
        ``3 * n_trained``.
    show_progress_bar
        Whether to show Optuna's progress bar.
    """
    if max_total_trials is None:
        max_total_trials = 3 * n_trained

    trained_before = count_trained_trials(study)
    target = trained_before + n_trained
    total_before = len(study.trials)

    while (
        count_trained_trials(study) < target
        and len(study.trials) - total_before < max_total_trials
    ):
        remaining_trained = target - count_trained_trials(study)
        remaining_total = max_total_trials - (len(study.trials) - total_before)
        # Run in small batches to re-check the trained count regularly.
        batch = min(remaining_trained, remaining_total, max(1, n_trained // 4))
        study.optimize(
            objective,
            n_trials=batch,
            show_progress_bar=show_progress_bar,
            gc_after_trial=True,
        )

        # --- Live progress summary after each batch ---
        trained_now = count_trained_trials(study) - trained_before
        total_now = len(study.trials) - total_before
        pruned = _count_pruned(study, since_trial=total_before)
        rejected = total_now - trained_now - pruned
        best = _best_objective_so_far(study)
        best_str = f"{best:.4f}" if best is not None else "n/a"
        parts = [f"{trained_now}/{n_trained} trained"]
        if pruned:
            parts.append(f"{pruned} pruned")
        if rejected:
            parts.append(f"{rejected} rejected")
        parts += [f"{total_now} total", f"best: {best_str}"]
        logger.info("Progress: %s", " | ".join(parts))

    # --- Final summary ---
    trained_now = count_trained_trials(study) - trained_before
    total_now = len(study.trials) - total_before
    pruned = _count_pruned(study, since_trial=total_before)
    rejected = total_now - trained_now - pruned
    if rejected > 0 or pruned > 0:
        parts = [f"{trained_now} trained", f"{total_now} total"]
        if pruned:
            parts.append(f"{pruned} pruned")
        if rejected:
            parts.append(f"{rejected} rejected")
        logger.info("Completed %s.", ", ".join(parts))

    # --- Failure reason breakdown ---
    reasons = _count_failure_reasons(study, since_trial=total_before)
    if reasons:
        reason_parts = [f"{reason}: {count}" for reason, count in reasons.items()]
        logger.info("Failure breakdown: %s", " | ".join(reason_parts))
        # Warn if a single reason dominates (signals a systemic issue).
        dominant = max(reasons.values())
        if total_now > 2 and dominant / total_now > 0.5:
            logger.warning(
                "Over half of trials failed for the same reason. "
                "Check the most common failure above — it may indicate "
                "a configuration issue rather than bad hyperparameters.",
            )

    if trained_now < n_trained:
        hint_parts = []
        if reasons:
            hint_parts.append(
                "failure breakdown: "
                + ", ".join(f"{r}: {c}" for r, c in reasons.items())
            )
        if pruned:
            hint_parts.append(f"{pruned} trials were pruned")
        hint = (
            f" ({'; '.join(hint_parts)})"
            if hint_parts
            else ""
        )
        logger.warning(
            "Reached max_total_trials=%d before achieving %d trained "
            "trials (got %d)%s. Consider raising max_total_trials, "
            "max_param_count, or adjusting the search space.",
            max_total_trials, n_trained, trained_now, hint,
        )
