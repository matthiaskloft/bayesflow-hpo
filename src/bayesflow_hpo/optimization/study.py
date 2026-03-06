"""Optuna study creation/resume helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import optuna
from optuna.trial import TrialState

logger = logging.getLogger(__name__)


def _budget_constraints_func(trial: optuna.trial.FrozenTrial) -> list[float]:
    """Optuna constraints_func: returns >0 when a trial violated a budget.

    This teaches the TPE sampler to avoid infeasible regions of the search
    space, even during startup trials.
    """
    if "rejected_reason" in trial.user_attrs:
        return [1.0]
    return [0.0]


def create_study(
    study_name: str = "bayesflow_hpo",
    directions: list[str] | None = None,
    metric_names: list[str] | None = None,
    storage: str | None = None,
    load_if_exists: bool = True,
    sampler: Any | None = None,
    pruner: Any | None = None,
    warm_start_from: optuna.Study | None = None,
    warm_start_top_k: int = 20,
    warm_start_metric_index: int = 0,
    budget_aware: bool = True,
) -> optuna.Study:
    """Create or resume an Optuna study."""
    if directions is None:
        directions = ["minimize", "minimize"]

    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
            n_startup_trials=20,
            constraints_func=_budget_constraints_func if budget_aware else None,
        )

    if pruner is None:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5,
        )

    create_kwargs: dict[str, Any] = dict(
        study_name=study_name,
        directions=directions,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )
    # metric_names was added in Optuna ≥4.x — pass it only when supported.
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
            metric_index=warm_start_metric_index,
        )
    return study


def resume_study(study_name: str, storage: str) -> optuna.Study:
    """Resume a persisted study."""
    return create_study(study_name=study_name, storage=storage, load_if_exists=True)


def warm_start_study(
    target_study: optuna.Study,
    source_study: optuna.Study,
    top_k: int = 20,
    metric_index: int = 0,
) -> int:
    """Seed `target_study` with best completed trials from `source_study`."""
    complete_trials = [
        trial
        for trial in source_study.trials
        if trial.state == TrialState.COMPLETE and trial.values is not None
    ]
    if not complete_trials:
        return 0

    ranked = sorted(
        complete_trials,
        key=lambda trial: trial.values[metric_index],
    )

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


def optimize_until(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], tuple[float, float]],
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
        Hard cap on total trials (including rejected). Defaults to ``5 * n_trained``.
    show_progress_bar
        Whether to show Optuna's progress bar.
    """
    if max_total_trials is None:
        max_total_trials = 5 * n_trained

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

    trained_now = count_trained_trials(study) - trained_before
    total_now = len(study.trials) - total_before
    rejected = total_now - trained_now
    if rejected > 0:
        logger.info(
            "Completed %d trained trials (%d total, %d budget-rejected).",
            trained_now, total_now, rejected,
        )
    if trained_now < n_trained:
        logger.warning(
            "Reached max_total_trials=%d before achieving %d trained trials "
            "(got %d). Consider raising max_param_count or tightening the search space.",
            max_total_trials, n_trained, trained_now,
        )
