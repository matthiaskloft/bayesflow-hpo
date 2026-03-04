"""Optuna study creation/resume helpers."""

from __future__ import annotations

from typing import Any

import optuna
from optuna.trial import TrialState


def create_study(
    study_name: str = "bayesflow_hpo",
    directions: list[str] | None = None,
    storage: str | None = None,
    load_if_exists: bool = True,
    sampler: Any | None = None,
    pruner: Any | None = None,
    warm_start_from: optuna.Study | None = None,
    warm_start_top_k: int = 20,
    warm_start_metric_index: int = 0,
) -> optuna.Study:
    """Create or resume an Optuna study."""
    if directions is None:
        directions = ["minimize", "minimize"]

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=20)

    if pruner is None:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5,
        )

    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )
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
