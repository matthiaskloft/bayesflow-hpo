"""Extract tabular/pareto results from Optuna studies."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd


def get_pareto_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Return Pareto-optimal trials from a multi-objective study."""
    return study.best_trials


def trials_to_dataframe(study: optuna.Study, include_pruned: bool = False) -> pd.DataFrame:
    """Convert study trials to a DataFrame."""
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            rec: dict[str, Any] = {"trial_number": trial.number, **trial.params}
            if trial.values is None:
                continue
            if len(trial.values) == 1:
                rec["objective"] = trial.values[0]
            else:
                rec["objective_0"] = trial.values[0]
                rec["objective_1"] = trial.values[1]
            records.append(rec)
        elif include_pruned and trial.state == optuna.trial.TrialState.PRUNED:
            records.append({"trial_number": trial.number, "pruned": True, **trial.params})
    return pd.DataFrame(records)
