"""Tests for warm-starting studies."""

import optuna

from bayesflow_hpo.optimization.study import warm_start_study


def _build_trial(value_a: float, value_b: float):
    return optuna.trial.create_trial(
        params={"x": float(value_a)},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 1.0)},
        values=(value_a, value_b),
    )


def test_warm_start_study_adds_top_k_trials():
    source = optuna.create_study(directions=["minimize", "minimize"])
    source.add_trial(_build_trial(0.30, 0.20))
    source.add_trial(_build_trial(0.10, 0.50))
    source.add_trial(_build_trial(0.20, 0.10))

    target = optuna.create_study(directions=["minimize", "minimize"])
    added = warm_start_study(target_study=target, source_study=source, top_k=2)

    assert added == 2
    assert len(target.trials) == 2
    best_first_objective = sorted(trial.values[0] for trial in target.trials)
    assert best_first_objective == [0.10, 0.20]
