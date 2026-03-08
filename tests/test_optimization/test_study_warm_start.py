"""Tests for warm-starting studies and ranking helpers."""

import optuna

from bayesflow_hpo.optimization.study import _mean_ranking_key, warm_start_study


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


def test_mean_ranking_key_two_objectives():
    """With 2 values, ranks by the first (only metric, excl param_score)."""
    trial = _build_trial(0.30, 0.80)
    assert _mean_ranking_key(trial) == 0.30


def test_mean_ranking_key_three_objectives():
    """With 3 values, ranks by mean of the first two (excl param_score)."""
    trial = optuna.trial.create_trial(
        params={"x": 0.1},
        distributions={
            "x": optuna.distributions.FloatDistribution(0.0, 1.0),
        },
        values=(0.10, 0.30, 0.99),  # cal_err, nrmse, param_score
    )
    # mean of [0.10, 0.30] = 0.20
    assert abs(_mean_ranking_key(trial) - 0.20) < 1e-9


def test_mean_ranking_key_single_value():
    """Single-objective trial returns that value directly."""
    trial = optuna.trial.create_trial(
        params={"x": 0.1},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 1.0)},
        values=(0.42,),
    )
    assert _mean_ranking_key(trial) == 0.42


def test_mean_ranking_key_no_values():
    trial = optuna.trial.create_trial(
        params={},
        distributions={},
        values=None,
        state=optuna.trial.TrialState.FAIL,
    )
    assert _mean_ranking_key(trial) == float("inf")
