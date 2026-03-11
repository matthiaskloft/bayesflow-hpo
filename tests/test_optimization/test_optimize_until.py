"""Tests for optimize_until and budget-aware sampling."""

import optuna

from bayesflow_hpo.optimization.study import (
    _budget_constraints_func,
    count_trained_trials,
    optimize_until,
)


def _make_trial(rejected: bool = False, **kwargs):
    user_attrs = {}
    if rejected:
        user_attrs["rejected_reason"] = "param_budget"
    return optuna.trial.create_trial(
        params={"x": 0.5},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 1.0)},
        values=(0.5, 0.5),
        user_attrs=user_attrs,
        **kwargs,
    )


def test_count_trained_trials_excludes_rejected():
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.add_trial(_make_trial(rejected=False))
    study.add_trial(_make_trial(rejected=True))
    study.add_trial(_make_trial(rejected=False))

    assert count_trained_trials(study) == 2


def test_count_trained_trials_empty_study():
    study = optuna.create_study(directions=["minimize", "minimize"])
    assert count_trained_trials(study) == 0


def test_budget_constraints_func_feasible():
    trial = _make_trial(rejected=False)
    assert _budget_constraints_func(trial) == [0.0]


def test_budget_constraints_func_infeasible():
    trial = _make_trial(rejected=True)
    assert _budget_constraints_func(trial) == [1.0]


def test_optimize_until_counts_only_trained():
    """optimize_until should keep going past rejected trials."""
    call_count = 0

    def objective(trial: optuna.Trial):
        nonlocal call_count
        call_count += 1
        x = trial.suggest_float("x", 0.0, 1.0)
        # Reject every other trial.
        if call_count % 2 == 0:
            trial.set_user_attr("rejected_reason", "param_budget")
        return (x, x)

    study = optuna.create_study(directions=["minimize", "minimize"])
    optimize_until(
        study,
        objective,
        n_trained=4,
        max_total_trials=20,
        show_progress_bar=False,
    )

    trained = count_trained_trials(study)
    assert trained >= 4


def test_optimize_until_respects_max_total_trials():
    """optimize_until should stop at the hard cap when all trials are rejected."""

    def always_reject(trial: optuna.Trial):
        trial.suggest_float("x", 0.0, 1.0)
        trial.set_user_attr("rejected_reason", "param_budget")
        return (1.0, 1.5)

    study = optuna.create_study(directions=["minimize", "minimize"])
    optimize_until(
        study,
        always_reject,
        n_trained=10,
        max_total_trials=5,
        show_progress_bar=False,
    )

    # All rejected -> non-rejected count never increases -> hits the hard
    # safety cap of 5 * max_total_trials = 25 (+ batch overshoot).
    assert len(study.trials) <= 30
    assert count_trained_trials(study) == 0


def test_budget_aware_sampler_uses_constraints():
    """Verify the TPE sampler is created with constraints_func."""
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(
            seed=0,
            constraints_func=_budget_constraints_func,
        ),
    )
    assert study.sampler._constraints_func is not None
