"""Tests for warm-starting studies and ranking helpers."""

import optuna
import pytest

from bayesflow_hpo.optimization.study import _mean_ranking_key, warm_start_study

_FLOAT_DIST = {"x": optuna.distributions.FloatDistribution(0.0, 1.0)}


def _build_trial(value_a: float, value_b: float):
    return optuna.trial.create_trial(
        params={"x": float(value_a)},
        distributions=_FLOAT_DIST,
        values=(value_a, value_b),
    )


def _build_failed_trial(x: float = 0.5):
    return optuna.trial.create_trial(
        params={"x": x},
        distributions=_FLOAT_DIST,
        values=None,
        state=optuna.trial.TrialState.FAIL,
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
        distributions=_FLOAT_DIST,
        values=(0.10, 0.30, 0.99),  # cal_err, nrmse, param_score
    )
    # mean of [0.10, 0.30] = 0.20
    assert abs(_mean_ranking_key(trial) - 0.20) < 1e-9


def test_mean_ranking_key_single_value():
    """Single-objective trial returns that value directly."""
    trial = optuna.trial.create_trial(
        params={"x": 0.1},
        distributions=_FLOAT_DIST,
        values=(0.42,),
    )
    assert _mean_ranking_key(trial) == 0.42


def test_mean_ranking_key_no_values():
    trial = _build_failed_trial()
    assert _mean_ranking_key(trial) == float("inf")


# --- Edge-case tests for warm_start_study ---


def test_warm_start_empty_source():
    """Source study with no COMPLETE trials returns 0, target stays empty."""
    source = optuna.create_study(directions=["minimize", "minimize"])
    source.add_trial(_build_failed_trial())
    target = optuna.create_study(directions=["minimize", "minimize"])
    added = warm_start_study(target_study=target, source_study=source, top_k=5)

    assert added == 0
    assert len(target.trials) == 0


@pytest.mark.parametrize("top_k", [0, -3])
def test_warm_start_non_positive_top_k_copies_nothing(top_k: int):
    """top_k <= 0 copies nothing."""
    source = optuna.create_study(directions=["minimize", "minimize"])
    source.add_trial(_build_trial(0.10, 0.20))

    target = optuna.create_study(directions=["minimize", "minimize"])
    added = warm_start_study(target_study=target, source_study=source, top_k=top_k)

    assert added == 0
    assert len(target.trials) == 0


def test_warm_start_top_k_exceeds_available():
    """top_k larger than available trials copies all of them."""
    source = optuna.create_study(directions=["minimize", "minimize"])
    source.add_trial(_build_trial(0.10, 0.20))
    source.add_trial(_build_trial(0.30, 0.40))
    source.add_trial(_build_trial(0.50, 0.60))

    target = optuna.create_study(directions=["minimize", "minimize"])
    added = warm_start_study(target_study=target, source_study=source, top_k=10)

    assert added == 3
    assert len(target.trials) == 3


def test_warm_start_skips_non_complete_trials():
    """Only COMPLETE trials are copied; FAIL, PRUNED, and RUNNING are skipped."""
    source = optuna.create_study(directions=["minimize", "minimize"])
    source.add_trial(_build_trial(0.10, 0.20))
    source.add_trial(_build_trial(0.30, 0.40))
    source.add_trial(_build_failed_trial(x=0.5))
    # PRUNED trials require values for Optuna to accept them
    source.add_trial(
        optuna.trial.create_trial(
            params={"x": 0.6},
            distributions=_FLOAT_DIST,
            values=(0.60, 0.70),
            state=optuna.trial.TrialState.PRUNED,
        )
    )
    source.add_trial(
        optuna.trial.create_trial(
            params={"x": 0.7},
            distributions=_FLOAT_DIST,
            values=None,
            state=optuna.trial.TrialState.RUNNING,
        )
    )

    target = optuna.create_study(directions=["minimize", "minimize"])
    added = warm_start_study(target_study=target, source_study=source, top_k=10)

    assert added == 2
    assert len(target.trials) == 2


def test_warm_start_preserves_user_attrs():
    """User attributes from source trials are preserved on target trials."""
    source = optuna.create_study(directions=["minimize", "minimize"])
    source.add_trial(
        optuna.trial.create_trial(
            params={"x": 0.1},
            distributions=_FLOAT_DIST,
            values=(0.10, 0.20),
            user_attrs={"param_count": 50_000, "network_type": "coupling_flow"},
        )
    )

    target = optuna.create_study(directions=["minimize", "minimize"])
    warm_start_study(target_study=target, source_study=source, top_k=5)

    assert len(target.trials) == 1
    assert target.trials[0].user_attrs["param_count"] == 50_000
    assert target.trials[0].user_attrs["network_type"] == "coupling_flow"
