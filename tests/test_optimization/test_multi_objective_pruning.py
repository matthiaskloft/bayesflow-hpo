"""Tests for multi-objective pruning in PeriodicValidationCallback."""

from unittest.mock import patch

import optuna
import pytest
from optuna.trial import TrialState

from bayesflow_hpo.optimization.validation_callback import (
    _should_prune_multi_objective,
)

_PRUNE = _should_prune_multi_objective


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_study(n_objectives: int = 2) -> optuna.Study:
    """Create an in-memory multi- or single-objective study."""
    directions = ["minimize"] * n_objectives
    return optuna.create_study(directions=directions)


def _add_completed_trial(
    study: optuna.Study,
    values: list[float],
    user_attrs: dict | None = None,
) -> None:
    """Add a synthetic completed trial to *study*."""
    trial = optuna.trial.create_trial(
        params={},
        distributions={},
        values=values,
        user_attrs=user_attrs or {},
        state=TrialState.COMPLETE,
    )
    study.add_trial(trial)


# -------------------------------------------------------------------
# _should_prune_multi_objective
# -------------------------------------------------------------------


def test_no_pruning_below_startup_threshold():
    """Should not prune when fewer than n_startup_trials references exist."""
    study = _make_study()
    for i in range(3):
        _add_completed_trial(study, [0.1, 0.5], {"val_score_step_1": 0.1 + i * 0.1})

    # Start a real trial so we get a live Trial object.
    trial = study.ask()
    assert not _PRUNE(trial, score=0.9, step=1, n_startup_trials=5)


def test_no_pruning_at_median():
    """Score equal to the median should NOT trigger pruning."""
    study = _make_study()
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    for s in scores:
        _add_completed_trial(study, [s, 0.5], {"val_score_step_1": s})

    trial = study.ask()
    # Median is 0.3 — equal should not prune.
    assert not _PRUNE(trial, score=0.3, step=1, n_startup_trials=5)


def test_prune_above_median():
    """Score above the median should trigger pruning."""
    study = _make_study()
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    for s in scores:
        _add_completed_trial(study, [s, 0.5], {"val_score_step_1": s})

    trial = study.ask()
    assert _PRUNE(trial, score=0.4, step=1, n_startup_trials=5)


def test_budget_rejected_trials_excluded():
    """Trials with rejected_reason should not count as references."""
    study = _make_study()
    # 5 completed trials, but 3 are budget-rejected.
    for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
        attrs = {"val_score_step_1": s}
        if s >= 0.3:
            attrs["rejected_reason"] = "param_budget"
        _add_completed_trial(study, [s, 0.5], attrs)

    trial = study.ask()
    # Only 2 non-rejected trials → below startup threshold of 5.
    assert not _PRUNE(trial, score=0.9, step=1, n_startup_trials=5)


def test_different_steps_independent():
    """Scores at step 1 should not affect pruning decision at step 2."""
    study = _make_study()
    for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _add_completed_trial(study, [s, 0.5], {"val_score_step_1": s})

    trial = study.ask()
    # No completed trials have val_score_step_2 → below startup threshold.
    assert not _PRUNE(trial, score=0.9, step=2, n_startup_trials=5)


def test_nan_score_triggers_pruning():
    """A NaN intermediate score should trigger immediate pruning."""
    study = _make_study()
    for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _add_completed_trial(study, [s, 0.5], {"val_score_step_1": s})

    trial = study.ask()
    assert _PRUNE(trial, score=float("nan"), step=1, n_startup_trials=5)


def test_inf_score_triggers_pruning():
    """An Inf intermediate score should trigger immediate pruning."""
    study = _make_study()
    trial = study.ask()
    assert _PRUNE(trial, score=float("inf"), step=1, n_startup_trials=5)


def test_nan_in_completed_scores_ignored():
    """NaN scores from completed trials should be filtered out."""
    study = _make_study()
    # 5 trials with valid scores + 2 with NaN.
    for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _add_completed_trial(study, [s, 0.5], {"val_score_step_1": s})
    _add_completed_trial(
        study, [0.9, 0.5], {"val_score_step_1": float("nan")}
    )
    _add_completed_trial(
        study, [0.9, 0.5], {"val_score_step_1": float("nan")}
    )

    trial = study.ask()
    # Median of [0.1, 0.2, 0.3, 0.4, 0.5] is 0.3.
    # NaN entries should be excluded, not corrupt the median.
    assert not _PRUNE(trial, score=0.2, step=1, n_startup_trials=5)
    assert _PRUNE(trial, score=0.4, step=1, n_startup_trials=5)


def test_n_startup_trials_zero_never_prunes():
    """n_startup_trials=0 should be treated as disabled (no pruning)."""
    study = _make_study()
    for s in [0.1, 0.2]:
        _add_completed_trial(study, [s, 0.5], {"val_score_step_1": s})

    trial = study.ask()
    assert not _PRUNE(trial, score=0.9, step=1, n_startup_trials=0)


# -------------------------------------------------------------------
# PeriodicValidationCallback integration
# -------------------------------------------------------------------


def test_callback_stores_step_keyed_attr():
    """Multi-objective callback should store val_score_step_{step}."""
    from bayesflow_hpo.optimization.validation_callback import (
        PeriodicValidationCallback,
    )

    study = _make_study()
    # Add enough completed trials so pruning could activate.
    for s in [0.01, 0.02, 0.03, 0.04, 0.05]:
        _add_completed_trial(study, [s, 0.5], {"val_score_step_1": s})

    trial = study.ask()
    cb = PeriodicValidationCallback(
        trial=trial,
        approximator=None,
        validation_data=None,
        param_keys=[],
        data_keys=[],
        interval=1,
        warmup=0,
        n_startup_trials=5,
    )

    # Mock _run_lightweight_validation to return a low score (no pruning).
    with patch.object(cb, "_run_lightweight_validation", return_value=0.01):
        cb.on_epoch_end(epoch=0)

    assert "val_score_step_1" in trial.user_attrs
    assert trial.user_attrs["val_score_step_1"] == 0.01


def test_callback_raises_trial_pruned():
    """Callback should raise TrialPruned when score exceeds median."""
    from bayesflow_hpo.optimization.validation_callback import (
        PeriodicValidationCallback,
    )

    study = _make_study()
    for s in [0.01, 0.02, 0.03, 0.04, 0.05]:
        _add_completed_trial(study, [s, 0.5], {"val_score_step_1": s})

    trial = study.ask()
    cb = PeriodicValidationCallback(
        trial=trial,
        approximator=None,
        validation_data=None,
        param_keys=[],
        data_keys=[],
        interval=1,
        warmup=0,
        n_startup_trials=5,
    )

    # Mock _run_lightweight_validation to return a bad score.
    with patch.object(cb, "_run_lightweight_validation", return_value=0.99):
        with pytest.raises(optuna.TrialPruned):
            cb.on_epoch_end(epoch=0)


def test_single_objective_uses_trial_report():
    """Single-objective path should still call trial.report + should_prune."""
    from bayesflow_hpo.optimization.validation_callback import (
        PeriodicValidationCallback,
    )

    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    cb = PeriodicValidationCallback(
        trial=trial,
        approximator=None,
        validation_data=None,
        param_keys=[],
        data_keys=[],
        interval=1,
        warmup=0,
    )

    with (
        patch.object(cb, "_run_lightweight_validation", return_value=0.5),
        patch.object(trial, "report") as mock_report,
    ):
        cb.on_epoch_end(epoch=0)

    mock_report.assert_called_once_with(0.5, step=1)
