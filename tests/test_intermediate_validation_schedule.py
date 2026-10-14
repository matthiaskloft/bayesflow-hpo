"""Intermediate-validation schedule: epoch placement, guards, warnings (#131)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import optuna
import pytest

from bayesflow_hpo.objectives import (
    INTERMEDIATE_VALIDATION_SCHEDULE_ATTR,
    check_intermediate_validation_schedule,
    stamp_trial_intermediate_validation_schedule,
)
from bayesflow_hpo.optimization.validation_callback import (
    PeriodicValidationCallback,
)
from bayesflow_hpo.validation.data import ValidationDataset

_DUMMY_VALIDATION_DATA = ValidationDataset(
    simulations=[],
    condition_labels=[],
    param_keys=["p"],
    data_keys=["x"],
    seed=0,
)


def _trial_with_step_scores(**kwargs) -> optuna.trial.FrozenTrial:
    return optuna.trial.create_trial(
        values=[0.2, 0.3],
        user_attrs={"val_nrmse_step_1": 0.3},
        **kwargs,
    )


def _two_objective_study() -> optuna.Study:
    return optuna.create_study(directions=["minimize", "minimize"])


# --- Where the checks land ---------------------------------------------------


@pytest.mark.parametrize(
    ("epochs", "warmup", "interval", "expected"),
    [
        (64, 48, 16, [48]),
        (64, 10, 10, [10, 20, 30, 40, 50, 60]),
        (40, 48, 16, []),
        (49, 48, 16, [48]),
        (48, 48, 16, []),
    ],
)
def test_validation_epochs(epochs, warmup, interval, expected):
    """The callback validates exactly at the epoch indices the docs state."""
    trial = _two_objective_study().ask()
    callback = PeriodicValidationCallback(
        trial=trial,
        approximator=None,
        validation_data=_DUMMY_VALIDATION_DATA,
        interval=interval,
        warmup=warmup,
        pruning_strategy="none",
        objective_metrics=["nrmse"],
    )
    validated: list[int] = []

    for epoch in range(epochs):
        with patch.object(
            callback,
            "_run_lightweight_validation",
            side_effect=lambda e=epoch: validated.append(e) or {"nrmse": 0.1},
        ):
            callback.on_epoch_end(epoch=epoch)

    assert validated == expected


# --- The study-level guard ---------------------------------------------------


def test_fresh_study_is_stamped():
    study = _two_objective_study()
    check_intermediate_validation_schedule(study, 48, 16)
    assert study.user_attrs[INTERMEDIATE_VALIDATION_SCHEDULE_ATTR] == {
        "warmup": 48,
        "interval": 16,
    }


def test_changed_schedule_with_step_scores_is_refused():
    study = _two_objective_study()
    check_intermediate_validation_schedule(study, 48, 16)
    study.add_trial(_trial_with_step_scores())
    check_intermediate_validation_schedule(study, 48, 16)
    with pytest.raises(ValueError, match="warmup=48, interval=16"):
        check_intermediate_validation_schedule(study, 10, 10)


def test_unstamped_study_reads_as_default_schedule():
    study = _two_objective_study()
    study.add_trial(_trial_with_step_scores())
    check_intermediate_validation_schedule(study, 10, 10)
    with pytest.raises(ValueError, match="intermediate validation"):
        check_intermediate_validation_schedule(study, 48, 16)


def test_unrecognized_attribute_reads_as_default_schedule():
    study = _two_objective_study()
    study.set_user_attr(INTERMEDIATE_VALIDATION_SCHEDULE_ATTR, "garbage")
    study.add_trial(_trial_with_step_scores())
    with pytest.raises(ValueError, match="warmup=10, interval=10"):
        check_intermediate_validation_schedule(study, 48, 16)


def test_trials_without_step_scores_do_not_conflict():
    """A study that never ran intermediate validation has nothing to mismatch."""
    study = _two_objective_study()
    study.add_trial(optuna.trial.create_trial(values=[0.2, 0.3]))
    check_intermediate_validation_schedule(study, 48, 16)
    assert study.user_attrs[INTERMEDIATE_VALIDATION_SCHEDULE_ATTR] == {
        "warmup": 48,
        "interval": 16,
    }


def test_single_objective_intermediate_values_count_as_step_scores():
    study = optuna.create_study()
    study.add_trial(
        optuna.trial.create_trial(value=0.2, intermediate_values={1: 0.4})
    )
    with pytest.raises(ValueError, match="intermediate validation"):
        check_intermediate_validation_schedule(study, 48, 16)


def test_current_trial_is_ignored_but_other_running_trials_are_not():
    study = _two_objective_study()
    check_intermediate_validation_schedule(study, 10, 10)
    other = study.ask()
    other.set_user_attr("val_nrmse_step_1", 0.3)
    current = study.ask()
    current.set_user_attr("val_nrmse_step_1", 0.3)

    with pytest.raises(ValueError, match="intermediate validation"):
        check_intermediate_validation_schedule(
            study, 48, 16, current_trial_number=current.number
        )

    lone = _two_objective_study()
    check_intermediate_validation_schedule(lone, 10, 10)
    only = lone.ask()
    only.set_user_attr("val_nrmse_step_1", 0.3)
    check_intermediate_validation_schedule(
        lone, 48, 16, current_trial_number=only.number
    )


def _start_worker_trial(study, warmup, interval):
    """What `GenericObjective` does for a trial before it trains."""
    trial = study.ask()
    check_intermediate_validation_schedule(
        study, warmup, interval, current_trial_number=trial.number
    )
    stamp_trial_intermediate_validation_schedule(trial, warmup, interval)
    return trial


def test_overlapping_worker_in_warmup_blocks_a_schedule_change():
    """Worker A's trial has no scores yet, but it will produce them at 10/10.

    Regression for the interleaving in the review of #132: without the
    per-trial record, worker B passed the guard and overwrote the study's
    schedule, and both trials then stored `val_*_step_1` at different epochs.
    """
    study = _two_objective_study()
    worker_a = _start_worker_trial(study, 10, 10)

    with pytest.raises(ValueError, match=f"Trial {worker_a.number} .* is running"):
        _start_worker_trial(study, 48, 16)
    assert study.user_attrs[INTERMEDIATE_VALIDATION_SCHEDULE_ATTR] == {
        "warmup": 10,
        "interval": 10,
    }


def test_per_trial_record_outranks_an_overwritten_study_attribute():
    study = _two_objective_study()
    worker_a = _start_worker_trial(study, 10, 10)
    worker_a.set_user_attr("val_nrmse_step_1", 0.3)
    study.tell(worker_a, [0.2, 0.3])
    # The study attribute is caller-writable; a trial's own record is what
    # its scores were taken at.
    study.set_user_attr(
        INTERMEDIATE_VALIDATION_SCHEDULE_ATTR, {"warmup": 48, "interval": 16}
    )

    with pytest.raises(ValueError, match="warmup=10, interval=10"):
        check_intermediate_validation_schedule(study, 48, 16)


def test_finished_trial_without_scores_does_not_block():
    """A failed or rejected trial produced nothing to mismatch."""
    study = _two_objective_study()
    failed = _start_worker_trial(study, 10, 10)
    study.tell(failed, state=optuna.trial.TrialState.FAIL)

    _start_worker_trial(study, 48, 16)
    assert study.user_attrs[INTERMEDIATE_VALIDATION_SCHEDULE_ATTR] == {
        "warmup": 48,
        "interval": 16,
    }


def test_source_check_does_not_mutate_metadata():
    study = _two_objective_study()
    check_intermediate_validation_schedule(study, 48, 16, record=False)
    assert INTERMEDIATE_VALIDATION_SCHEDULE_ATTR not in study.user_attrs


# --- optimize()'s study creation, resume and warm start ----------------------


def _objective(warmup, interval, pruning_strategy="dominance"):
    return SimpleNamespace(
        config=SimpleNamespace(
            objective_metrics=["nrmse"],
            pruning_n_startup_trials=5,
            pruning_strategy=pruning_strategy,
            training_mode="fixed_budget",
            intermediate_validation_warmup=warmup,
            intermediate_validation_interval=interval,
        )
    )


def _create_and_run(
    objective, *, storage, resume, warm_start_from=None, warm_start_top_k=5
):
    from bayesflow_hpo.api import _create_and_run_study

    with patch("bayesflow_hpo.api.optimize_until"):
        return _create_and_run_study(
            objective=objective,
            study_name="schedule",
            directions=["minimize"],
            metric_names=["nrmse"],
            storage=storage,
            resume=resume,
            warm_start_from=warm_start_from,
            warm_start_top_k=warm_start_top_k,
            n_trials=0,
            max_total_trials=None,
            show_progress_bar=False,
            has_cost=False,
        )


def _add_scored_trial(study):
    study.add_trial(
        optuna.trial.create_trial(
            value=0.2, user_attrs={"val_nrmse_step_1": 0.3}
        )
    )


def test_resume_with_changed_schedule_is_refused(tmp_path):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = _create_and_run(_objective(48, 16), storage=storage, resume=False)
    assert study.user_attrs[INTERMEDIATE_VALIDATION_SCHEDULE_ATTR] == {
        "warmup": 48,
        "interval": 16,
    }
    _add_scored_trial(study)

    _create_and_run(_objective(48, 16), storage=storage, resume=True)
    with pytest.raises(ValueError, match="intermediate validation"):
        _create_and_run(_objective(10, 10), storage=storage, resume=True)


def test_run_without_intermediate_validation_neither_checks_nor_stamps(
    tmp_path,
):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = _create_and_run(_objective(48, 16), storage=storage, resume=False)
    _add_scored_trial(study)

    resumed = _create_and_run(
        _objective(10, 10, pruning_strategy="none"),
        storage=storage,
        resume=True,
    )
    assert resumed.user_attrs[INTERMEDIATE_VALIDATION_SCHEDULE_ATTR] == {
        "warmup": 48,
        "interval": 16,
    }


def test_warm_start_from_a_different_schedule_is_refused():
    source = _create_and_run(_objective(48, 16), storage=None, resume=False)
    _add_scored_trial(source)

    with pytest.raises(ValueError, match="intermediate validation"):
        _create_and_run(
            _objective(10, 10),
            storage=None,
            resume=False,
            warm_start_from=source,
        )


def _source_with_uncopied_10_10_trials():
    """A 10/10 source whose only scores sit on trials warm start won't copy.

    The single unscored COMPLETE trial ranks first (value 0.1) and is the
    one copied at ``warm_start_top_k=1``.
    """
    # Created the way `optimize()` creates one, so it carries the objective
    # schema warm start requires; "none" leaves its schedule unrecorded.
    source = _create_and_run(
        _objective(10, 10, pruning_strategy="none"), storage=None, resume=False
    )
    source.add_trial(optuna.trial.create_trial(value=0.1))
    # Scored at 10/10, but outside top_k=1.
    _add_scored_trial(source)
    # Scored and pruned: warm start copies only COMPLETE trials.
    source.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.PRUNED,
            user_attrs={"val_nrmse_step_1": 0.3},
        )
    )
    # Still running at 10/10: its worker writes to the source, not the target.
    running = source.ask()
    stamp_trial_intermediate_validation_schedule(running, 10, 10)
    return source


def test_warm_start_judges_only_the_trials_it_copies():
    """Regression for the second review round of #132."""
    source = _source_with_uncopied_10_10_trials()

    target = _create_and_run(
        _objective(48, 16),
        storage=None,
        resume=False,
        warm_start_from=source,
        warm_start_top_k=1,
    )

    assert len(target.trials) == 1
    assert target.user_attrs[INTERMEDIATE_VALIDATION_SCHEDULE_ATTR] == {
        "warmup": 48,
        "interval": 16,
    }


def test_warm_start_still_refuses_a_copied_scored_trial():
    source = _source_with_uncopied_10_10_trials()

    with pytest.raises(ValueError, match="holds intermediate validation"):
        _create_and_run(
            _objective(48, 16),
            storage=None,
            resume=False,
            warm_start_from=source,
            warm_start_top_k=2,
        )


def test_warm_start_carries_the_source_schedule():
    source = _create_and_run(_objective(48, 16), storage=None, resume=False)
    _add_scored_trial(source)

    target = _create_and_run(
        _objective(48, 16),
        storage=None,
        resume=False,
        warm_start_from=source,
    )

    assert len(target.trials) == 1
    assert target.user_attrs[INTERMEDIATE_VALIDATION_SCHEDULE_ATTR] == {
        "warmup": 48,
        "interval": 16,
    }


# --- Per-trial guard and unreachable-warmup warning --------------------------


class _SampleReachedError(Exception):
    pass


def _direct_objective(warmup, interval, *, training_mode="fixed_budget"):
    from bayesflow_hpo.optimization.objective import GenericObjective

    def sample(trial):
        raise _SampleReachedError

    objective = GenericObjective.__new__(GenericObjective)
    objective.config = SimpleNamespace(
        aggregate="mean",
        search_space=SimpleNamespace(sample=sample),
        pruning_strategy="dominance",
        training_mode=training_mode,
        intermediate_validation_warmup=warmup,
        intermediate_validation_interval=interval,
    )
    objective._warned_unreachable_warmup = set()
    return objective


def test_direct_objective_refuses_changed_schedule_before_sampling():
    study = _two_objective_study()
    study.set_user_attr(
        INTERMEDIATE_VALIDATION_SCHEDULE_ATTR, {"warmup": 48, "interval": 16}
    )
    study.add_trial(_trial_with_step_scores())

    with pytest.raises(ValueError, match="intermediate validation"):
        _direct_objective(10, 10)(study.ask())
    trial = study.ask()
    with pytest.raises(_SampleReachedError):
        _direct_objective(48, 16)(trial)
    assert trial.user_attrs["intermediate_validation_warmup"] == 48
    assert trial.user_attrs["intermediate_validation_interval"] == 16


@pytest.mark.parametrize(
    ("training_mode", "consequence"),
    [
        ("fixed_budget", "pruning cannot act"),
        ("open_ended", "early stopping cannot fire"),
    ],
)
def test_unreachable_warmup_warns_once_per_epoch_count(
    caplog, training_mode, consequence
):
    objective = _direct_objective(48, 16, training_mode=training_mode)
    with caplog.at_level(
        logging.WARNING, logger="bayesflow_hpo.optimization.objective"
    ):
        objective._warn_if_warmup_unreachable(40)
        objective._warn_if_warmup_unreachable(40)
        objective._warn_if_warmup_unreachable(48)
        objective._warn_if_warmup_unreachable(49)

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "epochs=40" in messages[0]
    assert "epochs=48" in messages[1]
    assert all(consequence in m for m in messages)


def test_no_warning_when_intermediate_validation_is_off(caplog):
    objective = _direct_objective(48, 16)
    objective.config.pruning_strategy = "none"
    with caplog.at_level(
        logging.WARNING, logger="bayesflow_hpo.optimization.objective"
    ):
        objective._warn_if_warmup_unreachable(10)
    assert not caplog.records


# --- Argument validation ------------------------------------------------------


@pytest.mark.parametrize(
    ("warmup", "interval", "error", "match"),
    [
        (10, 0, ValueError, "interval must be >= 1"),
        (-1, 10, ValueError, "warmup must be >= 0"),
        (10, 2.5, TypeError, "interval must be an int"),
        (10.0, 10, TypeError, "warmup must be an int"),
        (10, True, TypeError, "interval must be an int"),
    ],
)
def test_invalid_schedule_rejected(warmup, interval, error, match):
    from bayesflow_hpo.optimization.objective import (
        _check_intermediate_validation_schedule,
    )

    with pytest.raises(error, match=match):
        _check_intermediate_validation_schedule(warmup, interval)
