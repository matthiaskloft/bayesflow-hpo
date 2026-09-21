"""A resumed study must compare scores using the same reduction."""

import optuna
import pytest

from bayesflow_hpo.objectives import check_aggregation_settings


def test_legacy_mean_can_resume_but_geometric_cannot():
    study = optuna.create_study()
    study.add_trial(optuna.trial.create_trial(value=0.2))
    check_aggregation_settings(study, "mean")
    with pytest.raises(ValueError, match="aggregation"):
        check_aggregation_settings(study, {"nrmse": "geometric"})


def test_mapping_round_trips_and_rejects_changed_axes():
    study = optuna.create_study()
    check_aggregation_settings(study, {"cal_error": "worst"})
    study.add_trial(optuna.trial.create_trial(value=0.2))
    check_aggregation_settings(study, {"calibration_error": "worst"})
    with pytest.raises(ValueError, match="aggregation"):
        check_aggregation_settings(study, "worst")


def test_source_check_does_not_mutate_metadata():
    study = optuna.create_study()
    check_aggregation_settings(study, "mean", record=False)
    assert "bayesflow_hpo_aggregate" not in study.user_attrs


def test_empty_mapping_is_legacy_mean():
    study = optuna.create_study()
    study.add_trial(optuna.trial.create_trial(value=0.2))
    check_aggregation_settings(study, {})
    assert study.user_attrs["bayesflow_hpo_aggregate"] == "mean"


def test_public_study_creation_records_aggregation(monkeypatch):
    from types import SimpleNamespace

    from bayesflow_hpo.api import _create_and_run_study

    objective = SimpleNamespace(config=SimpleNamespace(
        objective_metrics=["nrmse"], pruning_n_startup_trials=5,
    ))
    monkeypatch.setattr("bayesflow_hpo.api.optimize_until", lambda *a, **kw: None)
    study = _create_and_run_study(
        objective=objective, study_name="aggregation", directions=["minimize"],
        metric_names=["nrmse"], storage=None, resume=False,
        warm_start_from=None, warm_start_top_k=1, n_trials=0,
        max_total_trials=None, show_progress_bar=False, has_cost=False,
        aggregate={"nrmse": "geometric"},
    )
    assert study.user_attrs["bayesflow_hpo_aggregate"] == {"nrmse": "geometric"}


def test_warm_start_preserves_aggregation_metadata():
    from bayesflow_hpo.optimization.study import create_study

    source = create_study(
        storage=None, directions=["minimize"], metric_names=["nrmse"],
        has_cost=False,
    )
    source.set_user_attr("bayesflow_hpo_objective_schema", ["nrmse"])
    check_aggregation_settings(source, {"nrmse": "geometric"})
    source.add_trial(optuna.trial.create_trial(value=0.2))
    target = create_study(
        storage=None, directions=["minimize"], metric_names=["nrmse"],
        has_cost=False, warm_start_from=source,
    )
    assert len(target.trials) == 1
    check_aggregation_settings(target, {"nrmse": "geometric"})
    with pytest.raises(ValueError, match="aggregation"):
        check_aggregation_settings(target, "mean")

# Direct GenericObjective lifecycle guards (issue #118 regression coverage).
def _direct_objective(aggregate, sample):
    from types import SimpleNamespace

    from bayesflow_hpo.optimization.objective import GenericObjective

    objective = GenericObjective.__new__(GenericObjective)
    objective.config = SimpleNamespace(aggregate=aggregate, search_space=sample)
    return objective


class _SampleSpy:
    def __init__(self):
        self.calls = 0

    def sample(self, trial):
        self.calls += 1
        raise RuntimeError("sample reached")


def test_direct_first_geometric_trial_stamps_metadata_without_self_rejection():
    study = optuna.create_study()
    trial = study.ask()
    spy = _SampleSpy()
    objective = _direct_objective("geometric", spy)

    with pytest.raises(RuntimeError, match="sample reached"):
        objective(trial)

    assert spy.calls == 1
    assert study.user_attrs["bayesflow_hpo_aggregate"] == "geometric"


def test_direct_resume_with_same_aggregation_reaches_trial_work():
    study = optuna.create_study()
    study.set_user_attr("bayesflow_hpo_aggregate", {"nrmse": "geometric"})
    study.add_trial(optuna.trial.create_trial(value=0.2))
    trial = study.ask()
    spy = _SampleSpy()

    with pytest.raises(RuntimeError, match="sample reached"):
        _direct_objective({"nrmse": "geometric"}, spy)(trial)

    assert spy.calls == 1


def test_direct_changed_aggregation_rejects_before_sampling_after_complete_trial():
    study = optuna.create_study()
    study.set_user_attr("bayesflow_hpo_aggregate", "geometric")
    study.add_trial(optuna.trial.create_trial(value=0.2))
    trial = study.ask()
    spy = _SampleSpy()

    with pytest.raises(ValueError, match="aggregation"):
        _direct_objective("mean", spy)(trial)

    assert spy.calls == 0


def test_direct_legacy_completed_study_rejects_nonmean_before_sampling():
    study = optuna.create_study()
    study.add_trial(optuna.trial.create_trial(value=0.2))
    trial = study.ask()
    spy = _SampleSpy()

    with pytest.raises(ValueError, match="aggregation"):
        _direct_objective("geometric", spy)(trial)

    assert spy.calls == 0


def test_direct_changed_aggregation_does_not_ignore_other_running_trial():
    study = optuna.create_study()
    study.set_user_attr("bayesflow_hpo_aggregate", "mean")
    study.ask()  # Remains RUNNING in the study.
    trial = study.ask()
    spy = _SampleSpy()

    with pytest.raises(ValueError, match="aggregation"):
        _direct_objective("geometric", spy)(trial)

    assert spy.calls == 0
