"""Tests for objective training failure handling."""

import pytest

from bayesflow_hpo.objectives import FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_PARAM_SCORE
from bayesflow_hpo.optimization.objective import GenericObjective, ObjectiveConfig


class _FakeTrial:
    def __init__(self):
        self.number = 0
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _FakeInferenceSpace:
    def build(self, params):
        return object()


class _FakeSearchSpace:
    def __init__(self):
        self.inference_space = _FakeInferenceSpace()
        self.summary_space = None

    def sample(self, trial):
        return {"initial_lr": 1e-3}


def test_objective_training_failure_sets_user_attr_and_penalty(monkeypatch):
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_param_count",
        lambda params: 10,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_workflow",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    def _raise_training_error(workflow, params, callbacks):
        raise RuntimeError("You must call compile() before calling fit().")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            train_fn=_raise_training_error,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    # Default penalty: (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_PARAM_SCORE)
    assert values == (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_PARAM_SCORE)
    assert "training_error" in trial.user_attrs
    assert "compile" in trial.user_attrs["training_error"]


def test_objective_config_rejects_invalid_mode():
    """ObjectiveConfig eagerly validates objective_mode."""
    with pytest.raises(ValueError, match="Unknown objective_mode"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            objective_mode="paerto",  # typo
        )


def test_n_objectives_mean_mode_with_multiple_metrics():
    """Mean mode with objective_metrics still returns 2 objectives."""
    objective = GenericObjective(
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            objective_metrics=["calibration_error", "nrmse"],
            objective_mode="mean",
        )
    )
    assert objective.n_objectives == 2
    assert objective._penalty() == (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_PARAM_SCORE)


def test_objective_multi_metric_penalty_shape(monkeypatch):
    """Pareto mode with 2 metrics should return 3 penalty values."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_param_count",
        lambda params: 10,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_workflow",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    def _raise_training_error(workflow, params, callbacks):
        raise RuntimeError("boom")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            objective_metrics=["calibration_error", "nrmse"],
            objective_mode="pareto",
            train_fn=_raise_training_error,
        )
    )

    assert objective.n_objectives == 3
    trial = _FakeTrial()
    values = objective(trial)
    assert len(values) == 3
    assert values == (
        FAILED_TRIAL_CAL_ERROR,
        FAILED_TRIAL_CAL_ERROR,
        FAILED_TRIAL_PARAM_SCORE,
    )
