"""Tests for objective training failure handling and budget enforcement."""

import pytest

from bayesflow_hpo.objectives import FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_COST
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


class _FakeApproximator:
    """Approximator stub that reports a configurable param count."""

    def __init__(self, param_count: int):
        self._param_count = param_count

    def build_from_data(self, data):
        pass

    def count_params(self):
        return self._param_count


def test_objective_training_failure_sets_user_attr_and_penalty(monkeypatch):
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    fake_approx = _FakeApproximator(param_count=50_000)

    class _FakeWorkflow:
        approximator = fake_approx

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_workflow",
        lambda **kwargs: _FakeWorkflow(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    def _raise_training_error(workflow, params, callbacks):
        raise RuntimeError("You must call compile() before calling fit().")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            train_fn=_raise_training_error,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    # Default penalty: (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_COST)
    assert values == (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_COST)
    assert "training_error" in trial.user_attrs
    assert "compile" in trial.user_attrs["training_error"]


def test_objective_rejects_trial_exceeding_param_budget(monkeypatch):
    """Exact param count check rejects trials over max_param_count."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    fake_approx = _FakeApproximator(param_count=200_000)

    class _FakeWorkflow:
        approximator = fake_approx

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_workflow",
        lambda **kwargs: _FakeWorkflow(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            max_param_count=100_000,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    assert values == (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_COST)
    assert trial.user_attrs["rejected_reason"] == "param_budget"
    assert trial.user_attrs["param_count"] == 200_000


def test_objective_allows_trial_within_param_budget(monkeypatch):
    """Trials within budget pass the exact check and proceed to training."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    fake_approx = _FakeApproximator(param_count=50_000)

    class _FakeWorkflow:
        approximator = fake_approx

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_workflow",
        lambda **kwargs: _FakeWorkflow(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    train_called = []

    def _track_training(workflow, params, callbacks):
        train_called.append(True)
        raise RuntimeError("stop after confirming training was reached")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            max_param_count=100_000,
            train_fn=_track_training,
        )
    )

    trial = _FakeTrial()
    objective(trial)

    # Training was reached (not rejected by budget).
    assert len(train_called) == 1
    assert "rejected_reason" not in trial.user_attrs
    assert trial.user_attrs["param_count"] == 50_000


def test_objective_allows_trial_at_exact_budget_boundary(monkeypatch):
    """Trial with param_count == max_param_count should pass (strict >)."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    fake_approx = _FakeApproximator(param_count=100_000)

    class _FakeWorkflow:
        approximator = fake_approx

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_workflow",
        lambda **kwargs: _FakeWorkflow(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    train_called = []

    def _track_training(workflow, params, callbacks):
        train_called.append(True)
        raise RuntimeError("stop after confirming training was reached")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            max_param_count=100_000,
            train_fn=_track_training,
        )
    )

    trial = _FakeTrial()
    objective(trial)

    assert len(train_called) == 1
    assert "rejected_reason" not in trial.user_attrs
    assert trial.user_attrs["param_count"] == 100_000


def test_objective_rejects_trial_when_probe_fails(monkeypatch):
    """When the param count probe fails, the trial must be rejected."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    class _FakeWorkflow:
        class approximator:
            @staticmethod
            def build_from_data(data):
                raise RuntimeError("shape mismatch during probe")

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_workflow",
        lambda **kwargs: _FakeWorkflow(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    train_called = []

    def _track_training(workflow, params, callbacks):
        train_called.append(True)

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            train_fn=_track_training,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    # Trial should be rejected, not trained.
    assert len(train_called) == 0
    assert trial.user_attrs["rejected_reason"] == "param_probe_failed"
    assert "shape mismatch" in trial.user_attrs["param_probe_error"]
    assert values == (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_COST)


def test_objective_reraises_memory_error_from_probe(monkeypatch):
    """MemoryError during param probe must propagate, never be swallowed."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    class _FakeWorkflow:
        class approximator:
            @staticmethod
            def build_from_data(data):
                raise MemoryError("CUDA OOM")

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_workflow",
        lambda **kwargs: _FakeWorkflow(),
    )

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
        )
    )

    trial = _FakeTrial()
    with pytest.raises(MemoryError, match="CUDA OOM"):
        objective(trial)


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


def test_objective_config_rejects_invalid_cost_metric():
    """ObjectiveConfig eagerly validates cost_metric."""
    with pytest.raises(ValueError, match="Unknown cost_metric"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            batches_per_epoch=1,
            validation_data=None,
            cost_metric="unknown",
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
    assert objective._penalty() == (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_COST)


def test_objective_multi_metric_penalty_shape(monkeypatch):
    """Pareto mode with 2 metrics should return 3 penalty values."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    fake_approx = _FakeApproximator(param_count=50_000)

    class _FakeWorkflow:
        approximator = fake_approx

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_workflow",
        lambda **kwargs: _FakeWorkflow(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    def _raise_training_error(workflow, params, callbacks):
        raise RuntimeError("boom")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
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
        FAILED_TRIAL_COST,
    )
