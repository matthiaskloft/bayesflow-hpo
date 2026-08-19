"""Tests for objective training failure handling and budget enforcement."""

import logging

import numpy as np
import pytest

from bayesflow_hpo.objectives import FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_COST
from bayesflow_hpo.optimization.objective import (
    GenericObjective,
    ObjectiveConfig,
    _extract_best_training_loss,
    _training_loss_fallback,
    _validate_metric_keys,
)
from bayesflow_hpo.validation.data import ValidationDataset

_DUMMY_VALIDATION_DATA = ValidationDataset(
    simulations=[],
    condition_labels=[],
    param_keys=["p"],
    data_keys=["x"],
    seed=0,
)

_DUMMY_VALIDATION_DATA_1COND = ValidationDataset(
    simulations=[{"x": np.zeros((2, 1)), "p": np.zeros((2, 1))}],
    condition_labels=[{}],
    param_keys=["p"],
    data_keys=["x"],
    seed=0,
)


class _FakeStudy:
    """Minimal study stub for PeriodicValidationCallback compatibility."""
    directions = ["minimize", "minimize", "minimize"]

    def get_trials(self, deepcopy=False, states=None):
        return []


class _FakeTrial:
    def __init__(self):
        self.number = 0
        self.user_attrs = {}
        self.params = {}
        self.study = _FakeStudy()

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value


class _FakeInferenceSpace:
    def build(self, params):
        return object()


class _FakeSearchSpace:
    def __init__(self):
        self.inference_space = _FakeInferenceSpace()
        self.summary_space = None

    def sample(self, trial):
        return {"initial_lr": 1e-3}


class _PerTrialBudgetSearchSpace(_FakeSearchSpace):
    def sample(self, trial):
        return {
            "initial_lr": 1e-3,
            "epochs": 3,
            "num_batches": 7,
        }


class _FakeApproximator:
    """Approximator stub that reports a configurable param count."""

    def __init__(self, param_count: int):
        self._param_count = param_count

    def build_from_data(self, data):
        pass

    def count_params(self):
        return self._param_count

    def compile(self, *args, **kwargs):
        pass


def test_objective_training_failure_sets_user_attr_and_penalty(monkeypatch):
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    fake_approx = _FakeApproximator(param_count=50_000)

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_continuous_approximator",
        lambda params, adapter, search_space: fake_approx,
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

    def _raise_training_error(approximator, simulator, hparams, callbacks):
        raise RuntimeError("You must call compile() before calling fit().")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            train_fn=_raise_training_error,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    # Default: pareto mode with 2 metrics → 3 objectives
    assert len(values) == 3
    assert values[-1] == FAILED_TRIAL_COST
    assert "training_error" in trial.user_attrs
    assert "compile" in trial.user_attrs["training_error"]


def test_objective_rejects_trial_exceeding_param_budget(monkeypatch):
    """Exact param count check rejects trials over max_param_count."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    fake_approx = _FakeApproximator(param_count=200_000)

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_continuous_approximator",
        lambda params, adapter, search_space: fake_approx,
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
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            max_param_count=100_000,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    assert len(values) == 3  # pareto default: 2 metrics + cost
    assert values[-1] == FAILED_TRIAL_COST
    assert trial.user_attrs["rejected_reason"] == "param_budget"
    assert trial.user_attrs["param_count"] == 200_000


def test_objective_allows_trial_within_param_budget(monkeypatch):
    """Trials within budget pass the exact check and proceed to training."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    fake_approx = _FakeApproximator(param_count=50_000)

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_continuous_approximator",
        lambda params, adapter, search_space: fake_approx,
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

    def _track_training(approximator, simulator, hparams, callbacks):
        train_called.append(True)
        raise RuntimeError("stop after confirming training was reached")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
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

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_continuous_approximator",
        lambda params, adapter, search_space: fake_approx,
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

    def _track_training(approximator, simulator, hparams, callbacks):
        train_called.append(True)
        raise RuntimeError("stop after confirming training was reached")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
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

    class _BrokenApprox:
        def compile(self, *args, **kwargs):
            pass

        def build_from_data(self, data):
            raise RuntimeError("shape mismatch during probe")

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_continuous_approximator",
        lambda params, adapter, search_space: _BrokenApprox(),
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

    def _track_training(approximator, simulator, hparams, callbacks):
        train_called.append(True)

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            train_fn=_track_training,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    # Trial should be rejected, not trained.
    assert len(train_called) == 0
    assert trial.user_attrs["rejected_reason"] == "param_probe_failed"
    assert "shape mismatch" in trial.user_attrs["param_probe_error"]
    assert len(values) == 3  # pareto default
    assert values[-1] == FAILED_TRIAL_COST


def test_objective_reraises_memory_error_from_probe(monkeypatch):
    """MemoryError during param probe must propagate, never be swallowed."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )

    cleanup_called = []
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: cleanup_called.append(True),
    )

    class _OOMApprox:
        def compile(self, *args, **kwargs):
            pass

        def build_from_data(self, data):
            raise MemoryError("CUDA OOM")

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_continuous_approximator",
        lambda params, adapter, search_space: _OOMApprox(),
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
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
        )
    )

    trial = _FakeTrial()
    with pytest.raises(MemoryError, match="CUDA OOM"):
        objective(trial)

    # GPU cache must be released before the MemoryError propagates.
    assert cleanup_called == [True]


def test_objective_config_early_stopping_defaults():
    """Fixed-budget mode runs to completion by default."""
    config = ObjectiveConfig(
        simulator=object(),
        adapter=object(),
        search_space=_FakeSearchSpace(),
        epochs=1,
        num_batches=1,
        validation_data=_DUMMY_VALIDATION_DATA,
    )
    assert config.training_mode == "fixed_budget"
    assert config.early_stopping_patience is None
    assert config.early_stopping_window == 7
    assert config.lr_warmup_epochs == 0


def test_objective_config_early_stopping_custom_values():
    """ObjectiveConfig accepts custom early_stopping_patience and window."""
    config = ObjectiveConfig(
        simulator=object(),
        adapter=object(),
        search_space=_FakeSearchSpace(),
        epochs=1,
        num_batches=1,
        validation_data=_DUMMY_VALIDATION_DATA,
        training_mode="open_ended",
        early_stopping_patience=10,
        early_stopping_window=5,
    )
    assert config.early_stopping_patience == 10
    assert config.early_stopping_window == 5


def test_objective_config_rejects_unknown_early_stopping_monitor():
    with pytest.raises(ValueError, match="must be 'objective_mean'"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            validation_data=_DUMMY_VALIDATION_DATA,
            early_stopping_monitor="recovery",
        )


def test_open_ended_mode_resolves_default_patience():
    config = ObjectiveConfig(
        simulator=object(),
        adapter=object(),
        search_space=_FakeSearchSpace(),
        validation_data=_DUMMY_VALIDATION_DATA,
        training_mode="open_ended",
    )
    assert config.early_stopping_patience == 5
    assert config.lr_warmup_epochs == 1


def test_open_ended_mode_rejects_zero_warmup():
    with pytest.raises(ValueError, match="lr_warmup_epochs must be >= 1"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            validation_data=_DUMMY_VALIDATION_DATA,
            training_mode="open_ended",
            lr_warmup_epochs=0,
        )


def test_exact_warmup_steps_take_precedence_over_epoch_setting():
    config = ObjectiveConfig(
        simulator=object(),
        adapter=object(),
        search_space=_FakeSearchSpace(),
        validation_data=_DUMMY_VALIDATION_DATA,
        training_mode="open_ended",
        lr_warmup_epochs=0,
        lr_warmup_steps=25,
    )
    assert config.lr_warmup_steps == 25


def test_warmup_choices_must_be_unique():
    with pytest.raises(ValueError, match="choices must be unique"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            validation_data=_DUMMY_VALIDATION_DATA,
            lr_warmup_epochs=[1, 1],
        )


def test_objective_config_rejects_early_stopping_with_fixed_budget():
    """A finite-horizon schedule cannot stop before its horizon."""
    with pytest.raises(ValueError, match="cannot be set in fixed_budget"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            validation_data=_DUMMY_VALIDATION_DATA,
            early_stopping_patience=5,
        )


def test_objective_config_rejects_unknown_training_mode():
    with pytest.raises(ValueError, match="Unknown training_mode"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            validation_data=_DUMMY_VALIDATION_DATA,
            training_mode="adaptive",
        )


@pytest.mark.parametrize(
    ("warmup_kwargs", "expected_steps", "sampled_name"),
    [
        ({"lr_warmup_epochs": [1, 2]}, 7, "lr_warmup_epochs"),
        (
            {"lr_warmup_epochs": [1, 2], "lr_warmup_steps": [5, 8]},
            5,
            "lr_warmup_steps",
        ),
    ],
)
def test_fixed_budget_schedule_uses_per_trial_training_length(
    monkeypatch, warmup_kwargs, expected_steps, sampled_name,
):
    captured = {}
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective._make_cosine_decay_optimizer",
        lambda initial_lr, decay_steps, warmup_steps: captured.update(
            initial_lr=initial_lr,
            decay_steps=decay_steps,
            warmup_steps=warmup_steps,
        )
        or object(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    class _Simulator:
        def sample(self, shape):
            return {}

    class _Adapter:
        def __call__(self, data):
            return data

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_Simulator(),
            adapter=_Adapter(),
            search_space=_PerTrialBudgetSearchSpace(),
            validation_data=_DUMMY_VALIDATION_DATA_1COND,
            epochs=200,
            num_batches=50,
            build_approximator_fn=lambda params: _FakeApproximator(100),
            train_fn=lambda approximator, simulator, params, callbacks: None,
            validate_fn=lambda approximator, data, samples: {
                "calibration_error": 0.1,
                "nrmse": 0.2,
            },
            **warmup_kwargs,
        )
    )
    trial = _FakeTrial()

    objective(trial)

    assert captured == {
        "initial_lr": 1e-3,
        "decay_steps": 21,
        "warmup_steps": expected_steps,
    }
    assert trial.user_attrs["epochs"] == 3
    assert trial.user_attrs["num_batches"] == 7
    assert trial.params[sampled_name] == warmup_kwargs[sampled_name][0]
    assert trial.user_attrs["lr_warmup_steps"] == expected_steps
    assert trial.user_attrs["lr_warmup_epochs"] == pytest.approx(
        expected_steps / 7
    )
    assert trial.user_attrs["peak_learning_rate"] == pytest.approx(1e-3)


def test_objective_config_rejects_invalid_mode():
    """ObjectiveConfig eagerly validates objective_mode."""
    with pytest.raises(ValueError, match="Unknown objective_mode"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            objective_mode="paerto",  # typo
        )


def test_objective_config_rejects_invalid_report_frequency():
    """ObjectiveConfig rejects report_frequency < 1."""
    with pytest.raises(ValueError, match="report_frequency must be >= 1"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            report_frequency=0,
        )


def test_objective_config_rejects_invalid_cost_metric():
    """ObjectiveConfig eagerly validates cost_metric."""
    with pytest.raises(ValueError, match="Unknown cost_metric"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            cost_metric="unknown",
        )


def test_objective_config_rejects_invalid_hard_constraint_direction():
    """ObjectiveConfig eagerly validates hard metric constraint direction."""
    with pytest.raises(ValueError, match="Invalid hard metric constraint direction"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            metric_constraints_hard=[("calibration_error", 0.2, "sideways")],
        )


def test_objective_config_rejects_none_validation_data():
    """ObjectiveConfig raises TypeError when validation_data is None."""
    with pytest.raises(TypeError, match="validation_data must be a ValidationDataset"):
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            validation_data=None,
        )


def test_n_objectives_mean_mode_with_multiple_metrics():
    """Mean mode with objective_metrics still returns 2 objectives."""
    objective = GenericObjective(
        ObjectiveConfig(
            simulator=object(),
            adapter=object(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
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

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_continuous_approximator",
        lambda params, adapter, search_space: fake_approx,
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

    def _raise_training_error(approximator, simulator, hparams, callbacks):
        raise RuntimeError("boom")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
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


def test_objective_config_default_metrics_and_mode():
    """ObjectiveConfig has correct defaults for metrics and mode."""
    config = ObjectiveConfig(
        simulator=object(),
        adapter=object(),
        search_space=_FakeSearchSpace(),
        validation_data=_DUMMY_VALIDATION_DATA,
    )
    assert config.objective_metrics == ["calibration_error", "nrmse"]
    assert config.objective_mode == "pareto"


def test_objective_config_hooks_default_none():
    """Hook fields default to None."""
    config = ObjectiveConfig(
        simulator=object(),
        adapter=object(),
        search_space=_FakeSearchSpace(),
        validation_data=_DUMMY_VALIDATION_DATA,
    )
    assert config.build_approximator_fn is None
    assert config.train_fn is None
    assert config.validate_fn is None


def test_objective_uses_custom_build_fn(monkeypatch):
    """Custom build_approximator_fn is called instead of default builder."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    build_calls = []

    def custom_builder(hparams):
        build_calls.append(hparams)
        return _FakeApproximator(param_count=10_000)

    def _raise(approx, sim, hp, cb):
        raise RuntimeError("stop")

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
            validation_data=_DUMMY_VALIDATION_DATA,
            epochs=1,
            num_batches=1,
            build_approximator_fn=custom_builder,
            train_fn=_raise,
        )
    )

    trial = _FakeTrial()
    objective(trial)
    assert len(build_calls) == 1


def test_training_config_injected_into_hparams(monkeypatch):
    """epochs and num_batches are injected into hparams before training."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    captured_hparams = {}

    def custom_train(approximator, simulator, hparams, callbacks):
        captured_hparams.update(hparams)
        raise RuntimeError("stop")

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
            validation_data=_DUMMY_VALIDATION_DATA,
            epochs=42,
            num_batches=7,
            build_approximator_fn=lambda hp: _FakeApproximator(10_000),
            train_fn=custom_train,
        )
    )

    trial = _FakeTrial()
    objective(trial)
    assert captured_hparams["epochs"] == 42
    assert captured_hparams["num_batches"] == 7


def test_validate_metric_keys_passes_clean_dict():
    """Clean dict with all keys passes through unchanged."""
    raw = {"calibration_error": 0.05, "nrmse": 0.1}
    result = _validate_metric_keys(raw, ["calibration_error", "nrmse"])
    assert result == raw


def test_validate_metric_keys_fills_missing():
    """Missing keys are replaced with penalty value."""
    raw = {"calibration_error": 0.05}
    result = _validate_metric_keys(raw, ["calibration_error", "nrmse"])
    assert result["nrmse"] == FAILED_TRIAL_CAL_ERROR


def test_validate_metric_keys_replaces_nan():
    """NaN values are replaced with penalty value."""
    raw = {"calibration_error": float("nan"), "nrmse": 0.1}
    result = _validate_metric_keys(raw, ["calibration_error", "nrmse"])
    assert result["calibration_error"] == FAILED_TRIAL_CAL_ERROR
    assert result["nrmse"] == 0.1


def test_validate_metric_keys_replaces_inf():
    """Inf values are replaced with penalty value."""
    raw = {"calibration_error": float("inf"), "nrmse": 0.1}
    result = _validate_metric_keys(raw, ["calibration_error", "nrmse"])
    assert result["calibration_error"] == FAILED_TRIAL_CAL_ERROR


def test_objective_validate_fn_error_returns_penalty(monkeypatch):
    """Custom validate_fn that raises should result in penalty values, not crash."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    fake_approx = _FakeApproximator(param_count=10_000)

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    def _raise_validate(approx, vd, n):
        raise RuntimeError("custom validation exploded")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            build_approximator_fn=lambda hp: fake_approx,
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=_raise_validate,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    assert len(values) == 3  # pareto default
    assert values[-1] == FAILED_TRIAL_COST
    assert "validation_error" in trial.user_attrs
    assert "custom validation exploded" in trial.user_attrs["validation_error"]


def test_objective_custom_validate_fn_called_over_default(monkeypatch):
    """When validate_fn is provided, it's called instead of run_validation_pipeline."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    fake_approx = _FakeApproximator(param_count=10_000)
    validate_calls = []

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    def custom_validate(approx, vd, n):
        validate_calls.append(True)
        return {"calibration_error": 0.05, "nrmse": 0.1}

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA_1COND,
            build_approximator_fn=lambda hp: fake_approx,
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=custom_validate,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    assert len(validate_calls) == 1
    assert len(values) == 3
    assert values[0] == pytest.approx(0.05)  # calibration_error


# --- Validation failure fallback tests ---


def test_extract_best_training_loss_from_callback():
    """Extracts best_ma_loss from a MovingAverageEarlyStopping callback."""
    from bayesflow_hpo.optimization.callbacks import MovingAverageEarlyStopping

    cb = MovingAverageEarlyStopping()
    cb.best_ma_loss = 0.25
    assert _extract_best_training_loss([object(), cb]) == 0.25


def test_extract_best_training_loss_none_when_missing():
    """Returns None when no MovingAverageEarlyStopping in callbacks."""
    assert _extract_best_training_loss([object()]) is None


def test_extract_best_training_loss_none_when_inf():
    """Returns None when best_ma_loss is still inf (no improvement)."""
    from bayesflow_hpo.optimization.callbacks import MovingAverageEarlyStopping

    cb = MovingAverageEarlyStopping()
    # best_ma_loss defaults to np.inf
    assert _extract_best_training_loss([cb]) is None


def test_training_loss_fallback_pareto_param_count():
    """Fallback in pareto mode with param_count cost uses normalized params."""
    penalty = (1.0, 1.0, 1e6)
    values = _training_loss_fallback(
        best_training_loss=0.3,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="pareto",
        param_count=50_000,
        max_param_count=1_000_000,
        cost_metric="param_count",
        penalty=penalty,
    )
    assert len(values) == 3
    assert values[0] == pytest.approx(0.3)
    assert values[1] == pytest.approx(0.3)
    # Cost should be a real normalized value, not the penalty
    assert values[2] < 1e6
    assert values[2] > 0.0


def test_training_loss_fallback_pareto_inference_time():
    """Fallback with inference_time cost uses penalty cost (no inference ran)."""
    from bayesflow_hpo.objectives import FAILED_TRIAL_COST

    penalty = (1.0, 1.0, FAILED_TRIAL_COST)
    values = _training_loss_fallback(
        best_training_loss=0.3,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="pareto",
        param_count=50_000,
        max_param_count=1_000_000,
        cost_metric="inference_time",
        penalty=penalty,
    )
    assert len(values) == 3
    assert values[0] == pytest.approx(0.3)
    assert values[1] == pytest.approx(0.3)
    assert values[2] == FAILED_TRIAL_COST


def test_training_loss_fallback_mean():
    """Fallback in mean mode returns (clamped_loss, cost)."""
    penalty = (1.0, 1e6)
    values = _training_loss_fallback(
        best_training_loss=0.15,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="mean",
        param_count=50_000,
        max_param_count=1_000_000,
        cost_metric="param_count",
        penalty=penalty,
    )
    assert len(values) == 2
    assert values[0] == pytest.approx(0.15)
    assert values[1] < 1e6


def test_training_loss_fallback_clamps_above_one():
    """Training loss > 1 is clamped to 1.0."""
    values = _training_loss_fallback(
        best_training_loss=2.5,
        objective_metrics=["calibration_error"],
        objective_mode="pareto",
        param_count=50_000,
        max_param_count=1_000_000,
        cost_metric="param_count",
        penalty=(1.0, 1e6),
    )
    assert values[0] == pytest.approx(1.0)


# --- _validate_metric_keys with penalty_values ---


def test_validate_metric_keys_uses_per_metric_penalty():
    """Per-metric penalty values override the default constant."""
    raw = {"calibration_error": float("nan")}
    penalties = {"calibration_error": 1.0, "nrmse": 99.0}
    result = _validate_metric_keys(
        raw, ["calibration_error", "nrmse"], penalty_values=penalties,
    )
    assert result["calibration_error"] == 1.0
    assert result["nrmse"] == 99.0


def test_validate_metric_keys_penalty_fallback_for_unknown_key():
    """Keys missing from penalty_values fall back to FAILED_TRIAL_CAL_ERROR."""
    raw = {}
    penalties = {"calibration_error": 0.5}
    result = _validate_metric_keys(
        raw, ["calibration_error", "nrmse"], penalty_values=penalties,
    )
    assert result["calibration_error"] == 0.5
    assert result["nrmse"] == FAILED_TRIAL_CAL_ERROR


def test_validate_metric_keys_none_penalty_values_uses_default():
    """penalty_values=None preserves original behavior."""
    raw = {"calibration_error": float("inf")}
    result = _validate_metric_keys(
        raw, ["calibration_error", "nrmse"], penalty_values=None,
    )
    assert result["calibration_error"] == FAILED_TRIAL_CAL_ERROR
    assert result["nrmse"] == FAILED_TRIAL_CAL_ERROR


def test_validate_metric_keys_empty_penalty_dict_uses_default():
    """Empty penalty_values dict falls back to FAILED_TRIAL_CAL_ERROR."""
    raw = {}
    result = _validate_metric_keys(
        raw, ["calibration_error", "nrmse"], penalty_values={},
    )
    assert result["calibration_error"] == FAILED_TRIAL_CAL_ERROR
    assert result["nrmse"] == FAILED_TRIAL_CAL_ERROR


def test_validate_metric_keys_replaces_inf_with_custom_penalty():
    """Inf values are replaced with per-metric penalty, not default."""
    raw = {"calibration_error": float("inf"), "nrmse": float("inf")}
    penalties = {"calibration_error": 0.5, "nrmse": 99.0}
    result = _validate_metric_keys(
        raw, ["calibration_error", "nrmse"], penalty_values=penalties,
    )
    assert result["calibration_error"] == 0.5
    assert result["nrmse"] == 99.0


# --- Compile TypeError narrowing ---


def test_optimizer_creation_typeerror_not_swallowed(monkeypatch):
    """TypeError from _make_cosine_decay_optimizer propagates as compile failure."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    fake_approx = _FakeApproximator(param_count=50_000)

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.build_continuous_approximator",
        lambda params, adapter, search_space: fake_approx,
    )

    def _raise_type_error(initial_lr, decay_steps, warmup_steps):
        raise TypeError("bad arg type")

    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective._make_cosine_decay_optimizer",
        _raise_type_error,
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
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    # TypeError should be caught by the outer except Exception handler,
    # resulting in a rejected trial with compile_failed reason.
    assert trial.user_attrs["rejected_reason"] == "compile_failed"
    assert "bad arg type" in trial.user_attrs["compile_error"]
    assert values[-1] == FAILED_TRIAL_COST


def test_training_loss_fallback_none_returns_penalty():
    """When training loss is None, falls back to full penalty."""
    penalty = (1.0, 1.0, 1e6)
    values = _training_loss_fallback(
        best_training_loss=None,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="pareto",
        param_count=50_000,
        max_param_count=1_000_000,
        cost_metric="param_count",
        penalty=penalty,
    )
    assert values == penalty


def test_objective_validation_failure_uses_training_loss_fallback(monkeypatch):
    """End-to-end: validation failure falls back to training loss, not penalty."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    fake_approx = _FakeApproximator(param_count=10_000)

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    def _train_with_loss(approximator, simulator, hparams, callbacks):
        """Simulate training by setting best_ma_loss on early stopping cb."""
        from bayesflow_hpo.optimization.callbacks import MovingAverageEarlyStopping

        for cb in callbacks:
            if isinstance(cb, MovingAverageEarlyStopping):
                cb.best_ma_loss = 0.42
                break

    def _raise_validate(approx, vd, n):
        raise RuntimeError("OOM during validation")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            build_approximator_fn=lambda hp: fake_approx,
            train_fn=_train_with_loss,
            validate_fn=_raise_validate,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    # Should use training loss (0.42) for metrics, not penalty (1.0)
    assert len(values) == 3
    assert values[0] == pytest.approx(0.42)
    assert values[1] == pytest.approx(0.42)
    # Default cost_metric is "inference_time" — no inference ran, so
    # cost falls back to FAILED_TRIAL_COST.
    assert values[2] == FAILED_TRIAL_COST
    assert "validation_error" in trial.user_attrs
    assert trial.user_attrs["validation_fallback"] == "training_loss"


def test_objective_validation_failure_without_training_loss_sets_penalty_attr(
    monkeypatch,
):
    """When training loss is unavailable, validation_fallback is 'penalty'."""
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: None,
    )

    fake_approx = _FakeApproximator(param_count=10_000)

    class _FakeSimulator:
        def sample(self, shape):
            return {}

    class _FakeAdapter:
        def __call__(self, data):
            return data

    def _raise_validate(approx, vd, n):
        raise RuntimeError("OOM during validation")

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            epochs=1,
            num_batches=1,
            validation_data=_DUMMY_VALIDATION_DATA,
            build_approximator_fn=lambda hp: fake_approx,
            train_fn=lambda approx, sim, hp, cb: None,  # no loss set
            validate_fn=_raise_validate,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    # No training loss available → full penalty
    assert values == (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_COST)
    assert trial.user_attrs["validation_fallback"] == "penalty"


# --- _training_loss_fallback edge-case tests ---


def _single_metric_pareto_fallback(loss):
    """Call _training_loss_fallback with 1 pareto metric and param_count cost."""
    return _training_loss_fallback(
        best_training_loss=loss,
        objective_metrics=["calibration_error"],
        objective_mode="pareto",
        param_count=50_000,
        max_param_count=1_000_000,
        cost_metric="param_count",
        penalty=(1.0, 1e6),
    )


def test_training_loss_fallback_clamps_negative():
    """Negative training loss is clamped to 0.0."""
    assert _single_metric_pareto_fallback(-0.5)[0] == pytest.approx(0.0)


def test_training_loss_fallback_exact_zero():
    """Training loss of exactly 0.0 passes through unchanged."""
    assert _single_metric_pareto_fallback(0.0)[0] == pytest.approx(0.0)


def test_training_loss_fallback_exact_one():
    """Training loss of exactly 1.0 passes through unchanged."""
    assert _single_metric_pareto_fallback(1.0)[0] == pytest.approx(1.0)


def test_training_loss_fallback_single_metric_pareto():
    """Pareto mode with 1 metric returns (clamped_loss, cost_score)."""
    values = _single_metric_pareto_fallback(0.25)
    assert len(values) == 2
    assert values[0] == pytest.approx(0.25)
    assert 0.0 < values[1] < 1e6  # normalized param count


def test_checkpoint_pool_receives_mean_of_pareto_metrics(monkeypatch):
    """Step 10 must rank/evict by the mean of all metrics, not just the first.

    Regression test: previously ``values[0]`` (the first pareto metric
    only) was passed to the checkpoint pool, silently ignoring every
    other objective metric.
    """
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
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

    saved_calls = []

    class _RecordingCheckpointPool:
        def maybe_save(self, trial_number, objective_value, approximator):
            saved_calls.append(objective_value)
            return True

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=_FakeSimulator(),
            adapter=_FakeAdapter(),
            search_space=_FakeSearchSpace(),
            validation_data=_DUMMY_VALIDATION_DATA_1COND,
            epochs=1,
            num_batches=1,
            build_approximator_fn=lambda hp: _FakeApproximator(10_000),
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=lambda approx, vd, n: {
                "calibration_error": 0.05,  # first pareto metric — good
                "nrmse": 0.95,  # second pareto metric — bad
            },
            objective_metrics=["calibration_error", "nrmse"],
            objective_mode="pareto",
            checkpoint_pool=_RecordingCheckpointPool(),
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    # values = (calibration_error, nrmse, cost_score)
    assert values[0] == pytest.approx(0.05)
    assert values[1] == pytest.approx(0.95)

    # The checkpoint pool must have received the mean of the two
    # metrics (excluding cost), not just values[0].
    assert len(saved_calls) == 1
    assert saved_calls[0] == pytest.approx(np.mean([0.05, 0.95]))
    assert saved_calls[0] != values[0]


def test_hard_metric_constraints_pass_through_when_satisfied(monkeypatch):
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
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
            validation_data=_DUMMY_VALIDATION_DATA_1COND,
            epochs=1,
            num_batches=1,
            build_approximator_fn=lambda hp: _FakeApproximator(10_000),
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=lambda approx, vd, n: {
                "calibration_error": 0.05,
                "nrmse": 0.10,
            },
            metric_constraints_hard=[("calibration_error", 0.2, "above")],
        )
    )

    trial = _FakeTrial()
    values = objective(trial)
    assert trial.user_attrs.get("rejected_reason") is None
    assert values[0] == pytest.approx(0.05)


def test_hard_metric_constraints_violation_rejects_and_cleans_up(monkeypatch):
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
    )
    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.cleanup_trial",
        lambda: cleanup_calls.append(1),
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
            validation_data=_DUMMY_VALIDATION_DATA_1COND,
            epochs=1,
            num_batches=1,
            build_approximator_fn=lambda hp: _FakeApproximator(10_000),
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=lambda approx, vd, n: {
                "calibration_error": 0.35,
                "nrmse": 0.10,
            },
            metric_constraints_hard=[("calibration_error", 0.2, "above")],
        )
    )

    trial = _FakeTrial()
    values = objective(trial)
    assert trial.user_attrs["rejected_reason"] == "metric_constraint"
    assert values[-1] == FAILED_TRIAL_COST
    assert len(cleanup_calls) == 1


def test_hard_metric_constraints_missing_metric_warns_and_skips(
    monkeypatch, caplog
):
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
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
            validation_data=_DUMMY_VALIDATION_DATA_1COND,
            epochs=1,
            num_batches=1,
            build_approximator_fn=lambda hp: _FakeApproximator(10_000),
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=lambda approx, vd, n: {
                "calibration_error": 0.05,
                "nrmse": 0.10,
            },
            metric_constraints_hard=[("sbc_ks", 0.1, "above")],
        )
    )

    with caplog.at_level(logging.WARNING):
        trial = _FakeTrial()
        values = objective(trial)
    assert trial.user_attrs.get("rejected_reason") is None
    assert values[0] == pytest.approx(0.05)
    assert "missing metric" in caplog.text


def test_hard_metric_constraints_multiple_partial_violation(monkeypatch):
    monkeypatch.setattr(
        "bayesflow_hpo.optimization.objective.estimate_peak_memory_mb",
        lambda params: 1.0,
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
            validation_data=_DUMMY_VALIDATION_DATA_1COND,
            epochs=1,
            num_batches=1,
            build_approximator_fn=lambda hp: _FakeApproximator(10_000),
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=lambda approx, vd, n: {
                "calibration_error": 0.05,  # 0.05 <= 0.2, passes "above"
                "nrmse": 0.04,  # 0.04 < 0.05, violates "below"
            },
            metric_constraints_hard=[
                ("calibration_error", 0.2, "above"),
                ("nrmse", 0.05, "below"),
            ],
        )
    )

    trial = _FakeTrial()
    values = objective(trial)
    assert trial.user_attrs["rejected_reason"] == "metric_constraint"
    assert values[-1] == FAILED_TRIAL_COST
