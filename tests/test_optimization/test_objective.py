"""Tests for objective training failure handling."""

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
    penalty = (9.9, 8.8)

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
            training_failure_penalty=penalty,
            train_fn=_raise_training_error,
        )
    )

    trial = _FakeTrial()
    values = objective(trial)

    assert values == penalty
    assert "training_error" in trial.user_attrs
    assert "compile" in trial.user_attrs["training_error"]
