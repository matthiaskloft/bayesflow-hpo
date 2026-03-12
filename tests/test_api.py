"""Tests for the high-level optimize() API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from conftest import FakeRename, canonical_adapter, make_adapter

from bayesflow_hpo.api import optimize
from bayesflow_hpo.validation.data import ValidationDataset


@pytest.fixture()
def validation_ds():
    """Minimal pre-built ValidationDataset used across tests."""
    return ValidationDataset(
        simulations=[{"theta": np.zeros((5, 1)), "x": np.zeros((5, 1))}],
        condition_labels=[{}],
        param_keys=["theta"],
        data_keys=["x"],
        seed=0,
    )


def _patched_optimize(validation_ds=None, adapter=None, **extra_kwargs):
    """Call optimize() with heavy internals mocked out."""
    if adapter is None:
        adapter = canonical_adapter()

    with (
        patch("bayesflow_hpo.api.GenericObjective") as mock_obj_cls,
        patch("bayesflow_hpo.api.create_study"),
        patch("bayesflow_hpo.api.optimize_until"),
    ):
        mock_instance = MagicMock()
        mock_instance.n_objectives = 2
        mock_obj_cls.return_value = mock_instance

        kwargs = {"storage": None}
        if validation_ds is not None:
            kwargs["validation_data"] = validation_ds
        kwargs.update(extra_kwargs)

        optimize(
            simulator=MagicMock(),
            adapter=adapter,
            **kwargs,
        )

        # ObjectiveConfig passed to GenericObjective
        return mock_obj_cls.call_args[0][0]


# ---------------------------------------------------------------------------
# Keys derived from adapter
# ---------------------------------------------------------------------------


def test_keys_derived_from_adapter():
    """param_keys and data_keys are derived from adapter transforms."""
    config = _patched_optimize()
    assert config.param_keys == ["theta"]
    assert config.data_keys == ["x"]


def test_keys_derived_from_adapter_match_validation_data(validation_ds):
    """Adapter-derived keys that match validation_data succeed."""
    config = _patched_optimize(validation_ds)
    assert config.param_keys == ["theta"]
    assert config.data_keys == ["x"]


def test_keys_fallback_to_validation_data(validation_ds):
    """When adapter has no canonical keys, fall back to validation_data."""
    adapter = make_adapter([])  # no canonical transforms
    config = _patched_optimize(validation_ds, adapter=adapter)
    assert config.param_keys == ["theta"]
    assert config.data_keys == ["x"]


def test_explicit_keys_override_adapter():
    """Explicit param_keys/data_keys take precedence over adapter inference."""
    config = _patched_optimize(
        param_keys=["my_param"],
        data_keys=["my_data"],
    )
    assert config.param_keys == ["my_param"]
    assert config.data_keys == ["my_data"]


# ---------------------------------------------------------------------------
# Mismatch: ValueError when keys diverge from the dataset
# ---------------------------------------------------------------------------


def test_param_keys_mismatch_raises(validation_ds):
    """ValueError when param_keys disagree with validation_data."""
    adapter = make_adapter(
        [
            FakeRename("wrong", "inference_variables"),
            FakeRename("x", "summary_variables"),
        ]
    )
    with pytest.raises(ValueError, match="param_keys"):
        _patched_optimize(validation_ds, adapter=adapter)


def test_data_keys_mismatch_raises(validation_ds):
    """ValueError when data_keys disagree with validation_data."""
    adapter = make_adapter(
        [
            FakeRename("theta", "inference_variables"),
            FakeRename("wrong", "summary_variables"),
        ]
    )
    with pytest.raises(ValueError, match="data_keys"):
        _patched_optimize(validation_ds, adapter=adapter)


# ---------------------------------------------------------------------------
# Missing keys: TypeError when adapter has no canonical keys and no dataset
# ---------------------------------------------------------------------------


def test_missing_param_keys_raises_type_error():
    """TypeError when adapter lacks inference_variables and no validation_data."""
    adapter = make_adapter(
        [
            FakeRename("x", "summary_variables"),
        ]
    )
    with pytest.raises(TypeError, match="param_keys"):
        _patched_optimize(adapter=adapter)


def test_missing_data_keys_raises_type_error():
    """TypeError when adapter lacks summary_variables and no validation_data."""
    adapter = make_adapter(
        [
            FakeRename("theta", "inference_variables"),
        ]
    )
    with pytest.raises(TypeError, match="data_keys"):
        _patched_optimize(adapter=adapter)


# ---------------------------------------------------------------------------
# inference_conditions validation
# ---------------------------------------------------------------------------


def test_inference_conditions_validated_against_condition_labels():
    """ValueError when inference_conditions not in condition_labels."""
    adapter = make_adapter(
        [
            FakeRename("theta", "inference_variables"),
            FakeRename("x", "summary_variables"),
            FakeRename("N", "inference_conditions"),
        ]
    )
    vd = ValidationDataset(
        simulations=[{"theta": np.zeros((5, 1)), "x": np.zeros((5, 1))}],
        condition_labels=[{"T": 100}],  # N is not here
        param_keys=["theta"],
        data_keys=["x"],
        seed=0,
    )
    with pytest.raises(ValueError, match="inference_conditions"):
        _patched_optimize(vd, adapter=adapter)


def test_inference_conditions_valid_passes():
    """No error when inference_conditions are present in condition_labels."""
    adapter = make_adapter(
        [
            FakeRename("theta", "inference_variables"),
            FakeRename("x", "summary_variables"),
            FakeRename("N", "inference_conditions"),
        ]
    )
    vd = ValidationDataset(
        simulations=[{"theta": np.zeros((5, 1)), "x": np.zeros((5, 1))}],
        condition_labels=[{"N": 50}],
        param_keys=["theta"],
        data_keys=["x"],
        seed=0,
    )
    config = _patched_optimize(vd, adapter=adapter)
    assert config.inference_conditions == ["N"]


# ---------------------------------------------------------------------------
# Early stopping parameters
# ---------------------------------------------------------------------------


def test_optimize_forwards_early_stopping_params_to_objective_config(monkeypatch):
    """optimize() forwards early_stopping_patience/window to ObjectiveConfig."""
    captured = {}

    class _FakeGenericObjective:
        def __init__(self, config):
            captured["config"] = config
            self.n_objectives = 2

        def __call__(self, trial):  # pragma: no cover
            return (0.0, 0.0)

    monkeypatch.setattr(
        "bayesflow_hpo.api.GenericObjective",
        _FakeGenericObjective,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.api.create_study",
        lambda **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.api.optimize_until",
        lambda study, objective, **kwargs: None,
    )

    import bayesflow_hpo.api as api

    fake_simulator = MagicMock()
    fake_search_space = MagicMock()
    fake_search_space.inference_space = MagicMock()
    fake_search_space.summary_space = None

    api.optimize(
        simulator=fake_simulator,
        adapter=canonical_adapter(),
        search_space=fake_search_space,
        n_trials=1,
        epochs=10,
        batches_per_epoch=5,
        early_stopping_patience=10,
        early_stopping_window=5,
        storage=None,
    )

    config = captured["config"]
    assert config.early_stopping_patience == 10
    assert config.early_stopping_window == 5


def test_optimize_early_stopping_default_values(monkeypatch):
    """optimize() applies default patience=5, window=7 when not specified."""
    captured = {}

    class _FakeGenericObjective:
        def __init__(self, config):
            captured["config"] = config
            self.n_objectives = 2

        def __call__(self, trial):  # pragma: no cover
            return (0.0, 0.0)

    monkeypatch.setattr(
        "bayesflow_hpo.api.GenericObjective",
        _FakeGenericObjective,
    )
    monkeypatch.setattr(
        "bayesflow_hpo.api.create_study",
        lambda **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "bayesflow_hpo.api.optimize_until",
        lambda study, objective, **kwargs: None,
    )

    import bayesflow_hpo.api as api

    fake_simulator = MagicMock()
    fake_search_space = MagicMock()
    fake_search_space.inference_space = MagicMock()
    fake_search_space.summary_space = None

    api.optimize(
        simulator=fake_simulator,
        adapter=canonical_adapter(),
        search_space=fake_search_space,
        n_trials=1,
        epochs=10,
        batches_per_epoch=5,
        storage=None,
    )

    config = captured["config"]
    assert config.early_stopping_patience == 5
    assert config.early_stopping_window == 7
