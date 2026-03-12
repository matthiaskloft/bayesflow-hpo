"""Tests for the high-level optimize() API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bayesflow_hpo.api import optimize
from bayesflow_hpo.validation.data import ValidationDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRename:
    """Mimics bayesflow Rename transform."""

    def __init__(self, from_key: str, to_key: str):
        self.from_key = from_key
        self.to_key = to_key


def _make_adapter(transforms):
    """Return a mock adapter with a transforms list."""
    adapter = MagicMock()
    adapter.transforms = transforms
    return adapter


def _canonical_adapter():
    """Adapter with standard theta/x canonical renames."""
    return _make_adapter(
        [
            _FakeRename("theta", "inference_variables"),
            _FakeRename("x", "summary_variables"),
        ]
    )


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
        adapter = _canonical_adapter()

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
    adapter = _make_adapter([])  # no canonical transforms
    config = _patched_optimize(validation_ds, adapter=adapter)
    assert config.param_keys == ["theta"]
    assert config.data_keys == ["x"]


# ---------------------------------------------------------------------------
# Mismatch: ValueError when adapter keys diverge from the dataset
# ---------------------------------------------------------------------------


def test_param_keys_mismatch_raises(validation_ds):
    """ValueError when adapter-derived param_keys disagree with validation_data."""
    adapter = _make_adapter(
        [
            _FakeRename("wrong", "inference_variables"),
            _FakeRename("x", "summary_variables"),
        ]
    )
    with pytest.raises(ValueError, match="param_keys"):
        _patched_optimize(validation_ds, adapter=adapter)


def test_data_keys_mismatch_raises(validation_ds):
    """ValueError when adapter-derived data_keys disagree with validation_data."""
    adapter = _make_adapter(
        [
            _FakeRename("theta", "inference_variables"),
            _FakeRename("wrong", "summary_variables"),
        ]
    )
    with pytest.raises(ValueError, match="data_keys"):
        _patched_optimize(validation_ds, adapter=adapter)


# ---------------------------------------------------------------------------
# Missing keys: TypeError when adapter has no canonical keys and no dataset
# ---------------------------------------------------------------------------


def test_missing_param_keys_raises_type_error():
    """TypeError when adapter lacks inference_variables and no validation_data."""
    adapter = _make_adapter(
        [
            _FakeRename("x", "summary_variables"),
        ]
    )
    with pytest.raises(TypeError, match="param_keys"):
        _patched_optimize(adapter=adapter)


def test_missing_data_keys_raises_type_error():
    """TypeError when adapter lacks summary_variables and no validation_data."""
    adapter = _make_adapter(
        [
            _FakeRename("theta", "inference_variables"),
        ]
    )
    with pytest.raises(TypeError, match="data_keys"):
        _patched_optimize(adapter=adapter)


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
        adapter=_canonical_adapter(),
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
        adapter=_canonical_adapter(),
        search_space=fake_search_space,
        n_trials=1,
        epochs=10,
        batches_per_epoch=5,
        storage=None,
    )

    config = captured["config"]
    assert config.early_stopping_patience == 5
    assert config.early_stopping_window == 7
