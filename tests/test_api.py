"""Tests for the high-level optimize() API – key inference and validation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


def _patched_optimize(validation_ds, **extra_kwargs):
    """Call optimize() with heavy internals mocked out."""
    with (
        patch("bayesflow_hpo.api.GenericObjective") as mock_obj_cls,
        patch("bayesflow_hpo.api.create_study"),
        patch("bayesflow_hpo.api.optimize_until"),
    ):
        mock_instance = MagicMock()
        mock_instance.n_objectives = 2
        mock_obj_cls.return_value = mock_instance

        optimize(
            simulator=MagicMock(),
            adapter=MagicMock(),
            validation_data=validation_ds,
            storage=None,
            **extra_kwargs,
        )

        # ObjectiveConfig passed to GenericObjective
        return mock_obj_cls.call_args[0][0]


# ---------------------------------------------------------------------------
# Inference: param_keys / data_keys inferred from ValidationDataset
# ---------------------------------------------------------------------------


def test_infer_param_keys_from_validation_data(validation_ds):
    """param_keys should be inferred from validation_data when not provided."""
    config = _patched_optimize(validation_ds)
    assert config.param_keys == ["theta"]


def test_infer_data_keys_from_validation_data(validation_ds):
    """data_keys should be inferred from validation_data when not provided."""
    config = _patched_optimize(validation_ds)
    assert config.data_keys == ["x"]


def test_matching_keys_accepted(validation_ds):
    """Providing param_keys/data_keys that match validation_data should succeed."""
    config = _patched_optimize(validation_ds, param_keys=["theta"], data_keys=["x"])
    assert config.param_keys == ["theta"]
    assert config.data_keys == ["x"]


# ---------------------------------------------------------------------------
# Mismatch: ValueError when provided keys diverge from the dataset
# ---------------------------------------------------------------------------


def test_param_keys_mismatch_raises(validation_ds):
    """ValueError raised when param_keys disagrees with validation_data."""
    with pytest.raises(ValueError, match="param_keys mismatch"):
        optimize(
            simulator=MagicMock(),
            adapter=MagicMock(),
            param_keys=["wrong"],
            data_keys=["x"],
            validation_data=validation_ds,
            storage=None,
        )


def test_data_keys_mismatch_raises(validation_ds):
    """ValueError raised when data_keys disagrees with validation_data."""
    with pytest.raises(ValueError, match="data_keys mismatch"):
        optimize(
            simulator=MagicMock(),
            adapter=MagicMock(),
            param_keys=["theta"],
            data_keys=["wrong"],
            validation_data=validation_ds,
            storage=None,
        )


# ---------------------------------------------------------------------------
# Missing keys without validation_data: TypeError
# ---------------------------------------------------------------------------


def test_missing_param_keys_raises_type_error():
    """TypeError raised when param_keys is None and no validation_data given."""
    with pytest.raises(TypeError, match="param_keys is required"):
        optimize(
            simulator=MagicMock(),
            adapter=MagicMock(),
            data_keys=["x"],
            storage=None,
        )


def test_missing_data_keys_raises_type_error():
    """TypeError raised when data_keys is None and no validation_data given."""
    with pytest.raises(TypeError, match="data_keys is required"):
        optimize(
            simulator=MagicMock(),
            adapter=MagicMock(),
            param_keys=["theta"],
            storage=None,
        )
