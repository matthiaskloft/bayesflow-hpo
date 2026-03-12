"""Tests for infer_keys_from_adapter and adapter-based key inference in optimize()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bayesflow_hpo.api import infer_keys_from_adapter, optimize
from bayesflow_hpo.validation.data import ValidationDataset

# ---------------------------------------------------------------------------
# Helpers: lightweight transform stubs
# ---------------------------------------------------------------------------


class _FakeRename:
    """Mimics bayesflow Rename transform."""

    def __init__(self, from_key: str, to_key: str):
        self.from_key = from_key
        self.to_key = to_key


class _FakeConcatenate:
    """Mimics bayesflow Concatenate transform."""

    def __init__(self, keys: list[str], into: str):
        self.keys = list(keys)
        self.into = into


class _FakeBroadcast:
    """Mimics bayesflow Broadcast transform (no canonical target)."""

    def __init__(self, keys: list[str], to: str):
        self.keys = list(keys)
        self.to = to


def _make_adapter(transforms):
    """Return a mock adapter with a transforms list."""
    adapter = MagicMock()
    adapter.transforms = transforms
    return adapter


# ---------------------------------------------------------------------------
# infer_keys_from_adapter: unit tests
# ---------------------------------------------------------------------------


class TestInferKeysFromAdapter:
    def test_empty_adapter(self):
        adapter = _make_adapter([])
        result = infer_keys_from_adapter(adapter)
        assert result == {
            "param_keys": None,
            "data_keys": None,
            "inference_conditions": None,
        }

    def test_rename_inference_variables(self):
        adapter = _make_adapter(
            [
                _FakeRename("theta", "inference_variables"),
            ]
        )
        result = infer_keys_from_adapter(adapter)
        assert result["param_keys"] == ["theta"]
        assert result["data_keys"] is None
        assert result["inference_conditions"] is None

    def test_rename_all_canonical_keys(self):
        adapter = _make_adapter(
            [
                _FakeRename("theta", "inference_variables"),
                _FakeRename("x", "summary_variables"),
                _FakeRename("N", "inference_conditions"),
            ]
        )
        result = infer_keys_from_adapter(adapter)
        assert result["param_keys"] == ["theta"]
        assert result["data_keys"] == ["x"]
        assert result["inference_conditions"] == ["N"]

    def test_concatenate_into_inference_variables(self):
        adapter = _make_adapter(
            [
                _FakeConcatenate(["alpha", "beta"], into="inference_variables"),
                _FakeRename("x", "summary_variables"),
            ]
        )
        result = infer_keys_from_adapter(adapter)
        assert result["param_keys"] == ["alpha", "beta"]
        assert result["data_keys"] == ["x"]

    def test_non_canonical_transforms_ignored(self):
        adapter = _make_adapter(
            [
                _FakeRename("theta", "inference_variables"),
                _FakeBroadcast(["theta"], to="b_group"),
                _FakeRename("x", "summary_variables"),
            ]
        )
        result = infer_keys_from_adapter(adapter)
        assert result["param_keys"] == ["theta"]
        assert result["data_keys"] == ["x"]

    def test_multiple_renames_to_same_role(self):
        adapter = _make_adapter(
            [
                _FakeRename("alpha", "inference_variables"),
                _FakeRename("beta", "inference_variables"),
            ]
        )
        result = infer_keys_from_adapter(adapter)
        assert result["param_keys"] == ["alpha", "beta"]

    def test_no_transforms_attribute(self):
        adapter = MagicMock(spec=[])  # no .transforms
        result = infer_keys_from_adapter(adapter)
        assert result == {
            "param_keys": None,
            "data_keys": None,
            "inference_conditions": None,
        }


# ---------------------------------------------------------------------------
# optimize() integration: keys derived from adapter
# ---------------------------------------------------------------------------


def _patched_optimize(adapter, **extra_kwargs):
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
            adapter=adapter,
            storage=None,
            **extra_kwargs,
        )

        # ObjectiveConfig passed to GenericObjective
        return mock_obj_cls.call_args[0][0]


class TestOptimizeDeriveKeys:
    def test_derive_param_and_data_keys_from_adapter(self):
        adapter = _make_adapter(
            [
                _FakeRename("theta", "inference_variables"),
                _FakeRename("x", "summary_variables"),
            ]
        )
        config = _patched_optimize(adapter)
        assert config.param_keys == ["theta"]
        assert config.data_keys == ["x"]

    def test_derive_inference_conditions_from_adapter(self):
        adapter = _make_adapter(
            [
                _FakeRename("theta", "inference_variables"),
                _FakeRename("x", "summary_variables"),
                _FakeRename("N", "inference_conditions"),
            ]
        )
        config = _patched_optimize(adapter)
        assert config.inference_conditions == ["N"]

    def test_adapter_keys_match_validation_data(self):
        adapter = _make_adapter(
            [
                _FakeRename("theta", "inference_variables"),
                _FakeRename("x", "summary_variables"),
            ]
        )
        vd = ValidationDataset(
            simulations=[{"theta": np.zeros((5, 1)), "x": np.zeros((5, 1))}],
            condition_labels=[{}],
            param_keys=["theta"],
            data_keys=["x"],
            seed=0,
        )
        config = _patched_optimize(adapter, validation_data=vd)
        assert config.param_keys == ["theta"]
        assert config.data_keys == ["x"]

    def test_adapter_keys_mismatch_validation_data_raises(self):
        adapter = _make_adapter(
            [
                _FakeRename("theta", "inference_variables"),
                _FakeRename("x", "summary_variables"),
            ]
        )
        vd = ValidationDataset(
            simulations=[{"alpha": np.zeros((5, 1)), "x": np.zeros((5, 1))}],
            condition_labels=[{}],
            param_keys=["alpha"],
            data_keys=["x"],
            seed=0,
        )
        with pytest.raises(ValueError, match="param_keys"):
            _patched_optimize(adapter, validation_data=vd)

    def test_no_canonical_keys_no_validation_data_raises(self):
        adapter = _make_adapter(
            [
                _FakeBroadcast(["theta"], to="b_group"),
            ]
        )
        with pytest.raises(TypeError, match="param_keys"):
            _patched_optimize(adapter)

    def test_fallback_to_validation_data_keys(self):
        """When adapter lacks canonical keys, fall back to validation_data."""
        adapter = _make_adapter([])
        vd = ValidationDataset(
            simulations=[{"theta": np.zeros((5, 1)), "x": np.zeros((5, 1))}],
            condition_labels=[{}],
            param_keys=["theta"],
            data_keys=["x"],
            seed=0,
        )
        config = _patched_optimize(adapter, validation_data=vd)
        assert config.param_keys == ["theta"]
        assert config.data_keys == ["x"]
