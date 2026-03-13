"""Tests for infer_keys_from_adapter and adapter-based key inference in optimize()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from conftest import FakeBroadcast, FakeConcatenate, FakeRename, make_adapter

from bayesflow_hpo.api import infer_keys_from_adapter, optimize

# ---------------------------------------------------------------------------
# infer_keys_from_adapter: unit tests
# ---------------------------------------------------------------------------


class TestInferKeysFromAdapter:
    def test_empty_adapter(self):
        adapter = make_adapter([])
        result = infer_keys_from_adapter(adapter)
        assert result == {
            "param_keys": None,
            "data_keys": None,
            "inference_conditions": None,
        }

    def test_rename_inference_variables(self):
        adapter = make_adapter(
            [
                FakeRename("theta", "inference_variables"),
            ]
        )
        result = infer_keys_from_adapter(adapter)
        assert result["param_keys"] == ["theta"]
        assert result["data_keys"] is None
        assert result["inference_conditions"] is None

    def test_rename_all_canonical_keys(self):
        adapter = make_adapter(
            [
                FakeRename("theta", "inference_variables"),
                FakeRename("x", "summary_variables"),
                FakeRename("N", "inference_conditions"),
            ]
        )
        result = infer_keys_from_adapter(adapter)
        assert result["param_keys"] == ["theta"]
        assert result["data_keys"] == ["x"]
        assert result["inference_conditions"] == ["N"]

    def test_concatenate_into_inference_variables(self):
        adapter = make_adapter(
            [
                FakeConcatenate(
                    ["alpha", "beta"], into="inference_variables"
                ),
                FakeRename("x", "summary_variables"),
            ]
        )
        result = infer_keys_from_adapter(adapter)
        assert result["param_keys"] == ["alpha", "beta"]
        assert result["data_keys"] == ["x"]

    def test_non_canonical_transforms_ignored(self):
        adapter = make_adapter(
            [
                FakeRename("theta", "inference_variables"),
                FakeBroadcast(["theta"], to="b_group"),
                FakeRename("x", "summary_variables"),
            ]
        )
        result = infer_keys_from_adapter(adapter)
        assert result["param_keys"] == ["theta"]
        assert result["data_keys"] == ["x"]

    def test_multiple_renames_to_same_role(self):
        adapter = make_adapter(
            [
                FakeRename("alpha", "inference_variables"),
                FakeRename("beta", "inference_variables"),
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


def _make_fake_search_space():
    space = MagicMock()
    space.inference_space = MagicMock()
    space.summary_space = None
    return space


def _patched_optimize(adapter, **extra_kwargs):
    """Call optimize() with heavy internals mocked out."""
    with (
        patch("bayesflow_hpo.api.GenericObjective") as mock_obj_cls,
        patch("bayesflow_hpo.api.create_study"),
        patch("bayesflow_hpo.api.optimize_until"),
        patch("bayesflow_hpo.api.check_pipeline"),
        patch(
            "bayesflow_hpo.api.generate_validation_dataset"
        ) as mock_gen,
    ):
        mock_instance = MagicMock()
        mock_instance.n_objectives = 3
        mock_obj_cls.return_value = mock_instance
        mock_gen.return_value = MagicMock()

        kwargs = {
            "storage": None,
            "search_space": _make_fake_search_space(),
        }
        kwargs.update(extra_kwargs)

        optimize(
            simulator=MagicMock(),
            adapter=adapter,
            **kwargs,
        )

        return mock_obj_cls.call_args[0][0]


class TestOptimizeDeriveKeys:
    def test_derive_keys_from_adapter(self):
        """optimize() succeeds with canonical adapter."""
        adapter = make_adapter(
            [
                FakeRename("theta", "inference_variables"),
                FakeRename("x", "summary_variables"),
            ]
        )
        _patched_optimize(adapter)

    def test_no_canonical_keys_raises(self):
        """TypeError when adapter lacks canonical keys."""
        adapter = make_adapter(
            [
                FakeBroadcast(["theta"], to="b_group"),
            ]
        )
        with pytest.raises(TypeError, match="param_keys"):
            _patched_optimize(adapter)

    def test_missing_data_keys_raises(self):
        """TypeError when adapter lacks summary_variables."""
        adapter = make_adapter(
            [
                FakeRename("theta", "inference_variables"),
            ]
        )
        with pytest.raises(TypeError, match="data_keys"):
            _patched_optimize(adapter)
