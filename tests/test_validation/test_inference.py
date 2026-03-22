"""Tests for make_bayesflow_infer_fn data_keys validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn
from bayesflow_hpo.validation.pipeline import run_validation_pipeline


def _make_approximator(param_keys: list[str]) -> MagicMock:
    """Create a mock approximator that returns dummy posterior draws."""
    approx = MagicMock()
    approx.sample.return_value = {
        k: np.random.randn(2, 10, 1) for k in param_keys
    }
    return approx


class TestDataKeysValidation:
    def test_missing_data_keys_raises_keyerror(self):
        """Passing available_keys that miss a required data_key raises KeyError."""
        with pytest.raises(KeyError, match="missing_key"):
            make_bayesflow_infer_fn(
                approximator=_make_approximator(["theta"]),
                param_keys=["theta"],
                data_keys=["x", "missing_key"],
                available_keys={"x", "y"},
            )

    def test_available_keys_none_skips_check(self):
        """When available_keys is None, no upfront validation occurs."""
        fn = make_bayesflow_infer_fn(
            approximator=_make_approximator(["theta"]),
            param_keys=["theta"],
            data_keys=["x"],
            available_keys=None,
        )
        assert callable(fn)

    def test_all_keys_present_succeeds(self):
        """When all data_keys are in available_keys, construction succeeds."""
        fn = make_bayesflow_infer_fn(
            approximator=_make_approximator(["theta"]),
            param_keys=["theta"],
            data_keys=["x", "y"],
            available_keys={"x", "y", "z"},
        )
        assert callable(fn)

    def test_closure_raises_on_missing_key(self):
        """The infer_fn closure raises KeyError when sim_data lacks a data_key."""
        fn = make_bayesflow_infer_fn(
            approximator=_make_approximator(["theta"]),
            param_keys=["theta"],
            data_keys=["x", "y"],
            available_keys=None,
        )
        with pytest.raises(KeyError, match="y"):
            fn({"x": np.ones((2, 5))}, n_posterior_samples=10)


class TestPipelinePassesAvailableKeys:
    def test_pipeline_raises_on_mismatched_data_keys(self):
        """run_validation_pipeline detects data_keys missing from simulations."""
        vdata = ValidationDataset(
            simulations=[{"x": np.ones((5, 3))}],
            condition_labels=[{"cond": "a"}],
            param_keys=["theta"],
            data_keys=["x", "missing_key"],
            seed=0,
        )
        with pytest.raises(KeyError, match="missing_key"):
            run_validation_pipeline(
                approximator=_make_approximator(["theta"]),
                validation_data=vdata,
            )
