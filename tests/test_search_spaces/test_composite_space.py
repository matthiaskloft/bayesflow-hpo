"""Tests for composite search-space behavior."""

import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace
from bayesflow_hpo.search_spaces.training import TrainingSpace


class _DuplicateParameterSpace:
    """Minimal space returning a name owned by the training space."""

    @property
    def constants(self) -> dict[str, float]:
        """Return the colliding constant."""
        return {"initial_lr": 1e-3}

    def sample(self, trial: object) -> dict[str, float]:
        """Return the colliding sampled parameter."""
        return {"initial_lr": 1e-3}

    def build(self, params: dict[str, object]) -> object:
        """Return parameters for protocol compatibility."""
        return params


def test_composite_space_merges_inference_summary_and_training():
    space = CompositeSearchSpace(
        inference_space=CouplingFlowSpace(),
        summary_space=DeepSetSpace(),
        training_space=TrainingSpace(),
    )

    params = space.sample(FakeTrial())

    assert "cf_depth" in params
    assert "ds_summary_dim" in params
    assert "initial_lr" in params

    assert params["batch_size"] == 32


def test_composite_constants_merges_sub_spaces():
    space = CompositeSearchSpace(
        inference_space=CouplingFlowSpace(),
        summary_space=DeepSetSpace(),
        training_space=TrainingSpace(),
    )

    constants = space.constants
    assert "batch_size" not in constants


def test_composite_rejects_cross_space_parameter_collisions() -> None:
    """Composite sampling rejects names repeated across subspaces."""
    space = CompositeSearchSpace(inference_space=_DuplicateParameterSpace())

    with pytest.raises(ValueError, match="Duplicate parameter names.*initial_lr"):
        space.sample(FakeTrial())
