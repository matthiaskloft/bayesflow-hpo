"""Tests for composite search-space behavior."""

from conftest import FakeTrial

from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace
from bayesflow_hpo.search_spaces.training import TrainingSpace


def test_composite_space_merges_inference_summary_and_training():
    space = CompositeSearchSpace(
        inference_space=CouplingFlowSpace(),
        summary_space=DeepSetSpace(),
        training_space=TrainingSpace(include_optional=False),
    )

    params = space.sample(FakeTrial())

    assert "cf_depth" in params
    assert "ds_summary_dim" in params
    assert "initial_lr" in params

    assert "batch_size" in params
    assert "decay_rate" in params
