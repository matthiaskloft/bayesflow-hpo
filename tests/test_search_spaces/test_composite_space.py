"""Tests for composite search-space behavior."""

from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace
from bayesflow_hpo.search_spaces.training import TrainingSpace


class FakeTrial:
    def suggest_int(self, name, low, high, step=None, log=False):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


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
