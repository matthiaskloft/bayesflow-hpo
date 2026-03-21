"""Tests for bayesflow_hpo.optimization.sampling."""

from bayesflow_hpo.optimization.sampling import sample_hyperparameters
from bayesflow_hpo.search_spaces import (
    CompositeSearchSpace,
    CouplingFlowSpace,
    DeepSetSpace,
    TrainingSpace,
)


class _FakeTrial:
    def suggest_int(self, name, low, high, step=None, log=False):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


def test_sample_returns_dict():
    space = CompositeSearchSpace(
        inference_space=CouplingFlowSpace(),
        summary_space=DeepSetSpace(),
        training_space=TrainingSpace(),
    )
    params = sample_hyperparameters(_FakeTrial(), space)
    assert isinstance(params, dict)
    assert len(params) > 0


def test_sample_contains_inference_and_training_keys():
    space = CompositeSearchSpace(
        inference_space=CouplingFlowSpace(),
        training_space=TrainingSpace(),
    )
    params = sample_hyperparameters(_FakeTrial(), space)
    assert "cf_depth" in params  # inference key
    assert "initial_lr" in params  # training key
    assert params["batch_size"] == 256  # training constant
