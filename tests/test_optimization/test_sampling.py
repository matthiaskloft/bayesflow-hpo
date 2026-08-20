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
    assert params["batch_size"] == 32  # fake trial selects lower bound


def test_training_defaults_search_updated_lr_and_batch_ranges():
    dimensions = {dim.name: dim for dim in TrainingSpace().dimensions}

    assert dimensions["initial_lr"].low == 1e-4
    assert dimensions["initial_lr"].high == 1e-2
    assert dimensions["initial_lr"].log is True
    assert dimensions["batch_size"].low == 32
    assert dimensions["batch_size"].high == 256
    assert dimensions["batch_size"].step == 32
