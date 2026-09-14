"""Tests for the training search space and its optional couplings."""

import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.base import IntDimension
from bayesflow_hpo.search_spaces.training import TrainingSpace


def test_learning_rate_is_sampled_directly_by_default():
    """Without a reference batch size, `initial_lr` is the sampled name."""
    space = TrainingSpace()

    params = space.sample(FakeTrial())

    assert params["initial_lr"] == pytest.approx(1e-4)
    assert "lr_ref" not in params


def test_reference_batch_size_scales_the_learning_rate_linearly():
    """`initial_lr` follows `lr_ref * batch_size / reference_batch`."""
    space = TrainingSpace(
        batch_size=IntDimension("batch_size", constant=128),
        lr_reference_batch_size=32,
    )

    params = space.sample(FakeTrial())

    assert params["lr_ref"] == pytest.approx(1e-4)
    assert params["initial_lr"] == pytest.approx(1e-4 * 128 / 32)


def test_reference_batch_size_keeps_a_custom_dimension_name():
    """A user-named learning-rate dimension stays the sampled coordinate."""
    from bayesflow_hpo.search_spaces.base import FloatDimension

    space = TrainingSpace(
        initial_lr=FloatDimension("peak_lr", constant=2e-3),
        batch_size=IntDimension("batch_size", constant=64),
        lr_reference_batch_size=32,
    )

    params = space.sample(FakeTrial())

    assert params["peak_lr"] == pytest.approx(2e-3)
    assert params["initial_lr"] == pytest.approx(4e-3)


def test_reference_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="lr_reference_batch_size must be >= 1"):
        TrainingSpace(lr_reference_batch_size=0)


def test_simulation_budget_derives_num_batches():
    """`num_batches` keeps trials simulation-matched across batch sizes."""
    space = TrainingSpace(
        batch_size=IntDimension("batch_size", constant=64),
        epochs=IntDimension("epochs", constant=20),
        simulation_budget=102_400,
    )

    params = space.sample(FakeTrial())

    assert params["num_batches"] == 80
    assert params["batch_size"] * params["epochs"] * params["num_batches"] == 102_400


def test_simulation_budget_requires_an_epochs_dimension():
    with pytest.raises(ValueError, match="requires an 'epochs' dimension"):
        TrainingSpace(simulation_budget=102_400)


def test_simulation_budget_must_cover_the_largest_trial():
    """An infeasible budget is rejected at construction, not per trial."""
    with pytest.raises(ValueError, match="too small for this space"):
        TrainingSpace(
            batch_size=IntDimension("batch_size", low=32, high=256, step=32),
            epochs=IntDimension("epochs", low=10, high=50),
            simulation_budget=1000,
        )


def test_derived_values_are_not_reported_as_constants():
    """Derived couplings vary per trial, so they are not static constants."""
    space = TrainingSpace(
        batch_size=IntDimension("batch_size", constant=64),
        epochs=IntDimension("epochs", constant=20),
        simulation_budget=102_400,
        lr_reference_batch_size=32,
    )

    assert "num_batches" not in space.constants
    assert "initial_lr" not in space.constants
