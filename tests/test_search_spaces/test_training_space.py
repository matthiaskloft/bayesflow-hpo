"""Tests for the training search space and its optional couplings."""

import dataclasses

import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.base import (
    CategoricalDimension,
    FloatDimension,
    IntDimension,
)
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


def test_reference_batch_size_renames_a_custom_learning_rate_dimension():
    """Whatever the dimension was called, the sampled coordinate is `lr_ref`."""
    space = TrainingSpace(
        initial_lr=FloatDimension("peak_lr", constant=2e-3),
        batch_size=IntDimension("batch_size", constant=64),
        lr_reference_batch_size=32,
    )

    params = space.sample(FakeTrial())

    assert params["lr_ref"] == pytest.approx(2e-3)
    assert params["initial_lr"] == pytest.approx(4e-3)
    assert "peak_lr" not in params


def test_coupling_does_not_mutate_a_shared_dimension():
    """Two spaces sharing one dimension object stay independent."""
    shared = FloatDimension("initial_lr", low=1e-4, high=1e-2, log=True)

    coupled = TrainingSpace(initial_lr=shared, lr_reference_batch_size=32)
    plain = TrainingSpace(initial_lr=shared)

    assert "lr_ref" in coupled.sample(FakeTrial())
    assert plain.sample(FakeTrial())["initial_lr"] == pytest.approx(1e-4)
    assert shared.name == "initial_lr"


def test_coupling_can_be_turned_off_with_dataclasses_replace():
    """Rebuilding a space without the coupling restores direct sampling."""
    coupled = TrainingSpace(lr_reference_batch_size=32)

    plain = dataclasses.replace(coupled, lr_reference_batch_size=None)
    params = plain.sample(FakeTrial())

    assert "lr_ref" not in params
    assert params["initial_lr"] == pytest.approx(1e-4)


def test_reference_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="lr_reference_batch_size must be >= 1"):
        TrainingSpace(lr_reference_batch_size=0)


def test_reserved_reference_name_is_rejected():
    """`lr_ref` is assigned by the coupling, so it cannot be claimed."""
    with pytest.raises(ValueError, match="cannot be named"):
        TrainingSpace(
            initial_lr=FloatDimension("lr_ref", constant=1e-3),
            lr_reference_batch_size=32,
        )


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
    with pytest.raises(ValueError, match="too small"):
        TrainingSpace(
            batch_size=IntDimension("batch_size", low=32, high=256, step=32),
            epochs=IntDimension("epochs", low=10, high=50),
            simulation_budget=1000,
        )


def test_stepped_epochs_dimension_is_bounded_by_its_high_value():
    """The feasibility check uses declared bounds, which `step` only narrows."""
    space = TrainingSpace(
        batch_size=IntDimension("batch_size", constant=32),
        epochs=IntDimension("epochs", low=8, high=64, step=8),
        simulation_budget=32 * 64,
    )

    assert space.sample(FakeTrial())["num_batches"] >= 1

    with pytest.raises(ValueError, match="too small"):
        TrainingSpace(
            batch_size=IntDimension("batch_size", constant=32),
            epochs=IntDimension("epochs", low=8, high=64, step=8),
            simulation_budget=32 * 64 - 1,
        )


def test_zero_epochs_are_rejected_before_a_division_by_zero():
    """A degenerate lower bound must not abort the study at sample time."""
    with pytest.raises(ValueError, match="must be >= 1|to be\n?\\s*>= 1"):
        TrainingSpace(
            batch_size=IntDimension("batch_size", constant=64),
            epochs=IntDimension("epochs", low=0, high=10),
            simulation_budget=100_000,
        )


def test_non_integer_epochs_dimension_is_rejected():
    """A budget needs integer epoch bounds to derive `num_batches` from."""
    with pytest.raises(TypeError, match="epochs must be an IntDimension"):
        TrainingSpace(
            batch_size=IntDimension("batch_size", constant=64),
            epochs=CategoricalDimension("epochs", choices=[10, 20]),
            simulation_budget=100_000,
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
