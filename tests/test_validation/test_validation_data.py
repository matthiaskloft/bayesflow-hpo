"""Tests for validation dataset generation/serialization."""

from pathlib import Path

import numpy as np
from conftest import DummySimulator

from bayesflow_hpo.validation.data import (
    ValidationDataset,
    generate_validation_dataset,
    load_validation_dataset,
    make_condition_grid,
    make_validation_dataset,
    save_validation_dataset,
)


def test_make_condition_grid_linspace():
    grid = make_condition_grid(linspace={"N": (10, 100, 5)})
    assert "N" in grid
    assert len(grid["N"]) == 5
    assert abs(grid["N"][0] - 10) < 1e-9
    assert abs(grid["N"][-1] - 100) < 1e-9


def test_make_condition_grid_logspace():
    grid = make_condition_grid(logspace={"lr": (1e-4, 1e-1, 4)})
    assert len(grid["lr"]) == 4
    assert grid["lr"][0] > 0
    assert grid["lr"][-1] < grid["lr"][0] * 2000


def test_make_condition_grid_values():
    grid = make_condition_grid(values={"method": ["A", "B", "C"]})
    assert grid["method"] == ["A", "B", "C"]


def test_make_condition_grid_combined():
    grid = make_condition_grid(
        linspace={"N": (10, 50, 3)},
        values={"group": [1, 2]},
    )
    assert "N" in grid
    assert "group" in grid
    assert len(grid["N"]) == 3
    assert len(grid["group"]) == 2


def test_make_validation_dataset():
    simulator = DummySimulator()
    ds = make_validation_dataset(
        simulator=simulator,
        param_keys=["theta"],
        data_keys=["x"],
        linspace={"N": (10, 50, 3)},
        sims_per_condition=5,
    )
    assert isinstance(ds, ValidationDataset)
    assert len(ds.simulations) == 3


def test_generate_validation_dataset_without_conditions():
    simulator = DummySimulator()
    dataset = generate_validation_dataset(
        simulator=simulator,
        param_keys=["theta"],
        data_keys=["x"],
        condition_grid=None,
        sims_per_condition=20,
        seed=7,
    )

    assert isinstance(dataset, ValidationDataset)
    assert len(dataset.simulations) == 1
    assert dataset.condition_labels == [{}]
    assert dataset.simulations[0]["theta"].shape[0] == 20


def test_validation_dataset_round_trip(tmp_path: Path):
    simulator = DummySimulator()
    dataset = generate_validation_dataset(
        simulator=simulator,
        param_keys=["theta"],
        data_keys=["x"],
        condition_grid={"N": [20, 40]},
        sims_per_condition=10,
        seed=11,
    )

    save_validation_dataset(dataset, tmp_path)
    loaded = load_validation_dataset(tmp_path)

    assert loaded.param_keys == dataset.param_keys
    assert loaded.data_keys == dataset.data_keys
    assert loaded.condition_labels == dataset.condition_labels
    assert len(loaded.simulations) == len(dataset.simulations)

    for idx in range(len(dataset.simulations)):
        assert np.array_equal(
            loaded.simulations[idx]["theta"],
            dataset.simulations[idx]["theta"],
        )
        assert np.array_equal(
            loaded.simulations[idx]["x"],
            dataset.simulations[idx]["x"],
        )
