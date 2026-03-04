"""Tests for validation dataset generation/serialization."""

from pathlib import Path

import numpy as np

from bayesflow_hpo.validation.data import (
    ValidationDataset,
    generate_validation_dataset,
    load_validation_dataset,
    save_validation_dataset,
)


class DummySimulator:
    def sample(self, n_sims, conditions=None, seed=None):
        rng = np.random.default_rng(seed)
        theta = rng.normal(size=(n_sims, 1))
        x = theta + rng.normal(scale=0.1, size=(n_sims, 1))

        out = {"theta": theta, "x": x}
        if conditions is not None:
            for key, value in conditions.items():
                out[key] = np.full((n_sims, 1), value)
        return out


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
