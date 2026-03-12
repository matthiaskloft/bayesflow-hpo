"""Shared test fixtures for bayesflow_hpo."""

from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Adapter transform stubs
# ---------------------------------------------------------------------------


class FakeRename:
    """Mimics bayesflow Rename transform."""

    def __init__(self, from_key: str, to_key: str):
        self.from_key = from_key
        self.to_key = to_key


class FakeConcatenate:
    """Mimics bayesflow Concatenate transform."""

    def __init__(self, keys: list[str], into: str):
        self.keys = list(keys)
        self.into = into


class FakeBroadcast:
    """Mimics bayesflow Broadcast transform (no canonical target)."""

    def __init__(self, keys: list[str], to: str):
        self.keys = list(keys)
        self.to = to


def make_adapter(transforms):
    """Return a mock adapter with a transforms list."""
    adapter = MagicMock()
    adapter.transforms = transforms
    return adapter


def canonical_adapter():
    """Adapter with standard theta/x canonical renames."""
    return make_adapter(
        [
            FakeRename("theta", "inference_variables"),
            FakeRename("x", "summary_variables"),
        ]
    )


class FakeTrial:
    """Minimal Optuna trial stub for unit tests."""

    def suggest_int(self, name, low, high, step=None, log=False):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


class DummySimulator:
    """Simulator that generates Gaussian data."""

    def sample(self, n_sims, conditions=None, seed=None):
        rng = np.random.default_rng(seed)
        theta = rng.normal(size=(n_sims, 1))
        x = theta + rng.normal(scale=0.1, size=(n_sims, 1))
        out = {"theta": theta, "x": x}
        if conditions is not None:
            for key, value in conditions.items():
                out[key] = np.full((n_sims, 1), value)
        return out


@pytest.fixture
def fake_trial():
    return FakeTrial()


@pytest.fixture
def dummy_simulator():
    return DummySimulator()
