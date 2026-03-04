"""Sampling entry points around search spaces."""

from __future__ import annotations

from typing import Any

from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace


def sample_hyperparameters(trial: Any, space: CompositeSearchSpace) -> dict[str, Any]:
    """Sample one parameter dictionary from a composite search space."""
    return space.sample(trial)
