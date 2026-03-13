"""Sampling entry points around search spaces.

Thin convenience wrappers for sampling hyperparameters from composite
search spaces.
"""

from __future__ import annotations

from typing import Any

from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace


def sample_hyperparameters(trial: Any, space: CompositeSearchSpace) -> dict[str, Any]:
    """Sample one parameter dictionary from a composite search space.

    Parameters
    ----------
    trial
        An ``optuna.Trial`` instance.
    space
        Composite search space to sample from.

    Returns
    -------
    dict[str, Any]
        Merged parameter dict from all sub-spaces.
    """
    return space.sample(trial)
