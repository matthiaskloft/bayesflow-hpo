"""Inference adapter from approximator to batch posterior samples.

Creates a closure that translates from the validation pipeline's
``(sim_data, n_samples)`` calling convention to BayesFlow's
``approximator.sample(conditions=..., num_samples=...)`` API.

For multi-parameter models, posterior draws are concatenated on the
last axis so that metrics receive a single ``(n_sims, n_samples, n_params)``
array and can index into individual parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def make_bayesflow_infer_fn(
    approximator: Any,
    param_keys: list[str],
    data_keys: list[str],
) -> Callable[[dict[str, Any], int], np.ndarray]:
    """Create inference fn: `(sim_data, n_samples) -> draws`.

    For multi-parameter inference, parameters are concatenated on the last axis
    before returning draws of shape `(n_sims, n_samples, n_params)`.
    """

    def infer_fn(sim_data: dict[str, Any], n_posterior_samples: int) -> np.ndarray:
        conditions = {k: sim_data[k] for k in data_keys if k in sim_data}
        post_draws = approximator.sample(
            conditions=conditions, num_samples=int(n_posterior_samples),
        )

        if len(param_keys) == 1:
            draws = np.asarray(post_draws[param_keys[0]])
            if draws.ndim == 3 and draws.shape[-1] == 1:
                draws = np.squeeze(draws, axis=-1)
            return draws

        draw_parts = [np.asarray(post_draws[key]) for key in param_keys]
        normalized_parts = []
        for part in draw_parts:
            if part.ndim == 2:
                normalized_parts.append(part[..., None])
            else:
                normalized_parts.append(part)
        return np.concatenate(normalized_parts, axis=-1)

    return infer_fn
