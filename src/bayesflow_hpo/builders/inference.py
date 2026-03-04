"""Inference network builders."""

from __future__ import annotations

from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace


def build_inference_network(
    params: dict[str, Any],
    search_space: CompositeSearchSpace,
) -> bf.networks.InferenceNetwork:
    """Build the selected inference network from params + space."""
    return search_space.inference_space.build(params)
