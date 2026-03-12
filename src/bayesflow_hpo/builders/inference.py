"""Inference network builders.

Thin convenience wrapper that delegates to the search space's ``build()``
method.  Exists for API symmetry with the summary builder.
"""

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
