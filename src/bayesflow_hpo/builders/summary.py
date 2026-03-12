"""Summary network builders.

Thin convenience wrapper that delegates to the search space's ``build()``
method, returning ``None`` when no summary space is configured.
"""

from __future__ import annotations

from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace


def build_summary_network(
    params: dict[str, Any],
    search_space: CompositeSearchSpace,
) -> bf.networks.SummaryNetwork | None:
    """Build the selected summary network from params + space."""
    if search_space.summary_space is None:
        return None
    return search_space.summary_space.build(params)
