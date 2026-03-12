"""Builder registry for custom network factories.

Parallel to the search-space registry, this stores ``params → network``
callables for custom networks.  When a user registers a custom network,
they can provide both a search space (defines *what* to tune) and a
builder (defines *how* to construct the network from sampled params).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

INFERENCE_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {}
SUMMARY_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {}


def register_inference_builder(
    name: str,
    builder: Callable[[dict[str, Any]], Any],
    overwrite: bool = False,
) -> None:
    """Register a custom inference builder."""
    key = name.lower().strip()
    if not overwrite and key in INFERENCE_BUILDERS:
        raise KeyError(f"Inference builder already registered: {name}")
    INFERENCE_BUILDERS[key] = builder


def register_summary_builder(
    name: str,
    builder: Callable[[dict[str, Any]], Any],
    overwrite: bool = False,
) -> None:
    """Register a custom summary builder."""
    key = name.lower().strip()
    if not overwrite and key in SUMMARY_BUILDERS:
        raise KeyError(f"Summary builder already registered: {name}")
    SUMMARY_BUILDERS[key] = builder


def get_inference_builder(name: str) -> Callable[[dict[str, Any]], Any]:
    """Return a registered inference builder."""
    key = name.lower().strip()
    if key not in INFERENCE_BUILDERS:
        raise KeyError(f"Unknown inference builder: {name}")
    return INFERENCE_BUILDERS[key]


def get_summary_builder(name: str) -> Callable[[dict[str, Any]], Any]:
    """Return a registered summary builder."""
    key = name.lower().strip()
    if key not in SUMMARY_BUILDERS:
        raise KeyError(f"Unknown summary builder: {name}")
    return SUMMARY_BUILDERS[key]


def list_inference_builders() -> list[str]:
    """List all registered inference builders."""
    return sorted(INFERENCE_BUILDERS.keys())


def list_summary_builders() -> list[str]:
    """List all registered summary builders."""
    return sorted(SUMMARY_BUILDERS.keys())
