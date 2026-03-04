"""Custom network registration helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bayesflow_hpo.builders.registry import (
    register_inference_builder,
    register_summary_builder,
)
from bayesflow_hpo.search_spaces.base import SearchSpace
from bayesflow_hpo.search_spaces.registry import (
    list_inference_spaces,
    list_summary_spaces,
    register_inference_space,
    register_summary_space,
)


def register_custom_inference_network(
    name: str,
    space_factory: Callable[[], SearchSpace],
    builder: Callable[[dict[str, Any]], Any] | None = None,
    aliases: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a custom inference network search space and optional builder."""
    register_inference_space(
        name=name,
        factory=space_factory,
        aliases=aliases,
        overwrite=overwrite,
    )
    if builder is not None:
        register_inference_builder(name=name, builder=builder, overwrite=overwrite)
        for alias in aliases or []:
            register_inference_builder(name=alias, builder=builder, overwrite=overwrite)


def register_custom_summary_network(
    name: str,
    space_factory: Callable[[], SearchSpace],
    builder: Callable[[dict[str, Any]], Any] | None = None,
    aliases: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a custom summary network search space and optional builder."""
    register_summary_space(
        name=name,
        factory=space_factory,
        aliases=aliases,
        overwrite=overwrite,
    )
    if builder is not None:
        register_summary_builder(name=name, builder=builder, overwrite=overwrite)
        for alias in aliases or []:
            register_summary_builder(name=alias, builder=builder, overwrite=overwrite)


def list_registered_network_spaces() -> dict[str, list[str]]:
    """List currently available canonical network space names."""
    return {
        "inference": list_inference_spaces(),
        "summary": list_summary_spaces(),
    }
