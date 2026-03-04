"""Registry helpers for search spaces."""

from __future__ import annotations

from collections.abc import Callable

from bayesflow_hpo.search_spaces.base import SearchSpace

from bayesflow_hpo.search_spaces.inference import (
    ConsistencyModelSpace,
    CouplingFlowSpace,
    DiffusionModelSpace,
    FlowMatchingSpace,
    StableConsistencyModelSpace,
)
from bayesflow_hpo.search_spaces.summary import (
    DeepSetSpace,
    FusionTransformerSpace,
    SetTransformerSpace,
    TimeSeriesNetworkSpace,
    TimeSeriesTransformerSpace,
)


INFERENCE_SPACE_FACTORIES: dict[str, Callable[[], SearchSpace]] = {
    "coupling_flow": CouplingFlowSpace,
    "flow_matching": FlowMatchingSpace,
    "diffusion_model": DiffusionModelSpace,
    "consistency_model": ConsistencyModelSpace,
    "stable_consistency_model": StableConsistencyModelSpace,
}

INFERENCE_ALIASES: dict[str, str] = {
    "couplingflow": "coupling_flow",
    "cf": "coupling_flow",
    "flowmatching": "flow_matching",
    "fm": "flow_matching",
    "diffusion": "diffusion_model",
    "diffusionmodel": "diffusion_model",
    "dm": "diffusion_model",
    "consistency": "consistency_model",
    "consistencymodel": "consistency_model",
    "cm": "consistency_model",
    "stable_consistency": "stable_consistency_model",
    "stableconsistencymodel": "stable_consistency_model",
    "scm": "stable_consistency_model",
}

SUMMARY_SPACE_FACTORIES: dict[str, Callable[[], SearchSpace]] = {
    "deep_set": DeepSetSpace,
    "set_transformer": SetTransformerSpace,
    "time_series_network": TimeSeriesNetworkSpace,
    "time_series_transformer": TimeSeriesTransformerSpace,
    "fusion_transformer": FusionTransformerSpace,
}

SUMMARY_ALIASES: dict[str, str] = {
    "deepset": "deep_set",
    "ds": "deep_set",
    "settransformer": "set_transformer",
    "st": "set_transformer",
    "timeseriesnetwork": "time_series_network",
    "tsn": "time_series_network",
    "timeseriestransformer": "time_series_transformer",
    "tst": "time_series_transformer",
    "fusiontransformer": "fusion_transformer",
    "ft": "fusion_transformer",
}


def register_inference_space(
    name: str,
    factory: Callable[[], SearchSpace],
    aliases: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a custom inference search-space factory."""
    canonical = name.lower().strip()
    if not overwrite and canonical in INFERENCE_SPACE_FACTORIES:
        raise KeyError(f"Inference space already registered: {name}")

    INFERENCE_SPACE_FACTORIES[canonical] = factory
    for alias in aliases or []:
        alias_key = alias.lower().strip()
        if not overwrite and alias_key in INFERENCE_ALIASES:
            raise KeyError(f"Inference alias already registered: {alias}")
        INFERENCE_ALIASES[alias_key] = canonical


def register_summary_space(
    name: str,
    factory: Callable[[], SearchSpace],
    aliases: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a custom summary search-space factory."""
    canonical = name.lower().strip()
    if not overwrite and canonical in SUMMARY_SPACE_FACTORIES:
        raise KeyError(f"Summary space already registered: {name}")

    SUMMARY_SPACE_FACTORIES[canonical] = factory
    for alias in aliases or []:
        alias_key = alias.lower().strip()
        if not overwrite and alias_key in SUMMARY_ALIASES:
            raise KeyError(f"Summary alias already registered: {alias}")
        SUMMARY_ALIASES[alias_key] = canonical


def list_inference_spaces() -> list[str]:
    """List canonical inference space names currently registered."""
    return sorted(INFERENCE_SPACE_FACTORIES.keys())


def list_summary_spaces() -> list[str]:
    """List canonical summary space names currently registered."""
    return sorted(SUMMARY_SPACE_FACTORIES.keys())


def _resolve_inference_name(name: str) -> str:
    key = name.lower().strip()
    if key in INFERENCE_SPACE_FACTORIES:
        return key
    return INFERENCE_ALIASES.get(key, key)


def _resolve_summary_name(name: str) -> str:
    key = name.lower().strip()
    if key in SUMMARY_SPACE_FACTORIES:
        return key
    return SUMMARY_ALIASES.get(key, key)


def get_inference_space(name: str):
    """Construct an inference search space from a short name."""
    resolved = _resolve_inference_name(name)
    factory = INFERENCE_SPACE_FACTORIES.get(resolved)
    if factory is not None:
        return factory()
    raise KeyError(f"Unknown inference space: {name}")


def get_summary_space(name: str):
    """Construct a summary search space from a short name."""
    resolved = _resolve_summary_name(name)
    factory = SUMMARY_SPACE_FACTORIES.get(resolved)
    if factory is not None:
        return factory()
    raise KeyError(f"Unknown summary space: {name}")
