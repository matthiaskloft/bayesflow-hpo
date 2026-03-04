"""Inference-network search spaces."""

from bayesflow_hpo.search_spaces.inference.consistency import ConsistencyModelSpace
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.inference.diffusion import DiffusionModelSpace
from bayesflow_hpo.search_spaces.inference.flow_matching import FlowMatchingSpace
from bayesflow_hpo.search_spaces.inference.stable_consistency import (
    StableConsistencyModelSpace,
)

__all__ = [
    "ConsistencyModelSpace",
    "CouplingFlowSpace",
    "DiffusionModelSpace",
    "FlowMatchingSpace",
    "StableConsistencyModelSpace",
]
