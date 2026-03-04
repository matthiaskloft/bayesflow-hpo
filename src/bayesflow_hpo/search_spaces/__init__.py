"""Search-space definitions for BayesFlow HPO."""

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    CategoricalDimension,
    Dimension,
    FloatDimension,
    IntDimension,
    SearchSpace,
)
from bayesflow_hpo.search_spaces.composite import (
    CompositeSearchSpace,
    NetworkSelectionSpace,
    SummarySelectionSpace,
)
from bayesflow_hpo.search_spaces.inference.consistency import ConsistencyModelSpace
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.inference.diffusion import DiffusionModelSpace
from bayesflow_hpo.search_spaces.inference.flow_matching import FlowMatchingSpace
from bayesflow_hpo.search_spaces.inference.stable_consistency import (
    StableConsistencyModelSpace,
)
from bayesflow_hpo.search_spaces.registry import (
    get_inference_space,
    get_summary_space,
    list_inference_spaces,
    list_summary_spaces,
    register_inference_space,
    register_summary_space,
)
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace
from bayesflow_hpo.search_spaces.summary.fusion_transformer import (
    FusionTransformerSpace,
)
from bayesflow_hpo.search_spaces.summary.set_transformer import SetTransformerSpace
from bayesflow_hpo.search_spaces.summary.time_series_network import (
    TimeSeriesNetworkSpace,
)
from bayesflow_hpo.search_spaces.summary.time_series_transformer import (
    TimeSeriesTransformerSpace,
)
from bayesflow_hpo.search_spaces.training import TrainingSpace

__all__ = [
    "BaseSearchSpace",
    "CategoricalDimension",
    "ConsistencyModelSpace",
    "CompositeSearchSpace",
    "CouplingFlowSpace",
    "DeepSetSpace",
    "Dimension",
    "DiffusionModelSpace",
    "FloatDimension",
    "FusionTransformerSpace",
    "FlowMatchingSpace",
    "IntDimension",
    "NetworkSelectionSpace",
    "SetTransformerSpace",
    "SearchSpace",
    "StableConsistencyModelSpace",
    "SummarySelectionSpace",
    "TimeSeriesNetworkSpace",
    "TimeSeriesTransformerSpace",
    "TrainingSpace",
    "get_inference_space",
    "get_summary_space",
    "list_inference_spaces",
    "list_summary_spaces",
    "register_inference_space",
    "register_summary_space",
]
