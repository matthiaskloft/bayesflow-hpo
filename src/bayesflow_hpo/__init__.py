"""bayesflow_hpo public API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bayesflow-hpo")
except PackageNotFoundError:
    __version__ = "0.1.0"

from bayesflow_hpo.api import optimize
from bayesflow_hpo.builders import (
    AdapterSpec,
    PriorStandardize,
    WorkflowBuildConfig,
    build_inference_network,
    build_summary_network,
    build_workflow,
    create_adapter,
)
from bayesflow_hpo.objectives import (
    denormalize_param_count,
    get_param_count,
    normalize_param_count,
)
from bayesflow_hpo.optimization import (
    exceeds_memory_budget,
    estimate_peak_memory_mb,
    GenericObjective,
    MovingAverageEarlyStopping,
    ObjectiveConfig,
    OptunaReportCallback,
    cleanup_trial,
    create_study,
    warm_start_study,
)
from bayesflow_hpo.registration import (
    list_registered_network_spaces,
    register_custom_inference_network,
    register_custom_summary_network,
)
from bayesflow_hpo.results import (
    get_pareto_trials,
    get_workflow_metadata,
    load_workflow_with_metadata,
    plot_param_importance,
    plot_pareto_front,
    save_workflow_with_metadata,
    trials_to_dataframe,
)
from bayesflow_hpo.search_spaces import (
    CategoricalDimension,
    ConsistencyModelSpace,
    CompositeSearchSpace,
    CouplingFlowSpace,
    DeepSetSpace,
    DiffusionModelSpace,
    FloatDimension,
    FusionTransformerSpace,
    FlowMatchingSpace,
    IntDimension,
    NetworkSelectionSpace,
    SetTransformerSpace,
    StableConsistencyModelSpace,
    SummarySelectionSpace,
    TimeSeriesNetworkSpace,
    TimeSeriesTransformerSpace,
    TrainingSpace,
    list_inference_spaces,
    list_summary_spaces,
    register_inference_space,
    register_summary_space,
)
from bayesflow_hpo.utils import loguniform_float, loguniform_int
from bayesflow_hpo.validation import (
    ValidationDataset,
    generate_validation_dataset,
    load_validation_dataset,
    run_validation_pipeline,
    save_validation_dataset,
)

__all__ = [
    "CategoricalDimension",
    "ConsistencyModelSpace",
    "CompositeSearchSpace",
    "CouplingFlowSpace",
    "DeepSetSpace",
    "DiffusionModelSpace",
    "FloatDimension",
    "FusionTransformerSpace",
    "FlowMatchingSpace",
    "AdapterSpec",
    "GenericObjective",
    "IntDimension",
    "MovingAverageEarlyStopping",
    "NetworkSelectionSpace",
    "ObjectiveConfig",
    "OptunaReportCallback",
    "PriorStandardize",
    "SetTransformerSpace",
    "StableConsistencyModelSpace",
    "SummarySelectionSpace",
    "TimeSeriesNetworkSpace",
    "TimeSeriesTransformerSpace",
    "TrainingSpace",
    "ValidationDataset",
    "__version__",
    "cleanup_trial",
    "create_adapter",
    "create_study",
    "denormalize_param_count",
    "estimate_peak_memory_mb",
    "exceeds_memory_budget",
    "generate_validation_dataset",
    "get_param_count",
    "get_pareto_trials",
    "get_workflow_metadata",
    "list_inference_spaces",
    "list_registered_network_spaces",
    "list_summary_spaces",
    "load_validation_dataset",
    "load_workflow_with_metadata",
    "loguniform_float",
    "loguniform_int",
    "normalize_param_count",
    "optimize",
    "plot_param_importance",
    "plot_pareto_front",
    "register_custom_inference_network",
    "register_custom_summary_network",
    "register_inference_space",
    "register_summary_space",
    "run_validation_pipeline",
    "save_validation_dataset",
    "save_workflow_with_metadata",
    "WorkflowBuildConfig",
    "build_inference_network",
    "build_summary_network",
    "build_workflow",
    "trials_to_dataframe",
    "warm_start_study",
]
