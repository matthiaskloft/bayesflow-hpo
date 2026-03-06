"""bayesflow_hpo public API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bayesflow-hpo")
except PackageNotFoundError:
    __version__ = "0.1.0"

from bayesflow_hpo.api import optimize
from bayesflow_hpo.builders import (
    WorkflowBuildConfig,
    build_inference_network,
    build_summary_network,
    build_workflow,
)
from bayesflow_hpo.objectives import (
    denormalize_param_count,
    extract_objective_values,
    get_param_count,
    normalize_param_count,
)
from bayesflow_hpo.optimization import (
    GenericObjective,
    MovingAverageEarlyStopping,
    ObjectiveConfig,
    OptunaReportCallback,
    cleanup_trial,
    count_trained_trials,
    create_study,
    estimate_peak_memory_mb,
    exceeds_memory_budget,
    optimize_until,
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
    summarize_study,
    trials_to_dataframe,
)
from bayesflow_hpo.search_spaces import (
    CategoricalDimension,
    CompositeSearchSpace,
    ConsistencyModelSpace,
    CouplingFlowSpace,
    DeepSetSpace,
    DiffusionModelSpace,
    FloatDimension,
    FlowMatchingSpace,
    FusionTransformerSpace,
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
    DEFAULT_METRICS,
    ValidationDataset,
    ValidationResult,
    generate_validation_dataset,
    list_metrics,
    load_validation_dataset,
    make_condition_grid,
    make_coverage_metric,
    make_validation_dataset,
    register_metric,
    run_validation_pipeline,
    save_validation_dataset,
    validate_once,
)

__all__ = [
    # Version
    "__version__",
    # High-level API
    "optimize",
    # Builders
    "WorkflowBuildConfig",
    "build_inference_network",
    "build_summary_network",
    "build_workflow",
    # Objectives
    "denormalize_param_count",
    "extract_objective_values",
    "get_param_count",
    "normalize_param_count",
    # Optimization
    "GenericObjective",
    "MovingAverageEarlyStopping",
    "ObjectiveConfig",
    "OptunaReportCallback",
    "cleanup_trial",
    "count_trained_trials",
    "create_study",
    "estimate_peak_memory_mb",
    "exceeds_memory_budget",
    "optimize_until",
    "warm_start_study",
    # Registration
    "list_registered_network_spaces",
    "register_custom_inference_network",
    "register_custom_summary_network",
    # Results
    "get_pareto_trials",
    "get_workflow_metadata",
    "load_workflow_with_metadata",
    "plot_param_importance",
    "plot_pareto_front",
    "save_workflow_with_metadata",
    "summarize_study",
    "trials_to_dataframe",
    # Search spaces
    "CategoricalDimension",
    "CompositeSearchSpace",
    "ConsistencyModelSpace",
    "CouplingFlowSpace",
    "DeepSetSpace",
    "DiffusionModelSpace",
    "FloatDimension",
    "FlowMatchingSpace",
    "FusionTransformerSpace",
    "IntDimension",
    "NetworkSelectionSpace",
    "SetTransformerSpace",
    "StableConsistencyModelSpace",
    "SummarySelectionSpace",
    "TimeSeriesNetworkSpace",
    "TimeSeriesTransformerSpace",
    "TrainingSpace",
    "list_inference_spaces",
    "list_summary_spaces",
    "register_inference_space",
    "register_summary_space",
    # Utils
    "loguniform_float",
    "loguniform_int",
    # Validation
    "DEFAULT_METRICS",
    "ValidationDataset",
    "ValidationResult",
    "generate_validation_dataset",
    "list_metrics",
    "load_validation_dataset",
    "make_condition_grid",
    "make_coverage_metric",
    "make_validation_dataset",
    "register_metric",
    "run_validation_pipeline",
    "save_validation_dataset",
    "validate_once",
]
