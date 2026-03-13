"""bayesflow_hpo — Generic hyperparameter optimization for BayesFlow 2.x.

This package wraps Optuna multi-objective search with BayesFlow-aware
search spaces, builders, and validation.  The main entry point is
:func:`optimize`, which runs a complete HPO loop:

1. Define a search space over inference/summary networks
2. Generate a fixed validation dataset (reused across all trials)
3. For each trial: sample → build → compile → train → validate
4. Return the Optuna study with Pareto-optimal results

Quick start::

    import bayesflow_hpo as hpo

    study = hpo.optimize(
        simulator=my_simulator,
        adapter=my_adapter,
        search_space=hpo.CompositeSearchSpace(
            inference_space=hpo.FlowMatchingSpace(),
            summary_space=hpo.DeepSetSpace(),
            training_space=hpo.TrainingSpace(),
        ),
        validation_conditions={"N": [50, 100, 200]},
        n_trials=50,
    )
    print(hpo.summarize_study(study))
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bayesflow-hpo")
except PackageNotFoundError:
    __version__ = "0.1.0"

from bayesflow_hpo.api import infer_keys_from_adapter, optimize
from bayesflow_hpo.builders import build_continuous_approximator
from bayesflow_hpo.objectives import (
    compute_inference_time_ratio,
    denormalize_param_count,
    extract_multi_objective_values,
    extract_objective_values,
    get_param_count,
    normalize_param_count,
)
from bayesflow_hpo.optimization import (
    DEFAULT_STORAGE,
    CheckpointPool,
    GenericObjective,
    MovingAverageEarlyStopping,
    ObjectiveConfig,
    OptunaReportCallback,
    PeriodicValidationCallback,
    cleanup_trial,
    count_trained_trials,
    create_study,
    estimate_peak_memory_mb,
    exceeds_memory_budget,
    optimize_until,
    warm_start_study,
)
from bayesflow_hpo.optimization.objective import default_train_fn, default_validate_fn
from bayesflow_hpo.pipeline import PipelineError, check_pipeline
from bayesflow_hpo.registration import (
    list_registered_network_spaces,
    register_custom_inference_network,
    register_custom_summary_network,
)
from bayesflow_hpo.results import (
    get_pareto_trials,
    get_workflow_metadata,
    load_workflow_with_metadata,
    plot_metric_panels,
    plot_metric_scatter,
    plot_optimization_history,
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
from bayesflow_hpo.types import BuildApproximatorFn, TrainFn, ValidateFn
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
    "check_pipeline",
    "infer_keys_from_adapter",
    "optimize",
    # Type aliases
    "BuildApproximatorFn",
    "TrainFn",
    "ValidateFn",
    # Pipeline
    "PipelineError",
    # Builders
    "build_continuous_approximator",
    # Default wrappers
    "default_train_fn",
    "default_validate_fn",
    # Objectives
    "compute_inference_time_ratio",
    "denormalize_param_count",
    "extract_multi_objective_values",
    "extract_objective_values",
    "get_param_count",
    "normalize_param_count",
    # Optimization
    "CheckpointPool",
    "DEFAULT_STORAGE",
    "GenericObjective",
    "MovingAverageEarlyStopping",
    "ObjectiveConfig",
    "OptunaReportCallback",
    "PeriodicValidationCallback",
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
    "plot_metric_panels",
    "plot_metric_scatter",
    "plot_optimization_history",
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
