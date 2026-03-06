"""Optuna integration for bayesflow_hpo."""

from bayesflow_hpo.optimization.callbacks import (
    MovingAverageEarlyStopping,
    OptunaReportCallback,
)
from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool
from bayesflow_hpo.optimization.cleanup import cleanup_trial
from bayesflow_hpo.optimization.constraints import (
    estimate_param_count,
    estimate_peak_memory_mb,
    exceeds_memory_budget,
)
from bayesflow_hpo.optimization.objective import GenericObjective, ObjectiveConfig
from bayesflow_hpo.optimization.sampling import sample_hyperparameters
from bayesflow_hpo.optimization.study import (
    DEFAULT_STORAGE,
    count_trained_trials,
    create_study,
    optimize_until,
    resume_study,
    warm_start_study,
)
from bayesflow_hpo.optimization.validation_callback import PeriodicValidationCallback

__all__ = [
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
    "estimate_param_count",
    "estimate_peak_memory_mb",
    "exceeds_memory_budget",
    "optimize_until",
    "resume_study",
    "sample_hyperparameters",
    "warm_start_study",
]
