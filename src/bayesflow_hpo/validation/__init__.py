"""Validation pipeline and data utilities."""

from bayesflow_hpo.validation.data import (
    ValidationDataset,
    generate_validation_dataset,
    load_validation_dataset,
    make_condition_grid,
    make_validation_dataset,
    save_validation_dataset,
)
from bayesflow_hpo.validation.dry_run import validate_once
from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn
from bayesflow_hpo.validation.metrics import (
    aggregate_condition_rows,
    compute_condition_metrics,
)
from bayesflow_hpo.validation.pipeline import run_validation_pipeline
from bayesflow_hpo.validation.registry import (
    DEFAULT_METRICS,
    MetricFn,
    get_metric,
    list_metrics,
    make_coverage_metric,
    register_metric,
    resolve_metrics,
)
from bayesflow_hpo.validation.result import ValidationResult
from bayesflow_hpo.validation.sbc_tests import (
    compute_sbc_c2st,
    compute_sbc_uniformity_tests,
)

__all__ = [
    "DEFAULT_METRICS",
    "MetricFn",
    "ValidationDataset",
    "ValidationResult",
    "aggregate_condition_rows",
    "compute_condition_metrics",
    "compute_sbc_c2st",
    "compute_sbc_uniformity_tests",
    "generate_validation_dataset",
    "get_metric",
    "list_metrics",
    "load_validation_dataset",
    "make_bayesflow_infer_fn",
    "make_condition_grid",
    "make_coverage_metric",
    "make_validation_dataset",
    "register_metric",
    "resolve_metrics",
    "run_validation_pipeline",
    "save_validation_dataset",
    "validate_once",
]
