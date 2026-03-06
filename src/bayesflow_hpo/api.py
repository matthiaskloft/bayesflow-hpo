"""High-level user API for bayesflow_hpo."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import bayesflow as bf

from bayesflow_hpo.optimization.objective import GenericObjective, ObjectiveConfig
from bayesflow_hpo.optimization.study import create_study, optimize_until
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace
from bayesflow_hpo.search_spaces.training import TrainingSpace
from bayesflow_hpo.validation.data import ValidationDataset, generate_validation_dataset


def optimize(
    simulator: bf.simulators.Simulator,
    adapter: bf.adapters.Adapter,
    param_keys: list[str],
    data_keys: list[str],
    validation_data: ValidationDataset | None = None,
    validation_conditions: dict[str, list[Any]] | None = None,
    sims_per_condition: int = 200,
    search_space: CompositeSearchSpace | None = None,
    inference_conditions: list[str] | None = None,
    n_trials: int = 100,
    max_total_trials: int | None = None,
    epochs: int = 200,
    batches_per_epoch: int = 50,
    max_param_count: int = 2_000_000,
    max_memory_mb: float | None = None,
    metrics: list[str] | None = None,
    objective_metric: str = "mean_cal_error",
    train_fn: Callable[[bf.BasicWorkflow, dict, list], None] | None = None,
    storage: str | None = None,
    study_name: str = "bayesflow_hpo",
    directions: list[str] | None = None,
    warm_start_from: Any | None = None,
    warm_start_top_k: int = 20,
    warm_start_metric_index: int = 0,
    show_progress_bar: bool = True,
) -> Any:
    """Run HPO with a high-level convenience API.

    Parameters
    ----------
    n_trials
        Number of *trained* trials to collect. Budget-rejected trials
        (those exceeding ``max_param_count`` or ``max_memory_mb``) are
        not counted toward this number.
    max_total_trials
        Hard cap on total trials including budget-rejected ones.
        Defaults to ``5 * n_trials``.
    metrics
        List of metric names for validation (resolved via the metric
        registry). Defaults to ``DEFAULT_METRICS``.
    objective_metric
        Key in the validation summary used as the HPO objective.
    train_fn
        Optional custom training function ``(workflow, params, callbacks) -> None``.
        Defaults to ``workflow.fit_online(...)``.
    """
    if search_space is None:
        search_space = CompositeSearchSpace(
            inference_space=CouplingFlowSpace(),
            summary_space=DeepSetSpace(),
            training_space=TrainingSpace(),
        )

    if validation_data is None and validation_conditions is not None:
        validation_data = generate_validation_dataset(
            simulator=simulator,
            param_keys=param_keys,
            data_keys=data_keys,
            condition_grid=validation_conditions,
            sims_per_condition=sims_per_condition,
        )

    if directions is None:
        directions = ["minimize", "minimize"]

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=simulator,
            adapter=adapter,
            search_space=search_space,
            inference_conditions=inference_conditions,
            validation_data=validation_data,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            max_param_count=max_param_count,
            max_memory_mb=max_memory_mb,
            metrics=metrics,
            objective_metric=objective_metric,
            train_fn=train_fn,
        )
    )

    study = create_study(
        study_name=study_name,
        directions=directions,
        metric_names=[objective_metric, "param_count"] if len(directions) == 2 else None,
        storage=storage,
        load_if_exists=True,
        warm_start_from=warm_start_from,
        warm_start_top_k=warm_start_top_k,
        warm_start_metric_index=warm_start_metric_index,
    )
    optimize_until(
        study,
        objective,
        n_trained=n_trials,
        max_total_trials=max_total_trials,
        show_progress_bar=show_progress_bar,
    )
    return study
