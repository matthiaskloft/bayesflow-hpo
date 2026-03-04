"""High-level user API for bayesflow_hpo."""

from __future__ import annotations

from typing import Any

import bayesflow as bf

from bayesflow_hpo.optimization.objective import GenericObjective, ObjectiveConfig
from bayesflow_hpo.optimization.study import create_study
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
    epochs: int = 200,
    batches_per_epoch: int = 50,
    max_param_count: int = 2_000_000,
    max_memory_mb: float | None = None,
    storage: str | None = None,
    study_name: str = "bayesflow_hpo",
    directions: list[str] | None = None,
    warm_start_from: Any | None = None,
    warm_start_top_k: int = 20,
    warm_start_metric_index: int = 0,
    show_progress_bar: bool = True,
) -> Any:
    """Run HPO with a high-level convenience API."""
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
        )
    )

    study = create_study(
        study_name=study_name,
        directions=directions,
        storage=storage,
        load_if_exists=True,
        warm_start_from=warm_start_from,
        warm_start_top_k=warm_start_top_k,
        warm_start_metric_index=warm_start_metric_index,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        gc_after_trial=True,
    )
    return study
