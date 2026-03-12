"""High-level user API for bayesflow_hpo."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import bayesflow as bf
import optuna

from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool
from bayesflow_hpo.optimization.objective import GenericObjective, ObjectiveConfig
from bayesflow_hpo.optimization.study import (
    DEFAULT_STORAGE,
    create_study,
    optimize_until,
)
from bayesflow_hpo.search_spaces.composite import (
    CompositeSearchSpace,
    NetworkSelectionSpace,
    SummarySelectionSpace,
)
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.inference.flow_matching import FlowMatchingSpace
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace
from bayesflow_hpo.search_spaces.summary.set_transformer import SetTransformerSpace
from bayesflow_hpo.search_spaces.training import TrainingSpace
from bayesflow_hpo.validation.data import (
    ValidationDataset,
    generate_validation_dataset,
)

logger = logging.getLogger(__name__)

# BayesFlow canonical key names used by Adapter.rename / .concatenate
_CANONICAL_PARAM = "inference_variables"
_CANONICAL_DATA = "summary_variables"
_CANONICAL_COND = "inference_conditions"


def infer_keys_from_adapter(
    adapter: bf.adapters.Adapter,
) -> dict[str, list[str] | None]:
    """Infer ``param_keys``, ``data_keys``, and ``inference_conditions`` from *adapter*.

    The function walks the adapter's transform list and looks for
    :class:`~bayesflow.adapters.transforms.Rename` or
    :class:`~bayesflow.adapters.transforms.Concatenate` transforms whose
    target is one of BayesFlow's canonical keys (``inference_variables``,
    ``summary_variables``, ``inference_conditions``).

    Returns
    -------
    dict
        ``{"param_keys": [...] | None, "data_keys": [...] | None,
        "inference_conditions": [...] | None}``.  A value is ``None``
        when no matching transform was found for that role.
    """
    result: dict[str, list[str] | None] = {
        "param_keys": None,
        "data_keys": None,
        "inference_conditions": None,
    }

    canonical_to_result = {
        _CANONICAL_PARAM: "param_keys",
        _CANONICAL_DATA: "data_keys",
        _CANONICAL_COND: "inference_conditions",
    }

    transforms = getattr(adapter, "transforms", None)
    if transforms is None:
        return result

    for transform in transforms:
        # Rename('original' -> 'inference_variables')
        to_key = getattr(transform, "to_key", None)
        if to_key in canonical_to_result:
            from_key = getattr(transform, "from_key", None)
            if from_key is not None:
                result_key = canonical_to_result[to_key]
                existing = result[result_key]
                if existing is None:
                    result[result_key] = [from_key]
                else:
                    existing.append(from_key)
            continue

        # Concatenate(['a', 'b'] -> 'inference_variables')
        into = getattr(transform, "into", None)
        if into in canonical_to_result:
            keys = getattr(transform, "keys", None)
            if keys is not None:
                result_key = canonical_to_result[into]
                existing = result[result_key]
                if existing is None:
                    result[result_key] = list(keys)
                else:
                    existing.extend(keys)

    return result


def optimize(
    simulator: bf.simulators.Simulator,
    adapter: bf.adapters.Adapter,
    param_keys: list[str] | None = None,
    data_keys: list[str] | None = None,
    validation_data: ValidationDataset | None = None,
    validation_conditions: dict[str, list[Any]] | None = None,
    sims_per_condition: int = 200,
    search_space: CompositeSearchSpace | None = None,
    inference_conditions: list[str] | None = None,
    n_trials: int = 50,
    max_total_trials: int | None = None,
    epochs: int = 200,
    batches_per_epoch: int = 50,
    early_stopping_patience: int = 5,
    early_stopping_window: int = 7,
    max_param_count: int = 1_000_000,
    max_memory_mb: float | None = None,
    metrics: list[str] | None = None,
    objective_metric: str = "calibration_error",
    objective_metrics: list[str] | None = None,
    objective_mode: str = "mean",
    cost_metric: str = "inference_time",
    train_fn: Callable[[bf.BasicWorkflow, dict, list], None] | None = None,
    storage: str | None = DEFAULT_STORAGE,
    study_name: str = "bayesflow_hpo",
    directions: list[str] | None = None,
    resume: bool = False,
    warm_start_from: Any | None = None,
    warm_start_top_k: int = 25,
    checkpoint_pool: CheckpointPool | None = None,
    show_progress_bar: bool = True,
) -> Any:
    """Run HPO with a high-level convenience API.

    This is the main entry point for hyperparameter optimization.  It
    creates an Optuna study, builds a search space (if not provided),
    generates validation data (if ``validation_conditions`` are given),
    and runs ``n_trials`` fully-trained trials.

    Parameters
    ----------
    simulator
        BayesFlow simulator used for online training and (optionally)
        for generating validation data.
    adapter
        BayesFlow adapter for data preprocessing.
    param_keys
        Names of the parameters to infer.  Optional when the adapter
        uses canonical key names (``inference_variables``) or when
        ``validation_data`` is provided.  Must match
        ``validation_data.param_keys`` if both are given.
    data_keys
        Names of the data/observable variables.  Optional when the
        adapter uses canonical key names (``summary_variables``) or
        when ``validation_data`` is provided.  Must match
        ``validation_data.data_keys`` if both are given.
    validation_data
        Pre-generated :class:`ValidationDataset`.  When ``None`` and
        ``validation_conditions`` is provided, data is generated
        automatically.
    validation_conditions
        Condition grid specification
        (e.g. ``{"N": [50, 100, 200]}``).  Used to build a
        ``ValidationDataset`` via :func:`generate_validation_dataset`.
    sims_per_condition
        Simulations per condition grid point (default 200).
    search_space
        Search space defining the tunable dimensions.  Default is
        ``NetworkSelectionSpace`` over CouplingFlow + FlowMatching
        for inference, ``SummarySelectionSpace`` over DeepSet +
        SetTransformer for summary, plus ``TrainingSpace``.
    inference_conditions
        Names of conditioning variables passed to the workflow.
    n_trials
        Number of *trained* trials to collect (default 50).
        Budget-rejected trials (those exceeding ``max_param_count``
        or ``max_memory_mb``) are not counted toward this number.
    max_total_trials
        Hard cap on total trials including budget-rejected ones.
        Defaults to ``3 * n_trials``.
    epochs
        Maximum training epochs per trial (default 200).  A cosine-
        annealed learning rate schedule decays from ``initial_lr`` to
        near-zero over this many epochs.  Early stopping typically
        halts training before reaching the limit.
    batches_per_epoch
        Number of online simulation batches per epoch (default 50).
    early_stopping_patience
        Moving-average patience epochs for early stopping (default 5).
    early_stopping_window
        Moving-average window size for early stopping (default 7).
    max_param_count
        Trials with actual parameter count above this value are
        rejected before training (default 1 000 000).
    max_memory_mb
        Optional peak-memory budget in MB.  Disabled by default.
    metrics
        List of metric names for post-training validation (resolved
        via the metric registry).  Defaults to ``DEFAULT_METRICS``
        (calibration_error, nrmse, correlation, coverage, rmse,
        contraction).
    objective_metric
        Key in the validation summary used as the first HPO objective
        (default ``"calibration_error"``).  Ignored when
        ``objective_metrics`` is set.
    objective_metrics
        List of metric keys to optimize simultaneously.  When set,
        overrides ``objective_metric``.  The number of directions is
        computed automatically based on ``objective_mode``.
        Example: ``["calibration_error", "nrmse"]``.
    objective_mode
        ``"mean"`` (default) — arithmetic mean of the listed metrics
        forms one scalar; study has 2 directions (mean + cost).
        ``"pareto"`` — each metric is its own objective; study has
        ``len(objective_metrics) + 1`` directions.
    cost_metric
        Which cost objective to use as the last Optuna direction.
        ``"inference_time"`` (default) — inference-to-simulation time
        ratio.  ``"param_count"`` — normalized parameter count.
    train_fn
        Optional custom training function
        ``(workflow, params, callbacks) -> None``.  By default uses
        ``workflow.fit_online(...)``.
    storage
        Optuna storage URL (default ``"sqlite:///bayesflow_hpo.db"``
        for automatic persistence).  Pass ``None`` for in-memory.
    study_name
        Optuna study name (default ``"bayesflow_hpo"``).
    directions
        Optimization directions.  Default ``None`` (auto-derived as
        ``["minimize"] * n_objectives``).  Pass explicitly only to
        override auto-derivation.
    resume
        If ``True``, continue a previously persisted study with the
        same ``study_name`` and ``storage``.  If ``False`` (default),
        any existing study with that name is deleted first so the
        optimization starts from scratch.
    warm_start_from
        Optional source ``optuna.Study`` to seed initial trials from.
    warm_start_top_k
        Number of best trials to copy from the source study
        (default 25).
    checkpoint_pool
        Optional :class:`CheckpointPool` for persisting the best
        trial weights.  Default creates a pool of size 5 under
        ``checkpoints/``.
    show_progress_bar
        Whether to show Optuna's progress bar (default ``True``).

    Returns
    -------
    optuna.Study
        The optimized Optuna study.
    """
    # --- Infer keys from adapter when not explicitly provided ---
    adapter_keys = infer_keys_from_adapter(adapter)

    if param_keys is None and adapter_keys["param_keys"] is not None:
        param_keys = adapter_keys["param_keys"]
        logger.info("Inferred param_keys from adapter: %s", param_keys)

    if data_keys is None and adapter_keys["data_keys"] is not None:
        data_keys = adapter_keys["data_keys"]
        logger.info("Inferred data_keys from adapter: %s", data_keys)

    inferred_cond = adapter_keys["inference_conditions"]
    if inference_conditions is None and inferred_cond is not None:
        inference_conditions = inferred_cond
        logger.info(
            "Inferred inference_conditions from adapter: %s",
            inference_conditions,
        )

    # --- Validate / fall back to validation_data ---
    if validation_data is not None:
        if param_keys is None:
            param_keys = validation_data.param_keys
        elif param_keys != validation_data.param_keys:
            raise ValueError(
                f"param_keys mismatch: got {param_keys}, "
                f"dataset has {validation_data.param_keys}"
            )
        if data_keys is None:
            data_keys = validation_data.data_keys
        elif data_keys != validation_data.data_keys:
            raise ValueError(
                f"data_keys mismatch: got {data_keys}, "
                f"dataset has {validation_data.data_keys}"
            )
    else:
        if param_keys is None:
            raise TypeError(
                "param_keys is required when neither validation_data "
                "is provided nor the adapter uses canonical key names"
            )
        if data_keys is None:
            raise TypeError(
                "data_keys is required when neither validation_data "
                "is provided nor the adapter uses canonical key names"
            )

    if search_space is None:
        search_space = CompositeSearchSpace(
            inference_space=NetworkSelectionSpace(
                candidates={
                    "coupling_flow": CouplingFlowSpace(),
                    "flow_matching": FlowMatchingSpace(),
                }
            ),
            summary_space=SummarySelectionSpace(
                candidates={
                    "deep_set": DeepSetSpace(),
                    "set_transformer": SetTransformerSpace(),
                }
            ),
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

    objective = GenericObjective(
        ObjectiveConfig(
            simulator=simulator,
            adapter=adapter,
            search_space=search_space,
            inference_conditions=inference_conditions,
            validation_data=validation_data,
            param_keys=param_keys,
            data_keys=data_keys,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            early_stopping_patience=early_stopping_patience,
            early_stopping_window=early_stopping_window,
            max_param_count=max_param_count,
            max_memory_mb=max_memory_mb,
            metrics=metrics,
            objective_metric=objective_metric,
            objective_metrics=objective_metrics,
            objective_mode=objective_mode,
            cost_metric=cost_metric,
            train_fn=train_fn,
            checkpoint_pool=checkpoint_pool,
        )
    )

    # Derive directions and metric_names from the objective shape.
    n_obj = objective.n_objectives
    if directions is None:
        directions = ["minimize"] * n_obj
    elif len(directions) != n_obj:
        raise ValueError(
            f"directions has {len(directions)} entries but the "
            f"objective returns {n_obj} values "
            f"(objective_mode={objective_mode!r}, "
            f"objective_metrics={objective_metrics!r}). "
            f"Either pass directions=None to auto-derive, or "
            f"provide exactly {n_obj} directions."
        )

    cost_label = cost_metric  # "inference_time" or "param_count"
    if objective_metrics and objective_mode == "pareto":
        metric_names = list(objective_metrics) + [cost_label]
    elif objective_metrics:
        metric_names = ["mean(" + "+".join(objective_metrics) + ")", cost_label]
    else:
        metric_names = [objective_metric, cost_label]

    if not resume and storage is not None:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass  # no existing study to delete
        except Exception:
            logger.warning(
                "Could not delete existing study %r from storage %s",
                study_name,
                storage,
                exc_info=True,
            )

    study = create_study(
        study_name=study_name,
        directions=directions,
        metric_names=metric_names,
        storage=storage,
        load_if_exists=resume or storage is None,
        warm_start_from=warm_start_from,
        warm_start_top_k=warm_start_top_k,
    )
    optimize_until(
        study,
        objective,
        n_trained=n_trials,
        max_total_trials=max_total_trials,
        show_progress_bar=show_progress_bar,
    )
    return study
