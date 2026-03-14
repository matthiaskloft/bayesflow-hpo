"""High-level user API for bayesflow_hpo."""

from __future__ import annotations

import logging
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
from bayesflow_hpo.pipeline import check_pipeline
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.types import BuildApproximatorFn, TrainFn, ValidateFn
from bayesflow_hpo.validation.data import (
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
    search_space: CompositeSearchSpace,
    # Custom approximator hooks (all optional)
    build_approximator_fn: BuildApproximatorFn | None = None,
    train_fn: TrainFn | None = None,
    validate_fn: ValidateFn | None = None,
    # Validation data
    validation_conditions: dict[str, list[Any]] | None = None,
    sims_per_condition: int = 200,
    n_posterior_samples: int = 500,
    # Objectives
    objective_metrics: list[str] | None = None,
    objective_mode: str = "pareto",
    cost_metric: str = "inference_time",
    # Training
    epochs: int = 200,
    batches_per_epoch: int = 50,
    early_stopping_patience: int = 5,
    early_stopping_window: int = 7,
    # Logging
    report_frequency: int = 10,
    # Budget
    max_param_count: int = 1_000_000,
    max_memory_mb: float | None = None,
    # Study
    n_trials: int = 50,
    max_total_trials: int | None = None,
    study_name: str = "bayesflow_hpo",
    storage: str | None = DEFAULT_STORAGE,
    resume: bool = False,
    # Optional
    directions: list[str] | None = None,
    warm_start_from: Any | None = None,
    warm_start_top_k: int = 25,
    checkpoint_pool: CheckpointPool | None = None,
    show_progress_bar: bool = True,
) -> optuna.Study:
    """Run HPO with a high-level convenience API.

    This is the main entry point for hyperparameter optimization.  It
    creates an Optuna study, generates validation data, runs
    ``check_pipeline()`` to catch interface errors, and then runs
    ``n_trials`` fully-trained trials.

    Three optional hooks let callers replace the build, train, and
    validate steps while reusing the full trial lifecycle (budget
    rejection, early stopping, checkpoint management, cost scoring):

    - ``build_approximator_fn``: custom approximator construction
    - ``train_fn``: custom training loop
    - ``validate_fn``: custom validation metrics

    Parameters
    ----------
    simulator
        BayesFlow simulator used for online training and for
        generating validation data.
    adapter
        BayesFlow adapter for data preprocessing.
    search_space
        Search space defining the tunable dimensions.
    build_approximator_fn
        Optional custom build function ``(hparams) -> Approximator``.
        Must return an **uncompiled** approximator.  When ``None``
        (default), uses ``build_continuous_approximator()``.
    train_fn
        Optional custom training function
        ``(approximator, simulator, hparams, callbacks) -> None``.
        When ``None`` (default), uses ``default_train_fn()``.
    validate_fn
        Optional custom validation function
        ``(approximator, validation_data, n_posterior_samples) ->
        dict[str, float]``.  When ``None`` (default), uses
        ``default_validate_fn()``.
    validation_conditions
        Condition grid specification
        (e.g. ``{"N": [50, 100, 200]}``).  Used to build a
        ``ValidationDataset`` via :func:`generate_validation_dataset`.
        When ``None`` and no conditions are inferred from the adapter,
        a single unconditional batch is generated.
    sims_per_condition
        Simulations per condition grid point (default 200).
    n_posterior_samples
        Posterior draws for validation (default 500).
    objective_metrics
        List of metric keys to optimize simultaneously.  Default
        ``["calibration_error", "nrmse"]``.
    objective_mode
        ``"pareto"`` (default) — each metric is its own objective;
        study has ``len(objective_metrics) + 1`` directions.
        ``"mean"`` — arithmetic mean of the listed metrics forms one
        scalar; study has 2 directions (mean + cost).
    cost_metric
        Which cost objective to use as the last Optuna direction.
        ``"inference_time"`` (default) or ``"param_count"``.
    epochs
        Maximum training epochs per trial (default 200).
    batches_per_epoch
        Number of online simulation batches per epoch (default 50).
    early_stopping_patience
        Moving-average patience epochs for early stopping (default 5).
    early_stopping_window
        Moving-average window size for early stopping (default 7).
    report_frequency
        How often (in epochs) the ``OptunaReportCallback`` stores
        ``epoch_{N}_loss`` user attributes on each trial.  Higher
        values reduce SQLite bloat at the cost of coarser loss
        curves.  Default 10.
    max_param_count
        Trials with actual parameter count above this value are
        rejected before training (default 1 000 000).
    max_memory_mb
        Optional peak-memory budget in MB.  Disabled by default.
    n_trials
        Number of *trained* trials to collect (default 50).
    max_total_trials
        Hard cap on total trials including budget-rejected ones.
        Defaults to ``3 * n_trials``.
    study_name
        Optuna study name (default ``"bayesflow_hpo"``).
    storage
        Optuna storage URL (default ``"sqlite:///bayesflow_hpo.db"``).
        Pass ``None`` for in-memory.
    resume
        If ``True``, continue a previously persisted study.  If
        ``False`` (default), any existing study is deleted first.
    directions
        Optimization directions.  Default ``None`` (auto-derived as
        ``["minimize"] * n_objectives``).
    warm_start_from
        Optional source ``optuna.Study`` to seed initial trials from.
    warm_start_top_k
        Number of best trials to copy from the source study
        (default 25).
    checkpoint_pool
        Optional :class:`CheckpointPool` for persisting the best
        trial weights.
    show_progress_bar
        Whether to show Optuna's progress bar (default ``True``).

    Returns
    -------
    optuna.Study
        The optimized Optuna study.
    """
    if objective_metrics is None:
        objective_metrics = ["calibration_error", "nrmse"]

    # --- Derive keys from adapter ---
    adapter_keys = infer_keys_from_adapter(adapter)
    param_keys = adapter_keys["param_keys"]
    data_keys = adapter_keys["data_keys"]

    if param_keys is None:
        raise TypeError(
            "Could not infer param_keys: the adapter has no "
            "Rename/Concatenate targeting 'inference_variables'."
        )
    if data_keys is None:
        raise TypeError(
            "Could not infer data_keys: the adapter has no "
            "Rename/Concatenate targeting 'summary_variables'."
        )

    # --- Always build validation data internally ---
    validation_data = generate_validation_dataset(
        simulator=simulator,
        param_keys=param_keys,
        data_keys=data_keys,
        condition_grid=validation_conditions,
        sims_per_condition=sims_per_condition,
    )

    # --- Pre-flight validation ---
    check_pipeline(
        simulator=simulator,
        adapter=adapter,
        search_space=search_space,
        build_approximator_fn=build_approximator_fn,
        train_fn=train_fn,
        validate_fn=validate_fn,
        objective_metrics=objective_metrics,
        validation_conditions=validation_conditions,
    )

    # --- Build objective ---
    objective = GenericObjective(
        ObjectiveConfig(
            simulator=simulator,
            adapter=adapter,
            search_space=search_space,
            validation_data=validation_data,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            early_stopping_patience=early_stopping_patience,
            early_stopping_window=early_stopping_window,
            max_param_count=max_param_count,
            max_memory_mb=max_memory_mb,
            n_posterior_samples=n_posterior_samples,
            objective_metrics=objective_metrics,
            objective_mode=objective_mode,
            cost_metric=cost_metric,
            report_frequency=report_frequency,
            build_approximator_fn=build_approximator_fn,
            train_fn=train_fn,
            validate_fn=validate_fn,
            checkpoint_pool=checkpoint_pool,
        )
    )

    # --- Derive directions ---
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

    cost_label = cost_metric
    if objective_mode == "pareto":
        metric_names = list(objective_metrics) + [cost_label]
    else:
        metric_names = ["mean(" + "+".join(objective_metrics) + ")", cost_label]

    if not resume and storage is not None:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass
        except Exception:
            logger.warning(
                "Could not delete existing study %r from storage %s",
                study_name, storage, exc_info=True,
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
