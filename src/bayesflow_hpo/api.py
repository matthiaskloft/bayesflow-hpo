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
    search_space: CompositeSearchSpace,
    # Custom approximator hooks (all optional)
    build_approximator_fn: BuildApproximatorFn | None = None,
    train_fn: TrainFn | None = None,
    validate_fn: ValidateFn | None = None,
    # Validation data
    validation_simulator: bf.simulators.Simulator | None = None,
    validation_conditions: dict[str, list[Any]] | None = None,
    sims_per_condition: int = 200,
    n_posterior_samples: int = 500,
    # Objectives
    objective_metrics: list[str] | None = None,
    objective_mode: str = "pareto",
    cost_metric: str = "inference_time",
    # Pruning
    pruning_strategy: str | tuple[str, str] = "dominance",
    # Training
    epochs: int = 200,
    num_batches: int = 50,
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
        BayesFlow simulator used for online training and (unless
        *validation_simulator* is given) for generating validation data.
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

        The returned dict must contain all keys in ``objective_metrics``
        (default ``["calibration_error", "nrmse"]``).  Missing or
        non-finite values are replaced with a penalty value and a
        warning is logged.  Extra keys are silently ignored.

        **Timing caveat:** the wall-clock time of this function is
        recorded as the trial's inference time.  Unlike the default
        path, which isolates pure inference timing, a custom
        ``validate_fn`` lumps inference and metric computation together.

        **Intermediate pruning:** this function is also called during
        training at the configured interval with a reduced
        ``n_posterior_samples`` for median-based multi-objective
        pruning.
    validation_simulator
        Optional simulator used *only* for generating the validation
        dataset.  When ``None`` (default), ``simulator`` is used.
        Use this to pin validation to a specific condition — e.g. a
        simulator cloned with fixed sizes for edge-case testing —
        while keeping the training simulator's full range.
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

        Built-in metrics (pass any of these as strings):

        ================= ======================================
        Name              Description
        ================= ======================================
        calibration_error Expected Calibration Error (ECE)
        nrmse             Range-normalized RMSE
        rmse              RMSE of posterior means
        correlation       Pearson correlation (means vs true)
        contraction       Posterior contraction (1=learned)
        z_score           Posterior z-score (bias+calibration)
        log_gamma         Log-gamma calibration diagnostic
        coverage          Two-sided SBC rank coverage
        coverage_left     Left-sided coverage (efficiency)
        coverage_right    Right-sided coverage (futility)
        sbc_ks            SBC KS statistic (minimize → 0)
        sbc_chi2          SBC chi-squared stat (min → 0)
        bias              Mean signed error
        mae               Mean Absolute Error
        ================= ======================================

        Some metrics have aliases (e.g. ``"corr"`` for
        ``"correlation"``).  Call :func:`describe_metrics` for the
        full listing including aliases.

        Use :func:`~bayesflow_hpo.list_metrics` for just the names,
        or :func:`~bayesflow_hpo.register_metric` to add custom ones.
    objective_mode
        ``"pareto"`` (default) — each metric is its own objective;
        study has ``len(objective_metrics) + 1`` directions.
        ``"mean"`` — arithmetic mean of the listed metrics forms one
        scalar; study has 2 directions (mean + cost).
    cost_metric
        Which cost objective to use as the last Optuna direction.
        ``"inference_time"`` (default) or ``"param_count"``.
    pruning_strategy
        Multi-objective pruning strategy.  One of ``"dominance"``
        (default), ``"mo-sha"``, ``"primary"``, or ``"none"``.
        For ``"primary"``, pass a tuple ``("primary", metric_name)``
        to specify which metric to prune on (defaults to
        ``objective_metrics[0]``).  ``"none"`` disables intermediate
        validation entirely.

        Strategies are backed by literature — see
        :mod:`~bayesflow_hpo.optimization.pruning_strategies` for
        details and references.  Schmucker et al. (2021) provide the
        MO-ASHA foundation for ``"dominance"`` and ``"mo-sha"``.

        Only applies to multi-objective studies; single-objective
        studies always use Optuna's built-in pruner.
    epochs
        Maximum training epochs per trial (default 200).
    num_batches
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

    Notes
    -----
    **Trial counting.**  Each trial ends in one of four states:

    - **trained** — completed training and validation successfully.
      Only these count toward ``n_trials``.
    - **rejected** — skipped before training because the sampled model
      exceeded the parameter or memory budget, or failed to build.
      These are cheap (no GPU time) and do not count toward
      ``n_trials`` or ``max_total_trials``.
    - **failed** — started training but crashed with an unrecoverable
      error.  Counts toward ``max_total_trials`` but not ``n_trials``.
    - **pruned** — stopped early by intermediate validation because
      the trial looked unpromising.  Counts toward
      ``max_total_trials`` but not ``n_trials``.

    Because rejected trials are free, the optimizer keeps sampling
    until it reaches ``n_trials`` trained trials.  Two safety caps
    prevent runaway loops:

    - ``max_total_trials`` (default ``3 * n_trials``) caps
      non-rejected trials (trained + failed + pruned).
    - A hard cap of ``5 * max_total_trials`` on *all* trials
      (including rejected) catches cases where the entire search
      space is infeasible.
    """
    if objective_metrics is None:
        objective_metrics = ["calibration_error", "nrmse"]

    # --- Early validation ---
    if report_frequency < 1:
        raise ValueError(
            f"report_frequency must be >= 1, got {report_frequency}."
        )

    # Step 1: Infer keys
    param_keys, data_keys = _infer_and_validate_keys(adapter)

    # Step 2: Validation data
    validation_data = _setup_validation_data(
        simulator=simulator,
        validation_simulator=validation_simulator,
        param_keys=param_keys,
        data_keys=data_keys,
        validation_conditions=validation_conditions,
        sims_per_condition=sims_per_condition,
    )

    # Step 3: Pre-flight check
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

    # Step 4: Build objective
    objective = _build_objective(
        simulator=simulator,
        adapter=adapter,
        search_space=search_space,
        validation_data=validation_data,
        epochs=epochs,
        num_batches=num_batches,
        early_stopping_patience=early_stopping_patience,
        early_stopping_window=early_stopping_window,
        max_param_count=max_param_count,
        max_memory_mb=max_memory_mb,
        n_posterior_samples=n_posterior_samples,
        objective_metrics=objective_metrics,
        objective_mode=objective_mode,
        cost_metric=cost_metric,
        report_frequency=report_frequency,
        pruning_strategy=pruning_strategy,
        build_approximator_fn=build_approximator_fn,
        train_fn=train_fn,
        validate_fn=validate_fn,
        checkpoint_pool=checkpoint_pool,
    )

    # Step 5: Derive directions
    directions, metric_names = _derive_directions(
        objective=objective,
        directions=directions,
        objective_metrics=objective_metrics,
        objective_mode=objective_mode,
        cost_metric=cost_metric,
    )

    # Step 6: Run study
    return _create_and_run_study(
        objective=objective,
        study_name=study_name,
        directions=directions,
        metric_names=metric_names,
        storage=storage,
        resume=resume,
        warm_start_from=warm_start_from,
        warm_start_top_k=warm_start_top_k,
        n_trials=n_trials,
        max_total_trials=max_total_trials,
        show_progress_bar=show_progress_bar,
    )


# ---------------------------------------------------------------------------
# Private helpers — extracted from optimize() for readability & testability
# ---------------------------------------------------------------------------


def _infer_and_validate_keys(
    adapter: bf.adapters.Adapter,
) -> tuple[list[str], list[str]]:
    """Infer and validate ``param_keys`` and ``data_keys`` from *adapter*.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(param_keys, data_keys)``.

    Raises
    ------
    TypeError
        When required keys cannot be inferred from the adapter.
    """
    adapter_keys = infer_keys_from_adapter(adapter)
    param_keys = adapter_keys["param_keys"]
    data_keys = adapter_keys["data_keys"]

    if param_keys is None:
        raise TypeError(
            "Could not infer param_keys: the adapter has no "
            "Rename/Concatenate targeting 'inference_variables'."
        )
    if data_keys is None:
        # Models without a summary network pass observations directly as
        # inference_conditions (e.g. the Two Moons benchmark).  Fall back
        # to those keys so optimize() works for condition-only adapters.
        data_keys = adapter_keys.get("inference_conditions")
    if data_keys is None:
        raise TypeError(
            "Could not infer data_keys: the adapter has no "
            "Rename/Concatenate targeting 'summary_variables' or "
            "'inference_conditions'."
        )

    return param_keys, data_keys


def _setup_validation_data(
    *,
    simulator: bf.simulators.Simulator,
    validation_simulator: bf.simulators.Simulator | None,
    param_keys: list[str],
    data_keys: list[str],
    validation_conditions: dict[str, list[Any]] | None,
    sims_per_condition: int,
) -> ValidationDataset:
    """Generate the fixed validation dataset.

    Uses *validation_simulator* when provided, otherwise falls back to
    *simulator*.
    """
    val_sim = validation_simulator if validation_simulator is not None else simulator
    return generate_validation_dataset(
        simulator=val_sim,
        param_keys=param_keys,
        data_keys=data_keys,
        condition_grid=validation_conditions,
        sims_per_condition=sims_per_condition,
    )


def _build_objective(
    *,
    simulator: bf.simulators.Simulator,
    adapter: bf.adapters.Adapter,
    search_space: CompositeSearchSpace,
    validation_data: ValidationDataset,
    epochs: int,
    num_batches: int,
    early_stopping_patience: int,
    early_stopping_window: int,
    max_param_count: int,
    max_memory_mb: float | None,
    n_posterior_samples: int,
    objective_metrics: list[str],
    objective_mode: str,
    cost_metric: str,
    report_frequency: int,
    pruning_strategy: str | tuple[str, str],
    build_approximator_fn: BuildApproximatorFn | None,
    train_fn: TrainFn | None,
    validate_fn: ValidateFn | None,
    checkpoint_pool: CheckpointPool | None,
) -> GenericObjective:
    """Construct the :class:`GenericObjective` from configuration."""
    return GenericObjective(
        ObjectiveConfig(
            simulator=simulator,
            adapter=adapter,
            search_space=search_space,
            validation_data=validation_data,
            epochs=epochs,
            num_batches=num_batches,
            early_stopping_patience=early_stopping_patience,
            early_stopping_window=early_stopping_window,
            max_param_count=max_param_count,
            max_memory_mb=max_memory_mb,
            n_posterior_samples=n_posterior_samples,
            objective_metrics=objective_metrics,
            objective_mode=objective_mode,
            cost_metric=cost_metric,
            report_frequency=report_frequency,
            pruning_strategy=pruning_strategy,
            build_approximator_fn=build_approximator_fn,
            train_fn=train_fn,
            validate_fn=validate_fn,
            checkpoint_pool=checkpoint_pool,
        )
    )


def _derive_directions(
    *,
    objective: GenericObjective,
    directions: list[str] | None,
    objective_metrics: list[str],
    objective_mode: str,
    cost_metric: str,
) -> tuple[list[str], list[str]]:
    """Validate or auto-derive optimization directions and metric names.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(directions, metric_names)``.
    """
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

    if objective_mode == "pareto":
        metric_names = list(objective_metrics) + [cost_metric]
    else:
        metric_names = ["mean(" + "+".join(objective_metrics) + ")", cost_metric]

    return directions, metric_names


def _create_and_run_study(
    *,
    objective: GenericObjective,
    study_name: str,
    directions: list[str],
    metric_names: list[str],
    storage: str | None,
    resume: bool,
    warm_start_from: Any | None,
    warm_start_top_k: int,
    n_trials: int,
    max_total_trials: int | None,
    show_progress_bar: bool,
) -> optuna.Study:
    """Create (or resume) an Optuna study and run optimization."""
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

    # Auto-detect n_startup_trials from sampler if not set explicitly.
    cfg = objective.config
    if cfg.pruning_n_startup_trials is None:
        resolved = getattr(study.sampler, "n_startup_trials", 10)
        cfg.pruning_n_startup_trials = resolved
        logger.debug(
            "Auto-detected pruning_n_startup_trials=%d from %s",
            resolved,
            type(study.sampler).__name__,
        )

    optimize_until(
        study,
        objective,
        n_trained=n_trials,
        max_total_trials=max_total_trials,
        show_progress_bar=show_progress_bar,
    )
    return study
