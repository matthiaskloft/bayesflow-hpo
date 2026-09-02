"""High-level user API for bayesflow_hpo."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import bayesflow as bf
import optuna

from bayesflow_hpo.objectives import (
    ENCODING_CHANGED_AT_V2,
    OBJECTIVE_ENCODING_VERSION,
)
from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool
from bayesflow_hpo.optimization.constraints import (
    MetricConstraintSpec,
    _detect_gpu_memory_mb,
)
from bayesflow_hpo.optimization.objective import GenericObjective, ObjectiveConfig
from bayesflow_hpo.optimization.study import (
    DEFAULT_STORAGE,
    _resolve_n_startup_trials,
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
from bayesflow_hpo.validation.registry import (
    canonical_metric_name,
    validate_objective_metric_kinds,
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
        logger.debug("Adapter has no 'transforms' attribute; skipping key inference")
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
    training_mode: Literal["fixed_budget", "open_ended"] = "fixed_budget",
    epochs: int = 200,
    num_batches: int = 50,
    early_stopping_patience: int | None = None,
    early_stopping_window: int = 7,
    early_stopping_monitor: str = "objective_mean",
    lr_warmup_epochs: int | Sequence[int] | None = None,
    lr_warmup_steps: int | Sequence[int] | None = None,
    lr_warmup_fraction: float | Sequence[float] | None = None,
    # Logging
    report_frequency: int = 10,
    # Budget
    max_param_count: int = 1_000_000,
    max_memory_mb: float | None | str = None,
    metric_constraints_hard: list[MetricConstraintSpec] | None = None,
    metric_constraints_soft: list[MetricConstraintSpec] | None = None,
    memory_safety_margin: float = 0.2,
    # Study
    n_trials: int = 50,
    max_total_trials: int | None = None,
    study_name: str = "bayesflow_hpo",
    storage: str | None = DEFAULT_STORAGE,
    resume: bool = False,
    # Optional
    sampler: str | optuna.samplers.BaseSampler | None = None,
    directions: list[str] | None = None,
    warm_start_from: Any | None = None,
    warm_start_top_k: int = 25,
    qmc_startup_trials: int = 0,
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
        correlation       Diagnostic-only linear association
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
        ``objective_metrics[0]``). ``"none"`` disables pruning, but
        ``open_ended`` mode still runs intermediate validation for stopping.

        Strategies are backed by literature — see
        :mod:`~bayesflow_hpo.optimization.pruning_strategies` for
        details and references.  Schmucker et al. (2021) provide the
        MO-ASHA foundation for ``"dominance"`` and ``"mo-sha"``.

        Only applies to multi-objective studies; single-objective
        studies always use Optuna's built-in pruner.
    training_mode
        ``"fixed_budget"`` (default) couples cosine decay with training to
        the full trial budget. ``"open_ended"`` couples inverse-square-root
        decay with validation-objective early stopping.
    epochs
        Training epochs per trial. In ``open_ended`` mode this is a generous
        safety cap (default 200).
    num_batches
        Number of online simulation batches per epoch (default 50).
    early_stopping_patience
        Validation checks without improvement before stopping in
        ``open_ended`` mode. ``None`` selects 5 checks. Setting this in
        ``fixed_budget`` mode raises an error.
    early_stopping_window
        Moving-average window measured in validation checks for open-ended
        stopping and in epochs for the training-loss callback (default 7).
    early_stopping_monitor
        Validation stopping objective. ``"objective_mean"`` (default) averages
        ``objective_metrics`` after minimize-direction conversion; the separate
        cost objective is excluded. A metric name monitors only that metric.
    lr_warmup_epochs
        Linear-warmup length measured in each trial's actual epochs. ``None``
        selects 0 for ``fixed_budget`` and 1 for ``open_ended``. A sequence
        enables opt-in categorical HPO.
    lr_warmup_steps
        Exact optimizer-step warmup override. A sequence enables opt-in
        categorical HPO. Takes precedence over ``lr_warmup_epochs``.
    lr_warmup_fraction
        Fixed-budget warmup fraction, capped at 0.1. ``None`` selects 0.05.
        A sequence enables opt-in categorical HPO. Exact steps and epochs take
        precedence. Not valid in ``open_ended`` mode.
    report_frequency
        How often (in epochs) the ``OptunaReportCallback`` stores
        ``epoch_{N}_loss`` user attributes on each trial.  Higher
        values reduce SQLite bloat at the cost of coarser loss
        curves.  Default 10.
    max_param_count
        Trials with actual parameter count above this value are
        rejected before training (default 1 000 000).
    max_memory_mb
        Optional peak-memory budget in MB. Pass ``"auto"`` to detect
        free CUDA memory and apply ``memory_safety_margin``.
    metric_constraints_hard
        Optional hard metric thresholds as
        ``[(metric, threshold, "above"|"below"), ...]``.
        Violating trials are rejected after final validation.
    metric_constraints_soft
        Optional soft metric thresholds as
        ``[(metric, threshold, "above"|"below"), ...]``.
        Passed to Optuna's ``constraints_func`` for feasibility-guided
        sampling (when using sampler presets).
    memory_safety_margin
        Safety margin for ``max_memory_mb="auto"``. Default 0.2 (20%).
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
    sampler
        Optuna sampler.  Accepts a string preset, a ``BaseSampler``
        instance, or ``None`` (default ``"tpe"``).  See
        :func:`~bayesflow_hpo.create_study` for the full preset table.
    directions
        Optimization directions.  Default ``None`` (auto-derived as
        ``["minimize"] * n_objectives``).

        Every entry must be ``"minimize"``; anything else raises. The
        objective already converts each metric to minimize-is-better through
        :data:`bayesflow_hpo.objectives.METRIC_DIRECTIONS`, and the failure
        penalties are in minimize space too, so a ``"maximize"`` entry inverts
        that a second time -- the search would then prefer the *worst* model
        and the failure penalty would become its most attractive value, with
        nothing in the study output looking wrong. To optimize a
        higher-is-better metric, register its direction with
        :func:`bayesflow_hpo.objectives.register_metric_direction` and leave
        this ``None``.
    warm_start_from
        Optional source ``optuna.Study`` to seed initial trials from.
    warm_start_top_k
        Number of best trials to copy from the source study
        (default 25).
    qmc_startup_trials
        Number of initial trials to sample with a Sobol quasi-random
        sequence before the main sampler takes over.  Provides better
        space-filling coverage than random startup.  Only non-rejected
        completions count.  Default 0 (disabled).  See
        :func:`~bayesflow_hpo.create_study` for details.
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
    # Canonicalize at the PUBLIC boundary, before anything downstream sees the
    # names. Doing it inside ObjectiveConfig was too late: `check_pipeline`
    # already ran pre-flight against the caller's spelling, so
    # `objective_metrics=["cal_error"]` either failed pre-flight (the default
    # validator emits `calibration_error`) or passed pre-flight and then had
    # every trial penalized at runtime, depending on which validator was used.
    objective_metrics = [canonical_metric_name(m) for m in objective_metrics]
    validate_objective_metric_kinds(objective_metrics)

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

    # Resolve memory budget before building objective.
    resolved_max_memory_mb = _resolve_memory_budget(
        max_memory_mb=max_memory_mb,
        safety_margin=memory_safety_margin,
    )

    if metric_constraints_soft is not None and isinstance(
        sampler, optuna.samplers.BaseSampler
    ):
        logger.warning(
            "metric_constraints_soft was provided with a user-supplied sampler "
            "instance. Soft constraints are only auto-wired for sampler presets; "
            "skipping soft constraints."
        )

    # Step 4: Build objective
    objective = _build_objective(
        simulator=simulator,
        adapter=adapter,
        search_space=search_space,
        validation_data=validation_data,
        training_mode=training_mode,
        epochs=epochs,
        num_batches=num_batches,
        early_stopping_patience=early_stopping_patience,
        early_stopping_window=early_stopping_window,
        early_stopping_monitor=early_stopping_monitor,
        lr_warmup_epochs=lr_warmup_epochs,
        lr_warmup_steps=lr_warmup_steps,
        lr_warmup_fraction=lr_warmup_fraction,
        max_param_count=max_param_count,
        max_memory_mb=resolved_max_memory_mb,
        metric_constraints_hard=metric_constraints_hard,
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
        sampler=sampler,
        metric_constraints_soft=metric_constraints_soft,
        warm_start_from=warm_start_from,
        warm_start_top_k=warm_start_top_k,
        qmc_startup_trials=qmc_startup_trials,
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
    training_mode: Literal["fixed_budget", "open_ended"],
    epochs: int,
    num_batches: int,
    early_stopping_patience: int | None,
    early_stopping_window: int,
    early_stopping_monitor: str,
    lr_warmup_epochs: int | Sequence[int] | None,
    lr_warmup_steps: int | Sequence[int] | None,
    lr_warmup_fraction: float | Sequence[float] | None,
    max_param_count: int,
    max_memory_mb: float | None,
    metric_constraints_hard: list[MetricConstraintSpec] | None,
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
            training_mode=training_mode,
            epochs=epochs,
            num_batches=num_batches,
            early_stopping_patience=early_stopping_patience,
            early_stopping_window=early_stopping_window,
            early_stopping_monitor=early_stopping_monitor,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_warmup_steps=lr_warmup_steps,
            lr_warmup_fraction=lr_warmup_fraction,
            max_param_count=max_param_count,
            max_memory_mb=max_memory_mb,
            metric_constraints_hard=metric_constraints_hard,
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


def _guard_resumed_study(
    study: optuna.Study, objective_metrics: list[str]
) -> None:
    """Refuse to continue a study whose stored values use another encoding.

    Two hazards, both silent, and both specific to *resuming*.

    Optuna's ``create_study(load_if_exists=True)`` returns the stored study and
    keeps ITS directions, ignoring the ones requested here -- verified, not
    assumed. So a study created before objective values became minimize-space
    retains its ``maximize`` direction, never reaches
    :func:`_derive_directions` (the caller passes ``directions=None``), and
    then maximizes the newly negated values: the search would prefer the worst
    model available.

    Separately, trials stored under an older encoding hold raw values for a
    metric this version negates, so old and new trials in one study are on
    opposite scales. The sampler and the Pareto computation would compare them
    directly.

    Neither is repairable in place, so both refuse rather than warn.

    Parameters
    ----------
    study
        The study just created or loaded.
    objective_metrics
        Canonical objective metric names for this run.

    Raises
    ------
    ValueError
        If the study carries a non-minimize direction, or holds trials written
        under an older objective encoding while an encoding-sensitive metric
        is in use.
    """
    if any(d != optuna.study.StudyDirection.MINIMIZE for d in study.directions):
        raise ValueError(
            f"Study {study.study_name!r} was created with directions "
            f"{[d.name.lower() for d in study.directions]}, but this version "
            "returns minimize-is-better objective values for every metric. "
            "Optuna keeps a loaded study's own directions, so continuing "
            "would maximize already-negated values and select the worst "
            "model. Start a new study."
        )

    encoding = study.user_attrs.get("bayesflow_hpo_objective_encoding")
    if encoding == OBJECTIVE_ENCODING_VERSION:
        return

    has_trials = any(
        t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
    )
    # Only metrics whose stored numbers changed make old trials incomparable.
    # Read from an explicit record, not inferred from `higher_is_better`:
    # `contraction` is higher-is-better AND a usable objective, but its
    # conversion is identical to before this change, so inferring would refuse
    # a study that is perfectly comparable.
    sensitive = [m for m in objective_metrics if m in ENCODING_CHANGED_AT_V2]
    if has_trials and encoding is None and sensitive:
        raise ValueError(
            f"Study {study.study_name!r} holds trials written before objective "
            f"values were normalized to minimize-is-better, and this run "
            f"optimizes {sensitive!r}, whose stored values changed sign. Old "
            "and new trials would sit on opposite scales in one Pareto front. "
            "Start a new study, or drop the affected metric from "
            "objective_metrics."
        )
    if not has_trials:
        # Stamp ONLY a study with nothing in it yet. The stamp asserts "every
        # trial here was written by this encoding", and stamping a study that
        # already holds pre-encoding trials would make that assertion false --
        # a legacy study resumed once with an unaffected metric would be
        # marked compatible, and a later resume with `log_gamma` would then
        # sail past this guard and mix encodings after all. Leaving it
        # unstamped costs nothing: the check below re-evaluates against
        # whichever metrics are actually in use each time.
        study.set_user_attr(
            "bayesflow_hpo_objective_encoding", OBJECTIVE_ENCODING_VERSION
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
    elif any(d != "minimize" for d in directions):
        # Every value the objective returns is already in minimize space:
        # higher-is-better metrics are converted through METRIC_DIRECTIONS,
        # and the failure penalties are minimize-space too. Applying
        # "maximize" on top inverts that a second time, so a good raw
        # log_gamma of 1.5 (objective -1.5) would rank below a bad -25.5
        # (objective 25.5) -- and the +inf failure penalty would become the
        # single most attractive value in the study. Silently accepting the
        # override would make the search optimize for the worst model with
        # nothing in the output looking wrong.
        raise ValueError(
            "directions must be all 'minimize': the objective already "
            "converts every metric to minimize-is-better via "
            "bayesflow_hpo.objectives.METRIC_DIRECTIONS, so a 'maximize' "
            f"entry inverts it a second time. Got {directions!r}. To optimize "
            "a higher-is-better metric, register its direction with "
            "register_metric_direction() and leave directions=None."
        )
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
    sampler: str | optuna.samplers.BaseSampler | None = None,
    metric_constraints_soft: list[MetricConstraintSpec] | None = None,
    warm_start_from: Any | None,
    warm_start_top_k: int,
    qmc_startup_trials: int = 0,
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
        sampler=sampler,
        metric_constraints_soft=metric_constraints_soft,
        warm_start_from=warm_start_from,
        warm_start_top_k=warm_start_top_k,
        qmc_startup_trials=qmc_startup_trials,
    )
    _guard_resumed_study(study, objective.config.objective_metrics)

    # Auto-detect n_startup_trials from sampler if not set explicitly.
    cfg = objective.config
    if cfg.pruning_n_startup_trials is None:
        resolved = _resolve_n_startup_trials(study.sampler)
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


def _resolve_memory_budget(
    *,
    max_memory_mb: float | None | str,
    safety_margin: float,
) -> float | None:
    """Resolve ``max_memory_mb`` with optional auto-detection."""
    if not (0.0 <= float(safety_margin) < 1.0):
        raise ValueError(
            f"memory_safety_margin must be in [0, 1), got {safety_margin}."
        )

    if max_memory_mb is None:
        return None

    if isinstance(max_memory_mb, bool):
        raise ValueError(
            "max_memory_mb must be float, None, or 'auto', got bool."
        )
    if isinstance(max_memory_mb, (float, int)):
        return float(max_memory_mb)

    if isinstance(max_memory_mb, str):
        if max_memory_mb != "auto":
            raise ValueError(
                f"max_memory_mb must be float, None, or 'auto', got {max_memory_mb!r}."
            )
        resolved = _detect_gpu_memory_mb(float(safety_margin))
        if resolved is None:
            logger.warning(
                "max_memory_mb='auto' requested but CUDA memory could not be detected; "
                "disabling memory budget."
            )
            return None
        logger.info(
            "Resolved max_memory_mb='auto' to %.1f MB (safety_margin=%.2f).",
            resolved,
            safety_margin,
        )
        return float(resolved)

    raise ValueError(
        "max_memory_mb must be float, None, or 'auto', got "
        f"{type(max_memory_mb).__name__}."
    )
