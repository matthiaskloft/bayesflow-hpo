"""Generic Optuna objective for BayesFlow HPO.

This module implements the core trial loop: sample → build → compile →
train → validate → return objective values.  Each ``__call__`` invocation
maps one Optuna trial to a minimize-all tuple of metric and cost scores.

Key design decisions:

- **Pre-training budget rejection**: Trials exceeding ``max_param_count``
  or ``max_memory_mb`` are rejected *before* training to save GPU time.
  These trials still return penalty values so Optuna records them, but
  they are flagged via ``rejected_reason`` and excluded from the trained
  trial count.

- **Two-phase param count check**: First a heuristic estimate (fast, no
  GPU), then an exact count after lazy Keras weight initialization.

- **Three hooks**: ``build_approximator_fn``, ``train_fn``, and
  ``validate_fn`` let callers replace the build, train, and validate
  steps while reusing the full trial lifecycle.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf
import optuna

from bayesflow_hpo.builders.workflow import (
    _compile_for_compat,
    _make_cosine_decay_optimizer,
    build_continuous_approximator,
)
from bayesflow_hpo.objectives import (
    FAILED_TRIAL_CAL_ERROR,
    FAILED_TRIAL_COST,
    MAX_PARAM_COUNT,
    compute_inference_time_ratio,
    extract_multi_objective_values,
    get_param_count,
    normalize_param_count,
)
from bayesflow_hpo.optimization.callbacks import (
    MovingAverageEarlyStopping,
    OptunaReportCallback,
)
from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool
from bayesflow_hpo.optimization.cleanup import cleanup_trial
from bayesflow_hpo.optimization.constraints import estimate_peak_memory_mb
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.types import BuildApproximatorFn, TrainFn, ValidateFn
from bayesflow_hpo.validation.data import ValidationDataset

logger = logging.getLogger(__name__)


def default_train_fn(
    approximator: Any,
    simulator: bf.simulators.Simulator,
    hparams: dict[str, Any],
    callbacks: list[Any],
) -> None:
    """Train via ``approximator.fit(simulator=..., ...)``.

    This is the default used by ``optimize()`` when ``train_fn`` is ``None``.
    Reads ``epochs``, ``batches_per_epoch``, and ``batch_size`` from
    ``hparams`` (injected by the objective before calling).

    Parameters
    ----------
    approximator
        Compiled BayesFlow approximator.
    simulator
        BayesFlow simulator for online training.
    hparams
        Hyperparameters dict (must contain ``epochs``, ``batches_per_epoch``,
        and optionally ``batch_size``).
    callbacks
        Keras callbacks (early stopping, Optuna reporter, etc.).
    """
    approximator.fit(
        simulator=simulator,
        epochs=int(hparams["epochs"]),
        batch_size=int(hparams.get("batch_size", 256)),
        batches_per_epoch=int(hparams["batches_per_epoch"]),
        callbacks=callbacks,
    )


def default_validate_fn(
    approximator: Any,
    validation_data: ValidationDataset,
    n_posterior_samples: int,
) -> dict[str, float]:
    """Run the built-in validation pipeline and return metric dict.

    This is the default used by ``optimize()`` when ``validate_fn`` is ``None``.
    Wraps ``run_validation_pipeline()`` and returns its summary as a flat dict.

    Parameters
    ----------
    approximator
        Trained BayesFlow approximator with a ``.sample()`` method.
    validation_data
        Pre-generated validation dataset.
    n_posterior_samples
        Number of posterior draws per simulation.

    Returns
    -------
    dict[str, float]
        Metric name → value mapping (e.g. ``{"calibration_error": 0.05}``).
    """
    from bayesflow_hpo.validation.pipeline import run_validation_pipeline

    result = run_validation_pipeline(
        approximator=approximator,
        validation_data=validation_data,
        n_posterior_samples=n_posterior_samples,
    )
    return dict(result.summary)


def _validate_metric_keys(
    raw: dict[str, float],
    objective_metrics: list[str],
) -> dict[str, float]:
    """Validate and sanitize metric dict from a custom validate_fn.

    - Missing keys → replaced with penalty value + warning.
    - NaN/Inf values → replaced with penalty value + warning.

    Returns a cleaned copy of the dict.
    """
    cleaned = dict(raw)
    for key in objective_metrics:
        if key not in cleaned:
            logger.warning(
                "validate_fn output missing metric %r — using penalty value", key
            )
            cleaned[key] = FAILED_TRIAL_CAL_ERROR
        elif not math.isfinite(cleaned[key]):
            logger.warning(
                "validate_fn returned non-finite value for %r — using penalty", key
            )
            cleaned[key] = FAILED_TRIAL_CAL_ERROR
    return cleaned


@dataclass
class ObjectiveConfig:
    """Configuration for one objective function instance.

    Parameters
    ----------
    simulator, adapter
        BayesFlow simulator and adapter.
    search_space
        Composite search space defining the tunable dimensions.
    validation_data
        Pre-generated :class:`ValidationDataset`.
    epochs
        Maximum training epochs per trial (default 200).
    batches_per_epoch
        Online simulation batches per epoch (default 50).
    early_stopping_patience
        Moving-average patience epochs (default 5).
    early_stopping_window
        Moving-average window size (default 7).
    max_param_count
        Trials with actual param count above this are rejected
        before training (default 1 000 000).
    max_memory_mb
        Optional peak-memory budget in MB (disabled by default).
    n_posterior_samples
        Posterior draws for final validation (default 500).
    n_intermediate_posterior_samples
        Posterior draws for mid-training pruning validation
        (default 250).
    intermediate_validation_interval
        Run a lightweight validation every *n* epochs for pruning
        (default 10).
    intermediate_validation_warmup
        Skip the first *n* epochs before intermediate validation
        (default 10).
    pruning_n_startup_trials
        Minimum completed trials before multi-objective pruning
        activates (default 5).
    objective_metrics
        List of metric keys to optimize simultaneously.
        Default ``["calibration_error", "nrmse"]``.
    objective_mode
        ``"pareto"`` (default) — each metric becomes its own Optuna
        objective; returns ``len(objective_metrics) + 1`` values.
        ``"mean"`` — arithmetic mean of the listed metrics forms a
        single scalar; returns 2 values ``(mean, cost_score)``.
    cost_metric
        Which cost objective to use as the last Optuna objective.
        ``"inference_time"`` (default) or ``"param_count"``.
    report_frequency
        How often (in epochs) the ``OptunaReportCallback`` stores
        ``epoch_{N}_loss`` user attributes on each trial (default 10).
    checkpoint_pool
        Optional :class:`CheckpointPool` to persist the best trial
        weights.  When ``None`` a default pool of size 5 is created.
    build_approximator_fn
        Optional custom build function ``(hparams) -> Approximator``.
        Must return an **uncompiled** approximator.
    train_fn
        Optional custom training function
        ``(approximator, simulator, hparams, callbacks) -> None``.
    validate_fn
        Optional custom validation function
        ``(approximator, validation_data, n_posterior_samples) ->
        dict[str, float]``.
    """

    simulator: bf.simulators.Simulator
    adapter: bf.adapters.Adapter
    search_space: CompositeSearchSpace
    validation_data: ValidationDataset | None = None
    epochs: int = 200
    batches_per_epoch: int = 50
    early_stopping_patience: int = 5
    early_stopping_window: int = 7
    max_param_count: int = MAX_PARAM_COUNT
    max_memory_mb: float | None = None
    n_posterior_samples: int = 500
    n_intermediate_posterior_samples: int = 250
    intermediate_validation_interval: int = 10
    intermediate_validation_warmup: int = 10
    pruning_n_startup_trials: int = 5
    objective_metrics: list[str] = field(
        default_factory=lambda: ["calibration_error", "nrmse"]
    )
    objective_mode: str = "pareto"
    cost_metric: str = "inference_time"
    checkpoint_pool: CheckpointPool | None = None
    report_frequency: int = 10
    build_approximator_fn: BuildApproximatorFn | None = None
    train_fn: TrainFn | None = None
    validate_fn: ValidateFn | None = None

    def __post_init__(self):
        if self.objective_mode not in ("mean", "pareto"):
            raise ValueError(
                f"Unknown objective_mode: {self.objective_mode!r}. "
                f"Expected 'mean' or 'pareto'."
            )
        if self.cost_metric not in ("inference_time", "param_count"):
            raise ValueError(
                f"Unknown cost_metric: {self.cost_metric!r}. "
                f"Expected 'inference_time' or 'param_count'."
            )


def _log_trial_summary(
    trial: optuna.Trial,
    values: tuple[float, ...],
    param_count: int,
    training_time: float,
    metric_label: str = "obj[0]",
) -> None:
    """Log a concise one-line summary after a trial completes."""
    params_label = (
        f"{param_count / 1e6:.2f}M"
        if param_count >= 1e6
        else f"{param_count / 1e3:.1f}K"
        if param_count >= 1e3
        else str(param_count)
    )
    parts = [
        f"Trial #{trial.number} done ({training_time:.0f}s)",
        f"{metric_label}: {values[0]:.4f}",
        f"params: {params_label}",
    ]
    nrmse = trial.user_attrs.get("nrmse")
    if nrmse is not None:
        parts.append(f"nrmse: {nrmse:.4f}")
    corr = trial.user_attrs.get("correlation")
    if corr is not None:
        parts.append(f"corr: {corr:.4f}")
    logger.info(" | ".join(parts))


class GenericObjective:
    """Optuna objective returning a minimize-all tuple of metric and cost scores.

    Each call samples hyperparameters, builds the model, trains it,
    validates, and returns an objective tuple.  Failed, pruned, or
    budget-rejected trials return penalty values.

    The trial lifecycle:

    1. Sample hparams from search_space
    2. Inject training config into hparams
    3. Budget pre-check (memory estimate)
    4. BUILD approximator (custom or default)
    5. COMPILE with Adam + CosineDecay
    6. Exact param count check
    7. TRAIN (custom or default)
    8. VALIDATE (custom or default)
    9. Cost scoring
    10. Checkpoint pool
    11. Logging
    """

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        if config.checkpoint_pool is None:
            self._checkpoint_pool = CheckpointPool()
        else:
            self._checkpoint_pool = config.checkpoint_pool

    @property
    def checkpoint_pool(self) -> CheckpointPool:
        """The checkpoint pool used by this objective."""
        return self._checkpoint_pool

    @property
    def _metric_label(self) -> str:
        """Human-readable label for the first objective value in logs."""
        cfg = self.config
        if cfg.objective_mode == "pareto":
            return cfg.objective_metrics[0]
        return f"mean({'+'.join(cfg.objective_metrics)})"

    @property
    def n_objectives(self) -> int:
        """Number of objective values returned per trial."""
        if self.config.objective_mode == "pareto":
            return len(self.config.objective_metrics) + 1  # metrics + cost
        return 2  # mean + cost

    def _penalty(self) -> tuple[float, ...]:
        """Return penalty values matching the expected objective shape."""
        n = self.n_objectives
        if n == 2:
            return (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_COST)
        return tuple([FAILED_TRIAL_CAL_ERROR] * (n - 1)) + (FAILED_TRIAL_COST,)

    def __call__(self, trial: optuna.Trial) -> tuple[float, ...]:
        """Execute one HPO trial: sample → build → compile → train → validate.

        Returns
        -------
        tuple[float, ...]
            Objective values (all minimize-is-better).  Shape depends on
            ``objective_mode``: 2 values for ``"mean"``, N+1 for ``"pareto"``.
            Failed or budget-rejected trials return penalty values.
        """
        config = self.config

        # --- Step 1: Sample hparams ---
        params = config.search_space.sample(trial)

        # --- Step 2: Inject training config ---
        params["epochs"] = config.epochs
        params["batches_per_epoch"] = config.batches_per_epoch

        # --- Step 3: Budget pre-check (memory) ---
        estimated_memory = estimate_peak_memory_mb(params)
        trial.set_user_attr("estimated_peak_memory_mb", float(estimated_memory))
        if (
            config.max_memory_mb is not None
            and estimated_memory > config.max_memory_mb
        ):
            trial.set_user_attr("rejected_reason", "memory_budget")
            logger.info(
                "Trial #%d rejected: estimated %.0f MB > budget %.0f MB",
                trial.number, estimated_memory, config.max_memory_mb,
            )
            return self._penalty()

        # --- Step 4: BUILD approximator ---
        try:
            if config.build_approximator_fn is not None:
                approximator = config.build_approximator_fn(params)
            else:
                approximator = build_continuous_approximator(
                    params, config.adapter, config.search_space,
                )
        except Exception as exc:
            logger.warning(
                "Trial #%d: build failed: %s", trial.number, exc,
            )
            trial.set_user_attr("rejected_reason", "build_failed")
            trial.set_user_attr("build_error", str(exc))
            cleanup_trial()
            return self._penalty()

        # --- Step 5: COMPILE with Adam + CosineDecay ---
        if config.train_fn is None and "initial_lr" not in params:
            logger.warning(
                "Trial #%d: 'initial_lr' not in hparams, defaulting to 1e-3. "
                "Add 'initial_lr' to your search space or provide a custom "
                "train_fn.",
                trial.number,
            )
        initial_lr = float(params.get("initial_lr", 1e-3))
        decay_steps = config.batches_per_epoch * config.epochs
        try:
            optimizer = _make_cosine_decay_optimizer(
                initial_lr, decay_steps,
            )
            _compile_for_compat(approximator, optimizer)
        except TypeError:
            pass  # _compile_for_compat handles TypeError internally
        except Exception as exc:
            logger.warning(
                "Trial #%d: compile failed: %s", trial.number, exc,
            )
            trial.set_user_attr("rejected_reason", "compile_failed")
            trial.set_user_attr("compile_error", str(exc))
            cleanup_trial()
            return self._penalty()

        # --- Step 6: Exact param count check ---
        try:
            dummy = config.simulator.sample((2,))
            adapted = config.adapter(dummy)
            if hasattr(approximator, "build_from_data"):
                approximator.build_from_data(adapted)
            else:
                approximator.compute_loss(adapted)
            param_count_actual = get_param_count(approximator)
            trial.set_user_attr("param_count", int(param_count_actual))
            if param_count_actual > config.max_param_count:
                trial.set_user_attr("rejected_reason", "param_budget")
                logger.info(
                    "Trial #%d rejected: %d params > budget %d",
                    trial.number, param_count_actual, config.max_param_count,
                )
                cleanup_trial()
                return self._penalty()
        except MemoryError:
            raise
        except Exception as exc:
            logger.warning(
                "Trial #%d: param count probe failed, rejecting trial: %s",
                trial.number, exc,
            )
            trial.set_user_attr("rejected_reason", "param_probe_failed")
            trial.set_user_attr("param_probe_error", str(exc))
            cleanup_trial()
            return self._penalty()

        # --- Callbacks ---
        callbacks: list[Any] = [
            MovingAverageEarlyStopping(
                monitor="loss",
                window=config.early_stopping_window,
                patience=config.early_stopping_patience,
                restore_best_weights=True,
            ),
            OptunaReportCallback(
                trial, monitor="loss",
                report_frequency=config.report_frequency,
            ),
        ]

        if config.validation_data is not None:
            from bayesflow_hpo.optimization.validation_callback import (
                PeriodicValidationCallback,
            )

            callbacks.append(
                PeriodicValidationCallback(
                    trial=trial,
                    approximator=approximator,
                    validation_data=config.validation_data,
                    interval=config.intermediate_validation_interval,
                    warmup=config.intermediate_validation_warmup,
                    n_posterior_samples=config.n_intermediate_posterior_samples,
                    n_startup_trials=config.pruning_n_startup_trials,
                )
            )

        # --- Step 7: TRAIN ---
        t_train_start = time.perf_counter()
        try:
            if config.train_fn is not None:
                config.train_fn(approximator, config.simulator, params, callbacks)
            else:
                default_train_fn(approximator, config.simulator, params, callbacks)
        except optuna.TrialPruned:
            cleanup_trial()
            raise
        except Exception as exc:
            logger.warning("Trial %d failed during training: %s", trial.number, exc)
            trial.set_user_attr("training_error", str(exc))
            cleanup_trial()
            return self._penalty()
        training_time = time.perf_counter() - t_train_start
        trial.set_user_attr("training_time_s", round(training_time, 2))

        # --- Step 8: VALIDATE ---
        inference_time = 0.0
        try:
            if config.validation_data is not None:
                actual_validate = (
                    config.validate_fn
                    if config.validate_fn is not None
                    else default_validate_fn
                )
                t_val_start = time.perf_counter()
                raw = actual_validate(
                    approximator,
                    config.validation_data,
                    config.n_posterior_samples,
                )
                inference_time = time.perf_counter() - t_val_start
                metrics_summary = _validate_metric_keys(
                    raw, config.objective_metrics,
                )

                trial.set_user_attr(
                    "inference_time_s", round(inference_time, 2),
                )
                for key, val in metrics_summary.items():
                    trial.set_user_attr(key, round(float(val), 6))

                # Wrap for extract_multi_objective_values compatibility.
                metrics = {"summary": metrics_summary}
            else:
                # No validation data — use a penalty-like fallback.
                metrics = {
                    "summary": {
                        k: FAILED_TRIAL_CAL_ERROR
                        for k in config.objective_metrics
                    }
                }

        except optuna.TrialPruned:
            cleanup_trial()
            raise
        except Exception as exc:
            logger.warning(
                "Trial %d failed during final validation: %s",
                trial.number, exc,
            )
            trial.set_user_attr("validation_error", str(exc))
            values = self._penalty()
            _log_trial_summary(
                trial, values, -1, training_time, self._metric_label,
            )
            cleanup_trial()
            return values

        # --- Step 9: Cost score ---
        try:
            param_count = get_param_count(approximator)
        except (TypeError, ValueError) as exc:
            logger.warning("Trial %d: could not count params: %s", trial.number, exc)
            param_count = -1
        trial.set_user_attr("param_count", param_count)

        if config.cost_metric == "inference_time":
            vd = config.validation_data
            n_sims = sum(
                len(next(iter(batch.values()))) if batch else 0
                for batch in vd.simulations
            ) if vd is not None else 0
            cost_score = compute_inference_time_ratio(
                inference_time,
                sim_time_per_sim=vd.sim_time_per_sim if vd is not None else None,
                n_sims=n_sims,
            )
            trial.set_user_attr(
                "inference_time_ratio", round(cost_score, 6),
            )
        else:
            cost_score = normalize_param_count(
                param_count,
                max_count=config.max_param_count,
            )

        values = extract_multi_objective_values(
            metrics,
            cost_score,
            objective_metrics=config.objective_metrics,
            objective_mode=config.objective_mode,
        )

        # --- Step 10: Checkpoint pool ---
        self._checkpoint_pool.maybe_save(
            trial_number=trial.number,
            objective_value=values[0],
            approximator=approximator,
        )

        # --- Step 11: Per-trial summary log ---
        _log_trial_summary(
            trial, values, param_count, training_time, self._metric_label,
        )

        cleanup_trial()
        return values
