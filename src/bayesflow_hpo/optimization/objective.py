"""Generic Optuna objective for BayesFlow HPO."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf
import optuna

from bayesflow_hpo.builders.workflow import WorkflowBuildConfig, build_workflow
from bayesflow_hpo.objectives import (
    FAILED_TRIAL_CAL_ERROR,
    FAILED_TRIAL_PARAM_SCORE,
    MAX_PARAM_COUNT,
    MIN_PARAM_COUNT,
    extract_objective_values,
    get_param_count,
)
from bayesflow_hpo.optimization.callbacks import (
    MovingAverageEarlyStopping,
    OptunaReportCallback,
)
from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool
from bayesflow_hpo.optimization.cleanup import cleanup_trial
from bayesflow_hpo.optimization.constraints import (
    estimate_param_count,
    estimate_peak_memory_mb,
)
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.validation.data import ValidationDataset

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveConfig:
    """Configuration for one objective function instance.

    Parameters
    ----------
    simulator, adapter
        BayesFlow simulator and adapter.
    search_space
        Composite search space defining the tunable dimensions.
    inference_conditions
        Optional list of condition variable names.
    validation_data
        Pre-generated :class:`ValidationDataset`.  When ``None`` the
        objective falls back to the final training loss.
    param_keys, data_keys
        Keys used by the validation inference function.
    epochs
        Maximum training epochs per trial (default 200).
    batches_per_epoch
        Online simulation batches per epoch (default 50).
    early_stopping_patience
        Moving-average patience epochs (default 5).
    early_stopping_window
        Moving-average window size (default 7).
    max_param_count
        Trials with estimated param count above this are rejected
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
    metrics
        Metric names for final validation (default ``DEFAULT_METRICS``).
    objective_metric
        Key in the validation summary used as the first HPO objective
        (default ``"calibration_error"``).
    checkpoint_pool
        Optional :class:`CheckpointPool` to persist the best trial
        weights.  When ``None`` a default pool of size 5 is created.
    train_fn
        Optional custom training function
        ``(workflow, params, callbacks) -> None``.
    """

    simulator: bf.simulators.Simulator
    adapter: bf.adapters.Adapter
    search_space: CompositeSearchSpace
    inference_conditions: list[str] | None = None
    validation_data: ValidationDataset | None = None
    param_keys: list[str] = field(default_factory=list)
    data_keys: list[str] = field(default_factory=list)
    epochs: int = 200
    batches_per_epoch: int = 50
    early_stopping_patience: int = 5
    early_stopping_window: int = 7
    max_param_count: int = MAX_PARAM_COUNT
    param_budget_penalty: tuple[float, float] = (
        FAILED_TRIAL_CAL_ERROR,
        FAILED_TRIAL_PARAM_SCORE,
    )
    max_memory_mb: float | None = None
    memory_budget_penalty: tuple[float, float] = (
        FAILED_TRIAL_CAL_ERROR,
        FAILED_TRIAL_PARAM_SCORE,
    )
    training_failure_penalty: tuple[float, float] = (
        FAILED_TRIAL_CAL_ERROR,
        FAILED_TRIAL_PARAM_SCORE,
    )
    n_posterior_samples: int = 500
    n_intermediate_posterior_samples: int = 250
    intermediate_validation_interval: int = 10
    intermediate_validation_warmup: int = 10
    pruning_n_startup_trials: int = 5
    metrics: list[str] | None = None
    objective_metric: str = "calibration_error"
    checkpoint_pool: CheckpointPool | None = None
    train_fn: Callable[[bf.BasicWorkflow, dict, list], None] | None = None


def _default_train_fn(
    workflow: bf.BasicWorkflow,
    params: dict[str, Any],
    callbacks: list[Any],
    *,
    epochs: int,
    batches_per_epoch: int,
) -> None:
    """Default training function using fit_online."""
    workflow.fit_online(
        epochs=epochs,
        batch_size=int(params.get("batch_size", 256)),
        num_batches_per_epoch=batches_per_epoch,
        callbacks=callbacks,
    )


def _log_trial_summary(
    trial: optuna.Trial,
    values: tuple[float, float],
    param_count: int,
    training_time: float,
) -> None:
    """Log a concise one-line summary after a trial completes."""
    cal_err = values[0]
    params_label = (
        f"{param_count / 1e6:.2f}M"
        if param_count >= 1e6
        else f"{param_count / 1e3:.1f}K"
        if param_count >= 1e3
        else str(param_count)
    )
    # Collect key metric attrs if available
    parts = [
        f"Trial #{trial.number} done ({training_time:.0f}s)",
        f"{trial.user_attrs.get('objective_metric', 'cal_error')}: {cal_err:.4f}",
        f"params: {params_label}",
    ]
    nrmse = trial.user_attrs.get("nrmse")
    if nrmse is not None:
        parts.append(f"nrmse: {nrmse:.4f}")
    logger.info(" | ".join(parts))


class GenericObjective:
    """Optuna objective returning ``(calibration_error, param_score)``.

    Each call samples hyperparameters, builds the model, trains it,
    validates, and returns a bi-objective tuple.  Failed, pruned, or
    budget-rejected trials return penalty values.
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

    def __call__(self, trial: optuna.Trial) -> tuple[float, float]:
        params = self.config.search_space.sample(trial)

        # --- Budget pre-check (param count) ---
        # Inject actual model dimensions so the heuristic is accurate.
        if "n_params" not in params:
            params["n_params"] = len(self.config.param_keys) if self.config.param_keys else 1
        if "n_conditions" not in params:
            params["n_conditions"] = len(self.config.inference_conditions) if self.config.inference_conditions else 0
        estimated = estimate_param_count(params)
        trial.set_user_attr("estimated_param_count", int(estimated))
        if estimated > self.config.max_param_count:
            trial.set_user_attr("rejected_reason", "param_budget")
            logger.info(
                "Trial #%d rejected: estimated %d params > budget %d",
                trial.number, estimated, self.config.max_param_count,
            )
            return self.config.param_budget_penalty

        # --- Budget pre-check (memory) ---
        estimated_memory = estimate_peak_memory_mb(params)
        trial.set_user_attr("estimated_peak_memory_mb", float(estimated_memory))
        if (
            self.config.max_memory_mb is not None
            and estimated_memory > self.config.max_memory_mb
        ):
            trial.set_user_attr("rejected_reason", "memory_budget")
            logger.info(
                "Trial #%d rejected: estimated %.0f MB > budget %.0f MB",
                trial.number, estimated_memory, self.config.max_memory_mb,
            )
            return self.config.memory_budget_penalty

        # --- Build model ---
        inference_net = self.config.search_space.inference_space.build(params)
        summary_net = None
        if self.config.search_space.summary_space is not None:
            summary_net = self.config.search_space.summary_space.build(params)

        workflow = build_workflow(
            simulator=self.config.simulator,
            adapter=self.config.adapter,
            inference_network=inference_net,
            summary_network=summary_net,
            params=params,
            config=WorkflowBuildConfig(
                inference_conditions=self.config.inference_conditions,
                batches_per_epoch=self.config.batches_per_epoch,
            ),
        )

        # --- Callbacks ---
        callbacks: list[Any] = [
            MovingAverageEarlyStopping(
                monitor="loss",
                window=self.config.early_stopping_window,
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            ),
            OptunaReportCallback(trial, monitor="loss"),
        ]

        # Periodic validation callback for pruning.
        if self.config.validation_data is not None:
            from bayesflow_hpo.optimization.validation_callback import (
                PeriodicValidationCallback,
            )

            approximator = getattr(workflow, "approximator", workflow)
            callbacks.append(
                PeriodicValidationCallback(
                    trial=trial,
                    approximator=approximator,
                    validation_data=self.config.validation_data,
                    param_keys=self.config.param_keys,
                    data_keys=self.config.data_keys,
                    interval=self.config.intermediate_validation_interval,
                    warmup=self.config.intermediate_validation_warmup,
                    n_posterior_samples=self.config.n_intermediate_posterior_samples,
                    n_startup_trials=self.config.pruning_n_startup_trials,
                )
            )

        # --- Train ---
        t_train_start = time.perf_counter()
        try:
            if self.config.train_fn is not None:
                self.config.train_fn(workflow, params, callbacks)
            else:
                _default_train_fn(
                    workflow,
                    params,
                    callbacks,
                    epochs=self.config.epochs,
                    batches_per_epoch=self.config.batches_per_epoch,
                )
        except optuna.TrialPruned:
            cleanup_trial()
            raise
        except Exception as exc:
            logger.warning("Trial %d failed during training: %s", trial.number, exc)
            trial.set_user_attr("training_error", str(exc))
            cleanup_trial()
            return self.config.training_failure_penalty
        training_time = time.perf_counter() - t_train_start
        trial.set_user_attr("training_time_s", round(training_time, 2))

        # --- Final validation ---
        try:
            if self.config.validation_data is not None:
                from bayesflow_hpo.validation.pipeline import run_validation_pipeline

                validation_result = run_validation_pipeline(
                    approximator=workflow.approximator,
                    validation_data=self.config.validation_data,
                    n_posterior_samples=self.config.n_posterior_samples,
                    metrics=self.config.metrics,
                )
                cal_error = validation_result.objective_scalar(
                    self.config.objective_metric
                )
                metrics = {"summary": {self.config.objective_metric: cal_error}}

                # Store validation timing and summary metrics as trial attrs.
                trial.set_user_attr(
                    "inference_time_s",
                    round(validation_result.timing.get("inference", 0.0), 2),
                )
                for key, val in validation_result.summary.items():
                    trial.set_user_attr(key, round(float(val), 6))
            else:
                hist_obj = getattr(workflow, "history", None)
                hist_dict = (
                    getattr(hist_obj, "history", {}) if hist_obj is not None else {}
                )
                last_loss = float(
                    hist_dict.get("loss", [FAILED_TRIAL_CAL_ERROR])[-1]
                )
                metrics = {"summary": {self.config.objective_metric: last_loss}}

        except optuna.TrialPruned:
            cleanup_trial()
            raise
        except Exception as exc:
            logger.warning(
                "Trial %d failed during final validation: %s",
                trial.number,
                exc,
            )
            trial.set_user_attr("validation_error", str(exc))
            # Fall back to training loss for the objective value.
            hist_obj = getattr(workflow, "history", None)
            hist_dict = (
                getattr(hist_obj, "history", {}) if hist_obj is not None else {}
            )
            fallback_loss = float(
                hist_dict.get("loss", [FAILED_TRIAL_CAL_ERROR])[-1]
            )
            values = (fallback_loss, FAILED_TRIAL_PARAM_SCORE)
            param_count = -1
            _log_trial_summary(trial, values, param_count, training_time)
            cleanup_trial()
            return values

        # --- Parameter count (separate from validation) ---
        try:
            param_count = get_param_count(workflow.approximator)
        except (TypeError, ValueError) as exc:
            logger.warning("Trial %d: could not count params: %s", trial.number, exc)
            param_count = -1  # normalize_param_count maps to 1.0 (worst)
        trial.set_user_attr("param_count", param_count)
        values = extract_objective_values(
            metrics,
            param_count,
            objective_metric=self.config.objective_metric,
            max_param_count=self.config.max_param_count,
        )

        # --- Checkpoint pool ---
        self._checkpoint_pool.maybe_save(
            trial_number=trial.number,
            objective_value=values[0],
            workflow=workflow,
        )

        # --- Per-trial summary log ---
        _log_trial_summary(trial, values, param_count, training_time)

        cleanup_trial()
        return values
