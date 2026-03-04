"""Generic Optuna objective for BayesFlow HPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bayesflow as bf
import optuna

from bayesflow_hpo.builders.workflow import WorkflowBuildConfig, build_workflow
from bayesflow_hpo.objectives import (
    FAILED_TRIAL_CAL_ERROR,
    FAILED_TRIAL_PARAM_SCORE,
    extract_objective_values,
    get_param_count,
)
from bayesflow_hpo.optimization.callbacks import (
    MovingAverageEarlyStopping,
    OptunaReportCallback,
)
from bayesflow_hpo.optimization.cleanup import cleanup_trial
from bayesflow_hpo.optimization.constraints import (
    estimate_param_count,
    estimate_peak_memory_mb,
)
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.pipeline import run_validation_pipeline


@dataclass
class ObjectiveConfig:
    """Configuration for one objective function instance."""

    simulator: bf.simulators.Simulator
    adapter: bf.adapters.Adapter
    search_space: CompositeSearchSpace
    inference_conditions: list[str] | None = None
    validation_data: ValidationDataset | None = None
    epochs: int = 200
    batches_per_epoch: int = 50
    early_stopping_patience: int = 15
    early_stopping_window: int = 15
    max_param_count: int = 2_000_000
    param_budget_penalty: tuple[float, float] = (FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_PARAM_SCORE)
    max_memory_mb: float | None = None
    memory_budget_penalty: tuple[float, float] = (
        FAILED_TRIAL_CAL_ERROR,
        FAILED_TRIAL_PARAM_SCORE,
    )
    n_posterior_samples: int = 500


class GenericObjective:
    """Optuna objective returning `(calibration_error, param_score)`."""

    def __init__(self, config: ObjectiveConfig):
        self.config = config

    def __call__(self, trial: optuna.Trial) -> tuple[float, float]:
        params = self.config.search_space.sample(trial)

        estimated = estimate_param_count(params)
        trial.set_user_attr("estimated_param_count", int(estimated))
        if estimated > self.config.max_param_count:
            return self.config.param_budget_penalty

        estimated_memory = estimate_peak_memory_mb(params)
        trial.set_user_attr("estimated_peak_memory_mb", float(estimated_memory))
        if (
            self.config.max_memory_mb is not None
            and estimated_memory > self.config.max_memory_mb
        ):
            return self.config.memory_budget_penalty

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
            config=WorkflowBuildConfig(inference_conditions=self.config.inference_conditions),
        )

        callbacks: list[Any] = [
            MovingAverageEarlyStopping(
                window=self.config.early_stopping_window,
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            ),
            OptunaReportCallback(trial),
        ]

        try:
            workflow.fit_online(
                epochs=self.config.epochs,
                batch_size=int(params.get("batch_size", 256)),
                num_batches_per_epoch=self.config.batches_per_epoch,
                callbacks=callbacks,
            )
        except optuna.TrialPruned:
            cleanup_trial()
            raise
        except Exception:
            cleanup_trial()
            return self.config.param_budget_penalty

        if self.config.validation_data is not None:
            validation_result = run_validation_pipeline(
                approximator=workflow.approximator,
                validation_data=self.config.validation_data,
                n_posterior_samples=self.config.n_posterior_samples,
            )
            metrics = validation_result["metrics"]
        else:
            history = getattr(workflow.approximator, "history", {})
            last_loss = float(history.get("loss", [FAILED_TRIAL_CAL_ERROR])[-1])
            metrics = {"summary": {"mean_cal_error": last_loss}}

        param_count = get_param_count(workflow.approximator)
        values = extract_objective_values(metrics, param_count)
        cleanup_trial()
        return values
