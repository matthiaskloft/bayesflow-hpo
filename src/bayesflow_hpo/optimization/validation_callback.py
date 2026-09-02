"""Periodic validation callback for mid-training pruning.

Runs a lightweight validation every *interval* epochs and uses a
pluggable pruning strategy for multi-objective studies.

For single-objective studies, the standard ``trial.report()`` /
``trial.should_prune()`` API is used with the study's pruner.

For multi-objective studies (the default in bayesflow_hpo), Optuna
does not support ``trial.report()`` (Issue #3450, open since April
2022).  Instead, one of three custom pruning strategies is applied:

- ``"dominance"`` — per-objective normalized median check (AND rule).
  Simplified adaptation of MO-ASHA's dominance-based promotion
  (Schmucker et al., 2021).
- ``"mo-sha"`` — non-dominated sorting at each step, bottom-fraction
  pruning per MO-ASHA Algorithm 2 (Schmucker et al., 2021).
- ``"primary"`` — single-metric median pruning on a user-chosen
  objective (equivalent to Optuna's MedianPruner; Akiba et al., 2019).

Strategy implementations live in
:mod:`bayesflow_hpo.optimization.pruning_strategies`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from keras.callbacks import Callback

from bayesflow_hpo.objectives import _metric_to_minimize
from bayesflow_hpo.optimization.pruning_strategies import (
    should_prune_dominance,
    should_prune_mo_sha,
    should_prune_primary,
)
from bayesflow_hpo.types import ValidateFn
from bayesflow_hpo.validation.data import ValidationDataset

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = {"none", "dominance", "mo-sha", "primary"}

#: Fallback used when ``ObjectiveConfig.pruning_n_startup_trials`` is left
#: unresolved. ``optimize()`` normally auto-detects it from the sampler, but
#: building an objective directly bypasses that, and the pruning strategies
#: compare this against an int.
DEFAULT_PRUNING_N_STARTUP_TRIALS = 5


class PeriodicValidationCallback(Callback):
    """Run validation every *interval* epochs and report to Optuna.

    For single-objective studies the first metric in ``objective_metrics``
    is reported via ``trial.report()`` and pruning uses the study's
    pruner.  For multi-objective studies (where ``trial.report()`` is
    unsupported), per-metric user attributes are stored and the
    configured ``pruning_strategy`` decides whether to prune.

    Parameters
    ----------
    trial
        Current Optuna trial.
    approximator
        Trained approximator with a ``.sample()`` method (updated
        in-place during training).
    validation_data
        Pre-generated
        :class:`~bayesflow_hpo.validation.data.ValidationDataset`.
    interval
        Run validation every *interval* epochs.  Default 10.
    warmup
        Skip the first *warmup* epochs before running validation.
        Default 10.
    n_posterior_samples
        Number of posterior draws for intermediate validation.
        Default 250.
    n_startup_trials
        Minimum completed trials before multi-objective pruning
        activates.  ``None`` (the default) resolves to
        :data:`DEFAULT_PRUNING_N_STARTUP_TRIALS`, because
        ``ObjectiveConfig.pruning_n_startup_trials`` is auto-detected by
        ``optimize()`` and stays ``None`` when an objective is built
        directly.
    validate_fn
        Optional custom validation function with signature
        ``(approximator, validation_data, n_posterior_samples) ->
        dict[str, float]``.  When provided, replaces the default
        ``run_validation_pipeline`` for intermediate pruning.  The
        returned dict must contain all keys in ``objective_metrics``.
    pruning_strategy
        Multi-objective pruning strategy.  One of ``"dominance"``
        (default), ``"mo-sha"``, ``"primary"``, or ``"none"``.
        For ``"primary"``, pass a tuple ``("primary", metric_name)``
        to specify which metric to prune on (defaults to
        ``objective_metrics[0]``).
    objective_metrics
        Metric keys to compute during intermediate validation.
        Defaults to ``["calibration_error", "nrmse"]``.
    early_stopping_patience
        Validation checks without improvement before stopping. ``None``
        disables early stopping. This is intended for horizon-free schedules;
        finite-budget schedules should run to their horizon.
    early_stopping_window
        Number of validation scores in the moving average.
    early_stopping_monitor
        Validation objective used for stopping. ``"objective_mean"`` (default) averages
        all objective metrics after converting them to minimize-is-better
        values. A metric name selects that metric alone.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        approximator: Any,
        validation_data: ValidationDataset,
        interval: int = 10,
        warmup: int = 10,
        n_posterior_samples: int = 250,
        n_startup_trials: int | None = None,
        validate_fn: ValidateFn | None = None,
        pruning_strategy: str | tuple[str, str] = "dominance",
        objective_metrics: list[str] | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_window: int = 1,
        early_stopping_monitor: str = "objective_mean",
    ):
        super().__init__()
        self.trial = trial
        self.approximator = approximator
        self.validation_data = validation_data
        self.interval = interval
        self.warmup = warmup
        self.n_posterior_samples = n_posterior_samples
        # `optimize()` auto-detects this from the sampler, but building an
        # objective directly leaves it None, and every pruning strategy
        # compares it against an int.
        self.n_startup_trials = (
            DEFAULT_PRUNING_N_STARTUP_TRIALS
            if n_startup_trials is None
            else n_startup_trials
        )
        self.validate_fn = validate_fn
        self._step = 0  # monotonic step counter for Optuna
        self._consecutive_failures = 0
        self._is_multi_objective = len(trial.study.directions) > 1
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_window = early_stopping_window
        self.early_stopping_monitor = early_stopping_monitor
        self._early_stopping_values: list[float] = []
        self._early_stopping_wait = 0
        self.best_validation_score = np.inf
        self.best_weights: Any = None

        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1 or None.")
        if early_stopping_window < 1:
            raise ValueError("early_stopping_window must be >= 1.")

        # Parse pruning strategy.
        if isinstance(pruning_strategy, tuple):
            if (
                len(pruning_strategy) != 2
                or pruning_strategy[0] != "primary"
            ):
                raise ValueError(
                    f"Tuple pruning_strategy must be "
                    f"('primary', metric_name), got {pruning_strategy!r}"
                )
            self._strategy_name = "primary"
            self._primary_metric: str | None = pruning_strategy[1]
        else:
            if pruning_strategy not in _VALID_STRATEGIES:
                raise ValueError(
                    f"Unknown pruning_strategy: {pruning_strategy!r}. "
                    f"Expected one of {sorted(_VALID_STRATEGIES)}."
                )
            self._strategy_name = pruning_strategy
            self._primary_metric = None

        # Resolve objective_metrics with backward-compatible default.
        self.objective_metrics = (
            objective_metrics
            if objective_metrics is not None
            else ["calibration_error", "nrmse"]
        )
        if (
            self.early_stopping_monitor != "objective_mean"
            and self.early_stopping_monitor not in self.objective_metrics
        ):
            raise ValueError(
                "early_stopping_monitor must be 'objective_mean' or one of "
                "objective_metrics, got "
                f"{self.early_stopping_monitor!r}."
            )

        # Default primary metric to first objective metric.
        if self._strategy_name == "primary" and self._primary_metric is None:
            self._primary_metric = self.objective_metrics[0]

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        """Run validation and check for pruning at scheduled intervals.

        Skips epochs before ``warmup`` and non-interval epochs.
        After 3 consecutive validation failures, logs a warning
        (but does not prune — the trial continues without pruning).
        """
        if epoch < self.warmup:
            return
        if (epoch - self.warmup) % self.interval != 0:
            return

        raw_scores = self._run_lightweight_validation()
        if raw_scores is None:
            self._consecutive_failures += 1
            if self._consecutive_failures == 3:
                logger.warning(
                    "Trial %d: %d consecutive intermediate validation "
                    "failures — pruning may be ineffective.",
                    self.trial.number,
                    self._consecutive_failures,
                )
            return
        self._consecutive_failures = 0

        self._step += 1

        self._update_early_stopping(raw_scores)
        scores = {
            metric: _metric_to_minimize(metric, float(raw_scores[metric]))
            for metric in self.objective_metrics
        }

        if self._is_multi_objective:
            # Store per-metric user attrs for strategy functions.
            for metric, val in scores.items():
                self.trial.set_user_attr(
                    f"val_{metric}_step_{self._step}",
                    round(float(val), 6),
                )

            should_prune = self._evaluate_pruning(scores)
            if should_prune:
                raise optuna.TrialPruned()
        else:
            # Single-objective: use first metric with Optuna's pruner.
            primary_val = scores[self.objective_metrics[0]]
            self.trial.report(primary_val, step=self._step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def _update_early_stopping(self, scores: dict[str, float]) -> None:
        """Stop on a moving average of the configured validation objective."""
        if self.early_stopping_patience is None:
            return

        if self.early_stopping_monitor == "objective_mean":
            value = float(
                np.mean(
                    [
                        _metric_to_minimize(metric, float(scores[metric]))
                        for metric in self.objective_metrics
                    ]
                )
            )
        else:
            value = _metric_to_minimize(
                self.early_stopping_monitor,
                float(scores[self.early_stopping_monitor]),
            )
        self._early_stopping_values.append(value)
        if len(self._early_stopping_values) > self.early_stopping_window:
            self._early_stopping_values.pop(0)
        moving_average = float(np.mean(self._early_stopping_values))

        if moving_average < self.best_validation_score:
            self.best_validation_score = moving_average
            self._early_stopping_wait = 0
            self.best_weights = self.approximator.get_weights()
            return

        self._early_stopping_wait += 1
        if self._early_stopping_wait >= self.early_stopping_patience:
            self.approximator.stop_training = True
            if self.best_weights is not None:
                self.approximator.set_weights(self.best_weights)

    def on_train_end(self, logs: Any = None) -> None:
        """Restore the best validation weights when training reaches its cap."""
        if self.early_stopping_patience is not None and self.best_weights is not None:
            self.approximator.set_weights(self.best_weights)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_pruning(self, scores: dict[str, float]) -> bool:
        """Dispatch to the configured pruning strategy."""
        if self._strategy_name == "none":
            return False
        if self._strategy_name == "dominance":
            return should_prune_dominance(
                self.trial, scores, self._step, self.n_startup_trials
            )
        if self._strategy_name == "mo-sha":
            return should_prune_mo_sha(
                self.trial, scores, self._step, self.n_startup_trials
            )
        if self._strategy_name == "primary":
            if self._primary_metric is None:  # pragma: no cover - set together
                raise RuntimeError(
                    'pruning_strategy "primary" without a metric name.'
                )
            primary_score = scores[self._primary_metric]
            return should_prune_primary(
                self.trial,
                float(primary_score),
                self._primary_metric,
                self._step,
                self.n_startup_trials,
            )
        return False  # pragma: no cover

    def _run_lightweight_validation(self) -> dict[str, float] | None:
        """Compute objective_metrics via validation pipeline."""
        try:
            if self.validate_fn is not None:
                result_dict = self.validate_fn(
                    self.approximator,
                    self.validation_data,
                    self.n_posterior_samples,
                )
                # Validate that all objective_metrics are present.
                missing = [
                    k for k in self.objective_metrics
                    if k not in result_dict
                ]
                if missing:
                    logger.warning(
                        "validate_fn output missing metrics %s — "
                        "skipping pruning this step.",
                        missing,
                    )
                    return None
                return {
                    k: float(result_dict[k])
                    for k in self.objective_metrics
                }
            else:
                from bayesflow_hpo.validation.pipeline import (
                    run_validation_pipeline,
                )

                result = run_validation_pipeline(
                    approximator=self.approximator,
                    validation_data=self.validation_data,
                    n_posterior_samples=self.n_posterior_samples,
                    metrics=self.objective_metrics,
                )
                extracted = {
                    k: float(result.summary[k])
                    for k in self.objective_metrics
                    if k in result.summary
                }
                missing = [
                    k for k in self.objective_metrics
                    if k not in extracted
                ]
                if missing:
                    logger.warning(
                        "run_validation_pipeline output missing "
                        "metrics %s — skipping pruning this step.",
                        missing,
                    )
                    return None
                return extracted
        except optuna.TrialPruned:
            raise
        except Exception:
            logger.warning(
                "Intermediate validation failed (trial %d)",
                self.trial.number,
                exc_info=True,
            )
            return None
