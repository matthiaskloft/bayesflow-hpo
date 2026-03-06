"""Periodic validation callback for mid-training pruning.

Runs a lightweight validation (calibration_error + nrmse only) every
*interval* epochs and uses the geometric mean for pruning decisions.

For single-objective studies, the standard ``trial.report()`` /
``trial.should_prune()`` API is used with the study's pruner.

For multi-objective studies (the default in bayesflow_hpo), Optuna
does not support ``trial.report()``.  Instead, a custom median-based
pruning strategy compares the current trial's intermediate score
against completed trials at the same step and prunes if it exceeds
the median.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from keras.callbacks import Callback

logger = logging.getLogger(__name__)

# Metrics computed during intermediate validation (fast subset).
_INTERMEDIATE_METRICS = ["calibration_error", "nrmse"]


def _should_prune_multi_objective(
    trial: optuna.Trial,
    score: float,
    step: int,
    n_startup_trials: int = 5,
) -> bool:
    """Return True if this trial should be pruned (multi-objective).

    Collects ``val_score_step_{step}`` from all COMPLETE non-rejected
    trials and prunes when the current *score* exceeds the median.
    Requires at least *n_startup_trials* reference scores before
    activating.

    Parameters
    ----------
    trial
        The running Optuna trial.
    score
        Current intermediate pruning score (geometric mean).
    step
        Monotonic step counter (1-indexed).
    n_startup_trials
        Minimum completed trials before pruning activates.
    """
    if n_startup_trials < 1:
        return False

    # NaN/Inf scores indicate a degenerate trial — prune immediately.
    if not np.isfinite(score):
        return True

    attr_key = f"val_score_step_{step}"
    completed_scores: list[float] = []

    for t in trial.study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    ):
        if "rejected_reason" in t.user_attrs:
            continue
        if t.number == trial.number:
            continue
        stored = t.user_attrs.get(attr_key)
        if stored is not None:
            val = float(stored)
            if np.isfinite(val):
                completed_scores.append(val)

    if len(completed_scores) < n_startup_trials:
        return False

    median_score = float(np.median(completed_scores))
    return score > median_score


class PeriodicValidationCallback(Callback):
    """Run validation every *interval* epochs and report to Optuna.

    The pruning score is ``sqrt(nrmse * calibration_error)`` (geometric
    mean).  If either metric is missing the callback falls back to
    ``calibration_error`` alone.

    For single-objective studies the score is reported via
    ``trial.report()`` and pruning uses the study's pruner.  For
    multi-objective studies (where ``trial.report()`` is unsupported),
    a custom median-based strategy prunes trials whose intermediate
    score exceeds the median of completed trials at the same step.

    Parameters
    ----------
    trial
        Current Optuna trial.
    approximator
        Trained approximator with a ``.sample()`` method (updated
        in-place during training).
    validation_data
        Pre-generated :class:`~bayesflow_hpo.validation.data.ValidationDataset`.
    param_keys, data_keys
        Keys expected by the inference function.
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
        activates.  Default 5.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        approximator: Any,
        validation_data: Any,
        param_keys: list[str],
        data_keys: list[str],
        interval: int = 10,
        warmup: int = 10,
        n_posterior_samples: int = 250,
        n_startup_trials: int = 5,
    ):
        super().__init__()
        self.trial = trial
        self.approximator = approximator
        self.validation_data = validation_data
        self.param_keys = param_keys
        self.data_keys = data_keys
        self.interval = interval
        self.warmup = warmup
        self.n_posterior_samples = n_posterior_samples
        self.n_startup_trials = n_startup_trials
        self._step = 0  # monotonic step counter for Optuna
        self._is_multi_objective = len(trial.study.directions) > 1

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        if epoch < self.warmup:
            return
        if (epoch - self.warmup) % self.interval != 0:
            return

        score = self._run_lightweight_validation()
        if score is None:
            return

        self._step += 1

        if self._is_multi_objective:
            self.trial.set_user_attr(
                f"val_score_step_{self._step}", round(float(score), 6)
            )
            if _should_prune_multi_objective(
                self.trial, score, self._step, self.n_startup_trials
            ):
                raise optuna.TrialPruned()
        else:
            self.trial.report(float(score), step=self._step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_lightweight_validation(self) -> float | None:
        """Compute geometric mean of nrmse and calibration_error."""
        try:
            from bayesflow_hpo.validation.pipeline import run_validation_pipeline

            result = run_validation_pipeline(
                approximator=self.approximator,
                validation_data=self.validation_data,
                n_posterior_samples=self.n_posterior_samples,
                metrics=_INTERMEDIATE_METRICS,
            )

            cal_err = result.summary.get("calibration_error")
            nrmse = result.summary.get("nrmse")

            if cal_err is not None and nrmse is not None:
                # Geometric mean — both should be positive.
                return float(np.sqrt(max(cal_err, 1e-12) * max(nrmse, 1e-12)))
            if cal_err is not None:
                return float(cal_err)
            return None
        except Exception:
            logger.debug(
                "Intermediate validation failed (trial %d)",
                self.trial.number,
                exc_info=True,
            )
            return None
