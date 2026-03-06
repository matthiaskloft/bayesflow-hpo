"""Periodic validation callback for mid-training pruning.

Runs a lightweight validation (calibration_error + nrmse only) every
*interval* epochs and reports the geometric mean to Optuna for pruning
decisions.  This allows the MedianPruner to compare trials on a
metric that is comparable across different network types (unlike the
training loss, which differs between CouplingFlow and FlowMatching).
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


class PeriodicValidationCallback(Callback):
    """Run validation every *interval* epochs and report to Optuna.

    The pruning score is ``sqrt(nrmse * calibration_error)`` (geometric
    mean).  If either metric is missing the callback falls back to
    ``calibration_error`` alone.

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
        self._step = 0  # monotonic step counter for Optuna

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        if epoch < self.warmup:
            return
        if (epoch - self.warmup) % self.interval != 0:
            return

        score = self._run_lightweight_validation()
        if score is None:
            return

        self._step += 1
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
