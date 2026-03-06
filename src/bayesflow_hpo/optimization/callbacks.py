"""Optuna/Keras callback helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from keras.callbacks import Callback


class OptunaReportCallback(Callback):
    """Report monitored Keras metric to Optuna for logging.

    This callback records the training loss to the Optuna trial's
    intermediate values.  It does **not** trigger pruning — pruning is
    handled by :class:`~bayesflow_hpo.optimization.validation_callback.PeriodicValidationCallback`
    which uses validation metrics that are comparable across network types.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "loss",
        report_frequency: int = 10,
    ):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.report_frequency = report_frequency

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        if logs is None or epoch % self.report_frequency != 0:
            return

        value = logs.get(self.monitor)
        if value is None:
            return

        self.trial.set_user_attr(f"epoch_{epoch}_loss", round(float(value), 6))


class MovingAverageEarlyStopping(Callback):
    """Early stopping on moving average of a monitored metric.

    Parameters
    ----------
    monitor
        Metric name to track (default ``"loss"``).
    window
        Moving-average window size (default 7).
    patience
        Number of epochs with no MA improvement before stopping
        (default 5).  Together with *window* this means stagnation is
        detected within roughly ``window + patience`` epochs.
    restore_best_weights
        Whether to restore model weights from the epoch with the best
        moving average (default ``True``).
    """

    def __init__(
        self,
        monitor: str = "loss",
        window: int = 7,
        patience: int = 5,
        restore_best_weights: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.window = window
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self._losses: list[float] = []
        self.wait = 0
        self.best_ma_loss = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return

        self._losses.append(float(value))
        if len(self._losses) > self.window:
            self._losses.pop(0)

        moving_avg = float(np.mean(self._losses))
        logs[f"moving_avg_{self.monitor}"] = moving_avg

        if moving_avg < self.best_ma_loss:
            self.best_ma_loss = moving_avg
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return

        self.wait += 1
        if self.wait >= self.patience:
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
