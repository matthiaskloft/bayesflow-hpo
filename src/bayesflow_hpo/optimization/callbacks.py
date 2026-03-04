"""Optuna/Keras callback helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from keras.callbacks import Callback


class OptunaReportCallback(Callback):
    """Report monitored Keras metric to Optuna and enable pruning."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss", report_frequency: int = 1):
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

        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


class MovingAverageEarlyStopping(Callback):
    """Early stopping on moving average of `val_loss`."""

    def __init__(self, window: int = 5, patience: int = 3, restore_best_weights: bool = True):
        super().__init__()
        self.window = window
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.val_losses: list[float] = []
        self.wait = 0
        self.best_ma_loss = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return

        self.val_losses.append(float(val_loss))
        if len(self.val_losses) > self.window:
            self.val_losses.pop(0)

        moving_avg = float(np.mean(self.val_losses))
        logs["moving_avg_val_loss"] = moving_avg

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
