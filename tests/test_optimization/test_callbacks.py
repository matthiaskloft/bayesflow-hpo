"""Tests for optimization callbacks."""

from types import SimpleNamespace

from bayesflow_hpo.optimization.callbacks import OptunaReportCallback


class _TrialStub:
    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def test_optuna_report_callback_reports_for_single_objective():
    trial = _TrialStub()
    callback = OptunaReportCallback(trial=trial, monitor="loss")

    callback.on_epoch_end(epoch=0, logs={"loss": 0.123})

    assert trial.user_attrs == {"epoch_0_loss": 0.123}


def test_optuna_report_callback_skips_multi_objective_trials():
    """OptunaReportCallback now logs for all trial types (no pruning)."""
    trial = _TrialStub()
    callback = OptunaReportCallback(trial=trial, monitor="loss")

    callback.on_epoch_end(epoch=0, logs={"loss": 0.456})

    assert trial.user_attrs == {"epoch_0_loss": 0.456}
