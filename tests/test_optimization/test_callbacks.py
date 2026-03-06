"""Tests for optimization callbacks."""

from types import SimpleNamespace

from bayesflow_hpo.optimization.callbacks import OptunaReportCallback


class _TrialStub:
    def __init__(self, directions):
        self.study = SimpleNamespace(directions=directions)
        self.report_calls = []
        self.should_prune_calls = 0

    def report(self, value, step):
        self.report_calls.append((value, step))

    def should_prune(self):
        self.should_prune_calls += 1
        return False


def test_optuna_report_callback_reports_for_single_objective():
    trial = _TrialStub(directions=["minimize"])
    callback = OptunaReportCallback(trial=trial, monitor="loss")

    callback.on_epoch_end(epoch=0, logs={"loss": 0.123})

    assert trial.report_calls == [(0.123, 0)]
    assert trial.should_prune_calls == 1


def test_optuna_report_callback_skips_multi_objective_trials():
    trial = _TrialStub(directions=["minimize", "minimize"])
    callback = OptunaReportCallback(trial=trial, monitor="loss")

    callback.on_epoch_end(epoch=0, logs={"loss": 0.123})

    assert trial.report_calls == []
    assert trial.should_prune_calls == 0
