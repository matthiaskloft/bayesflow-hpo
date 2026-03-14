"""Tests for optimization callbacks."""


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


def test_optuna_report_callback_custom_report_frequency():
    """Only epochs divisible by report_frequency are recorded."""
    trial = _TrialStub()
    callback = OptunaReportCallback(
        trial=trial, monitor="loss", report_frequency=5,
    )

    for epoch in range(12):
        callback.on_epoch_end(epoch=epoch, logs={"loss": float(epoch)})

    # Epochs 0, 5, 10 should be recorded
    assert set(trial.user_attrs.keys()) == {
        "epoch_0_loss", "epoch_5_loss", "epoch_10_loss",
    }


def test_optuna_report_callback_default_frequency_is_10():
    """Default report_frequency=10 skips non-multiples of 10."""
    trial = _TrialStub()
    callback = OptunaReportCallback(trial=trial, monitor="loss")

    assert callback.report_frequency == 10

    for epoch in range(25):
        callback.on_epoch_end(epoch=epoch, logs={"loss": float(epoch)})

    assert set(trial.user_attrs.keys()) == {
        "epoch_0_loss", "epoch_10_loss", "epoch_20_loss",
    }


def test_optuna_report_callback_frequency_one_records_every_epoch():
    """report_frequency=1 records every epoch."""
    trial = _TrialStub()
    callback = OptunaReportCallback(
        trial=trial, monitor="loss", report_frequency=1,
    )

    for epoch in range(5):
        callback.on_epoch_end(epoch=epoch, logs={"loss": float(epoch)})

    assert len(trial.user_attrs) == 5
