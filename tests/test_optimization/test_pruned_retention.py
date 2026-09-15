"""Retention of pruned trials' weights, objective side (bayesflow-hpo#106).

``CheckpointPool.maybe_save`` runs at Step 10 of the objective, which a
pruned trial never reaches.  These tests cover the two pieces that make
the opt-in path work: recovering the rung a trial was pruned at, and
offering its weights to the pruned pool before ``cleanup_trial()`` frees
the approximator.
"""

from unittest.mock import MagicMock, patch

import optuna

from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool
from bayesflow_hpo.optimization.objective import GenericObjective, _pruning_rung
from bayesflow_hpo.optimization.validation_callback import (
    PeriodicValidationCallback,
)
from bayesflow_hpo.validation.data import ValidationDataset

_DUMMY_VALIDATION_DATA = ValidationDataset(
    simulations=[],
    condition_labels=[],
    param_keys=["p"],
    data_keys=["x"],
    seed=0,
)


def _make_callback(trial) -> PeriodicValidationCallback:
    return PeriodicValidationCallback(
        trial=trial,
        approximator=None,
        validation_data=_DUMMY_VALIDATION_DATA,
        interval=1,
        warmup=0,
        pruning_strategy="none",
        objective_metrics=["calibration_error", "nrmse"],
    )


class TestPruningRung:
    def test_no_callback_yields_no_rung(self):
        assert _pruning_rung([MagicMock()]) == (None, None)

    def test_before_first_validation_yields_no_rung(self):
        """A trial pruned before warmup has no rung to record."""
        study = optuna.create_study(directions=["minimize"] * 2)
        cb = _make_callback(study.ask())
        assert cb.validation_step == 0
        assert _pruning_rung([cb]) == (None, None)

    def test_reports_step_and_mean_score(self):
        study = optuna.create_study(directions=["minimize"] * 2)
        cb = _make_callback(study.ask())
        with patch.object(
            cb,
            "_run_lightweight_validation",
            return_value={"calibration_error": 0.02, "nrmse": 0.04},
        ):
            cb.on_epoch_end(epoch=0)
            cb.on_epoch_end(epoch=1)

        step, score = _pruning_rung([cb])
        assert step == 2
        # Both metrics are lower-is-better, so the converted scores are the
        # raw ones and the mean is exact.
        assert score == 0.03


class TestRetainPruned:
    def _objective(self, pool: CheckpointPool) -> GenericObjective:
        obj = GenericObjective.__new__(GenericObjective)
        obj._checkpoint_pool = pool
        return obj

    def test_noop_when_pruned_pool_disabled(self, tmp_path):
        pool = CheckpointPool(pool_dir=tmp_path / "cp")
        approx = MagicMock()
        self._objective(pool)._retain_pruned(MagicMock(number=1), approx, [])
        approx.save_weights.assert_not_called()

    def test_saves_and_stamps_the_rung(self, tmp_path):
        pool = CheckpointPool(
            pool_dir=tmp_path / "cp", pruned_pool_size=2, seed=0,
        )
        study = optuna.create_study(directions=["minimize"] * 2)
        trial = study.ask()
        cb = _make_callback(trial)
        with patch.object(
            cb,
            "_run_lightweight_validation",
            return_value={"calibration_error": 0.02, "nrmse": 0.04},
        ):
            cb.on_epoch_end(epoch=0)

        self._objective(pool)._retain_pruned(trial, MagicMock(), [cb])

        assert pool.pruned_trial_numbers == [trial.number]
        assert trial.user_attrs["pruned_at_step"] == 1

    def test_save_failure_does_not_escape(self, tmp_path):
        """A retention failure must not change what the study records."""
        pool = CheckpointPool(
            pool_dir=tmp_path / "cp", pruned_pool_size=2, seed=0,
        )
        approx = MagicMock()
        approx.save_weights.side_effect = RuntimeError("disk full")
        self._objective(pool)._retain_pruned(MagicMock(number=3), approx, [])
        assert pool.pruned_trial_numbers == []
