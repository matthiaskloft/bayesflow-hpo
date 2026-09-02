"""Tests for PeriodicValidationCallback with pluggable pruning strategies."""

from unittest.mock import patch

import optuna
import pytest
from optuna.trial import TrialState

from bayesflow_hpo.optimization.validation_callback import (
    DEFAULT_PRUNING_N_STARTUP_TRIALS,
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


def _make_study(n_objectives: int = 2) -> optuna.Study:
    """Create an in-memory multi- or single-objective study."""
    return optuna.create_study(directions=["minimize"] * n_objectives)


def _add_completed_trial(
    study: optuna.Study,
    values: list[float],
    user_attrs: dict | None = None,
) -> None:
    """Add a synthetic completed trial to *study*."""
    trial = optuna.trial.create_trial(
        params={},
        distributions={},
        values=values,
        user_attrs=user_attrs or {},
        state=TrialState.COMPLETE,
    )
    study.add_trial(trial)


def _make_metric_attrs(
    metrics: dict[str, float],
    step: int,
) -> dict[str, float]:
    """Build ``val_{metric}_step_{step}`` user attrs from a metric dict."""
    return {f"val_{m}_step_{step}": v for m, v in metrics.items()}


class TestCallbackPerMetricAttrs:
    """Verify per-metric user attribute storage."""

    def test_stores_per_metric_attrs(self):
        """Multi-objective callback should store val_{metric}_step_{step}."""
        study = _make_study()
        for s in [0.01, 0.02, 0.03, 0.04, 0.05]:
            _add_completed_trial(
                study,
                [s, 0.5],
                _make_metric_attrs(
                    {"calibration_error": s, "nrmse": s + 0.01}, step=1
                ),
            )

        trial = study.ask()
        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            n_startup_trials=5,
            objective_metrics=["calibration_error", "nrmse"],
        )

        mock_scores = {"calibration_error": 0.01, "nrmse": 0.02}
        with patch.object(
            cb, "_run_lightweight_validation", return_value=mock_scores
        ):
            cb.on_epoch_end(epoch=0)

        assert "val_calibration_error_step_1" in trial.user_attrs
        assert trial.user_attrs["val_calibration_error_step_1"] == 0.01
        assert "val_nrmse_step_1" in trial.user_attrs
        assert trial.user_attrs["val_nrmse_step_1"] == 0.02

    def test_stores_contraction_in_minimize_orientation(self) -> None:
        """Higher contraction is converted before storage and pruning."""
        study = _make_study()
        trial = study.ask()
        callback = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            pruning_strategy="none",
            objective_metrics=["contraction"],
        )

        with patch.object(
            callback,
            "_run_lightweight_validation",
            return_value={"contraction": 0.8},
        ):
            callback.on_epoch_end(epoch=0)

        assert trial.user_attrs["val_contraction_step_1"] == pytest.approx(0.2)


class _WeightTrackingApproximator:
    def __init__(self):
        self.stop_training = False
        self.weights = [0]

    def get_weights(self):
        return list(self.weights)

    def set_weights(self, weights):
        self.weights = list(weights)


def test_open_ended_early_stopping_uses_validation_metric_and_restores_weights():
    study = _make_study()
    trial = study.ask()
    approximator = _WeightTrackingApproximator()
    callback = PeriodicValidationCallback(
        trial=trial,
        approximator=approximator,
        validation_data=_DUMMY_VALIDATION_DATA,
        interval=1,
        warmup=0,
        pruning_strategy="none",
        objective_metrics=["calibration_error", "nrmse"],
        early_stopping_patience=2,
        early_stopping_window=1,
        early_stopping_monitor="calibration_error",
    )

    scores = iter(
        [
            {"calibration_error": 0.4, "nrmse": 0.1},
            {"calibration_error": 0.3, "nrmse": 0.1},
            {"calibration_error": 0.35, "nrmse": 0.1},
            {"calibration_error": 0.36, "nrmse": 0.1},
        ]
    )
    with patch.object(
        callback,
        "_run_lightweight_validation",
        side_effect=lambda: next(scores),
    ):
        for epoch in range(4):
            approximator.weights = [epoch]
            callback.on_epoch_end(epoch)

    assert approximator.stop_training is True
    assert approximator.weights == [1]
    assert callback.best_validation_score == pytest.approx(0.3)


def test_open_ended_early_stopping_defaults_to_mean_objective():
    study = _make_study()
    trial = study.ask()
    approximator = _WeightTrackingApproximator()
    callback = PeriodicValidationCallback(
        trial=trial,
        approximator=approximator,
        validation_data=_DUMMY_VALIDATION_DATA,
        interval=1,
        warmup=0,
        pruning_strategy="none",
        objective_metrics=["calibration_error", "correlation"],
        early_stopping_patience=1,
    )

    callback._update_early_stopping(
        {"calibration_error": 0.2, "correlation": 0.8}
    )

    # correlation is higher-is-better and therefore contributes 1 - 0.8.
    assert callback.best_validation_score == pytest.approx(0.2)


def test_open_ended_restores_best_weights_at_epoch_cap():
    study = _make_study()
    approximator = _WeightTrackingApproximator()
    callback = PeriodicValidationCallback(
        trial=study.ask(),
        approximator=approximator,
        validation_data=_DUMMY_VALIDATION_DATA,
        objective_metrics=["calibration_error"],
        early_stopping_patience=3,
    )
    approximator.weights = [1]
    callback._update_early_stopping({"calibration_error": 0.2})
    approximator.weights = [2]
    callback._update_early_stopping({"calibration_error": 0.3})

    callback.on_train_end()

    assert approximator.weights == [1]


def test_objective_mean_maximizes_contraction():
    study = _make_study()
    callback = PeriodicValidationCallback(
        trial=study.ask(),
        approximator=_WeightTrackingApproximator(),
        validation_data=_DUMMY_VALIDATION_DATA,
        objective_metrics=["contraction"],
        early_stopping_patience=2,
    )

    callback._update_early_stopping({"contraction": 0.8})

    assert callback.best_validation_score == pytest.approx(0.2)


def test_early_stopping_rejects_unknown_monitor():
    study = _make_study()
    with pytest.raises(ValueError, match="must be 'objective_mean'"):
        PeriodicValidationCallback(
            trial=study.ask(),
            approximator=_WeightTrackingApproximator(),
            validation_data=_DUMMY_VALIDATION_DATA,
            objective_metrics=["calibration_error", "nrmse"],
            early_stopping_monitor="recovery",
        )


class TestCallbackPruning:
    """Verify pruning dispatch."""

    def test_dominance_raises_trial_pruned(self):
        """Dominance strategy should raise TrialPruned for bad trial."""
        study = _make_study()
        for s in [0.01, 0.02, 0.03, 0.04, 0.05]:
            _add_completed_trial(
                study,
                [s, 0.5],
                _make_metric_attrs(
                    {"calibration_error": s, "nrmse": s}, step=1
                ),
            )

        trial = study.ask()
        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            n_startup_trials=5,
            pruning_strategy="dominance",
            objective_metrics=["calibration_error", "nrmse"],
        )

        mock_scores = {"calibration_error": 0.99, "nrmse": 0.99}
        with patch.object(
            cb, "_run_lightweight_validation", return_value=mock_scores
        ):
            with pytest.raises(optuna.TrialPruned):
                cb.on_epoch_end(epoch=0)
        # Confirm strategy ran at step 1, matching seeded attrs.
        assert cb._step == 1

    def test_none_strategy_never_prunes(self):
        """Strategy 'none' should never raise TrialPruned."""
        study = _make_study()
        for s in [0.01, 0.02, 0.03, 0.04, 0.05]:
            _add_completed_trial(
                study,
                [s, 0.5],
                _make_metric_attrs(
                    {"calibration_error": s, "nrmse": s}, step=1
                ),
            )

        trial = study.ask()
        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            n_startup_trials=5,
            pruning_strategy="none",
            objective_metrics=["calibration_error", "nrmse"],
        )

        mock_scores = {"calibration_error": 0.99, "nrmse": 0.99}
        with patch.object(
            cb, "_run_lightweight_validation", return_value=mock_scores
        ):
            # Should NOT raise.
            cb.on_epoch_end(epoch=0)


class TestCallbackStrategyDispatch:
    """Verify the correct strategy function is called."""

    def test_dominance_dispatches(self):
        """pruning_strategy='dominance' calls should_prune_dominance."""
        study = _make_study()
        trial = study.ask()
        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            pruning_strategy="dominance",
            objective_metrics=["calibration_error", "nrmse"],
        )

        with (
            patch.object(
                cb,
                "_run_lightweight_validation",
                return_value={"calibration_error": 0.5, "nrmse": 0.5},
            ),
            patch(
                "bayesflow_hpo.optimization.validation_callback"
                ".should_prune_dominance",
                return_value=False,
            ) as mock_fn,
        ):
            cb.on_epoch_end(epoch=0)

        mock_fn.assert_called_once()

    def test_mo_sha_dispatches(self):
        """pruning_strategy='mo-sha' calls should_prune_mo_sha."""
        study = _make_study()
        trial = study.ask()
        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            pruning_strategy="mo-sha",
            objective_metrics=["calibration_error", "nrmse"],
        )

        with (
            patch.object(
                cb,
                "_run_lightweight_validation",
                return_value={"calibration_error": 0.5, "nrmse": 0.5},
            ),
            patch(
                "bayesflow_hpo.optimization.validation_callback"
                ".should_prune_mo_sha",
                return_value=False,
            ) as mock_fn,
        ):
            cb.on_epoch_end(epoch=0)

        mock_fn.assert_called_once()

    def test_primary_dispatches(self):
        """pruning_strategy='primary' calls should_prune_primary."""
        study = _make_study()
        trial = study.ask()
        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            pruning_strategy="primary",
            objective_metrics=["calibration_error", "nrmse"],
        )

        with (
            patch.object(
                cb,
                "_run_lightweight_validation",
                return_value={"calibration_error": 0.5, "nrmse": 0.5},
            ),
            patch(
                "bayesflow_hpo.optimization.validation_callback"
                ".should_prune_primary",
                return_value=False,
            ) as mock_fn,
        ):
            cb.on_epoch_end(epoch=0)

        mock_fn.assert_called_once()

    def test_primary_tuple_dispatches_with_metric(self):
        """Tuple ('primary', 'nrmse') passes correct metric."""
        study = _make_study()
        trial = study.ask()
        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            pruning_strategy=("primary", "nrmse"),
            objective_metrics=["calibration_error", "nrmse"],
        )

        with (
            patch.object(
                cb,
                "_run_lightweight_validation",
                return_value={"calibration_error": 0.5, "nrmse": 0.5},
            ),
            patch(
                "bayesflow_hpo.optimization.validation_callback"
                ".should_prune_primary",
                return_value=False,
            ) as mock_fn,
        ):
            cb.on_epoch_end(epoch=0)

        # Check the metric argument was "nrmse".
        call_args = mock_fn.call_args
        assert call_args[0][2] == "nrmse"  # 3rd positional: metric


class TestSingleObjective:
    """Verify single-objective path."""

    def test_uses_trial_report(self):
        """Single-objective path calls trial.report with first metric."""
        study = optuna.create_study(direction="minimize")
        trial = study.ask()

        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            objective_metrics=["calibration_error"],
        )

        mock_scores = {"calibration_error": 0.5}
        with (
            patch.object(
                cb, "_run_lightweight_validation", return_value=mock_scores
            ),
            patch.object(trial, "report") as mock_report,
        ):
            cb.on_epoch_end(epoch=0)

        mock_report.assert_called_once_with(0.5, step=1)

    def test_empty_scores_does_not_crash(self):
        """Empty dict from validation should not cause KeyError."""
        study = optuna.create_study(direction="minimize")
        trial = study.ask()

        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            objective_metrics=["calibration_error"],
        )

        # Simulate validation returning None (missing metrics path).
        with patch.object(
            cb, "_run_lightweight_validation", return_value=None
        ):
            cb.on_epoch_end(epoch=0)

        # Step should not advance.
        assert cb._step == 0


class TestValidateFnMissingMetrics:
    """Verify missing metric handling from validate_fn."""

    def test_missing_metrics_returns_none(self):
        """validate_fn missing required metrics → skip pruning step."""
        study = _make_study()
        trial = study.ask()

        def bad_validate_fn(approx, data, n):
            return {"calibration_error": 0.5}  # Missing "nrmse".

        cb = PeriodicValidationCallback(
            trial=trial,
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            validate_fn=bad_validate_fn,
            objective_metrics=["calibration_error", "nrmse"],
        )

        # Should not crash — returns None → no pruning.
        cb.on_epoch_end(epoch=0)
        # Step counter should not have advanced.
        assert cb._step == 0


class TestStrategyValidation:
    """Verify invalid strategy names are rejected."""

    def test_invalid_strategy_raises(self):
        """Unknown strategy string should raise ValueError."""
        study = _make_study()
        trial = study.ask()
        with pytest.raises(ValueError, match="Unknown pruning_strategy"):
            PeriodicValidationCallback(
                trial=trial,
                approximator=None,
                validation_data=_DUMMY_VALIDATION_DATA,
                pruning_strategy="invalid",
            )

    def test_invalid_tuple_raises(self):
        """Invalid tuple form should raise ValueError."""
        study = _make_study()
        trial = study.ask()
        with pytest.raises(ValueError, match="Tuple pruning_strategy"):
            PeriodicValidationCallback(
                trial=trial,
                approximator=None,
                validation_data=_DUMMY_VALIDATION_DATA,
                pruning_strategy=("dominance", "metric"),
            )


class TestStartupTrialsNeverNone:
    """``n_startup_trials`` must reach the pruning strategies as an int.

    ``ObjectiveConfig.pruning_n_startup_trials`` defaults to ``None`` and is
    auto-detected from the sampler by ``optimize()``. Building an objective
    directly skips that resolution, so the ``None`` was passed straight
    through to ``should_prune_dominance``, which compares it with
    ``n_startup_trials < 1`` and raises ``TypeError``. The trial handler turns
    that into a training-failure penalty, so the trial looked merely bad
    rather than broken -- and ``pruning_strategy`` defaults to
    ``"dominance"``, so this was the default path.
    """

    def test_none_resolves_to_the_documented_default(self):
        study = _make_study()
        callback = PeriodicValidationCallback(
            trial=study.ask(),
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            n_startup_trials=None,
        )
        assert callback.n_startup_trials == DEFAULT_PRUNING_N_STARTUP_TRIALS
        assert isinstance(callback.n_startup_trials, int)

    def test_an_explicit_value_still_wins(self):
        study = _make_study()
        callback = PeriodicValidationCallback(
            trial=study.ask(),
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            n_startup_trials=17,
        )
        assert callback.n_startup_trials == 17

    def test_the_pruning_strategies_reject_none(self):
        """Why the resolution has to happen before the strategies see it."""
        from bayesflow_hpo.optimization.pruning_strategies import (
            should_prune_dominance,
        )

        study = _make_study()
        with pytest.raises(TypeError):
            should_prune_dominance(study.ask(), {"m1": 0.1, "m2": 0.2}, 1, None)


class TestTheCallbackCanonicalizesItsOwnNames:
    """`PeriodicValidationCallback` is public and takes metric names directly.

    `ObjectiveConfig.__post_init__` canonicalized on the way in, so the
    callback was protected when built through the pipeline and unprotected
    when built on its own -- which `__all__` invites. An alias then missed
    every summary lookup: pruning reported its metrics as "missing" and
    skipped, and early stopping read a key nothing writes.
    """

    def _callback(self, **kwargs):
        study = _make_study()
        return PeriodicValidationCallback(
            trial=study.ask(),
            approximator=None,
            validation_data=_DUMMY_VALIDATION_DATA,
            interval=1,
            warmup=0,
            n_startup_trials=5,
            **kwargs,
        )

    def test_an_alias_in_objective_metrics_is_resolved(self) -> None:
        cb = self._callback(objective_metrics=["cal_error", "nrmse"])
        assert cb.objective_metrics == ["calibration_error", "nrmse"]

    def test_an_alias_pruning_target_is_resolved(self) -> None:
        cb = self._callback(
            objective_metrics=["cal_error", "nrmse"],
            pruning_strategy=("primary", "cal_error"),
        )
        assert cb._primary_metric == "calibration_error"

    def test_an_alias_pairing_still_constructs(self) -> None:
        """Canonicalizing one field but not the other breaks a valid pairing.

        `objective_metrics=["cal_error"]` with
        `early_stopping_monitor="cal_error"` is a legitimate request. Were the
        list canonicalized before the membership check and the monitor after,
        it would raise -- the ordering bug `__post_init__` already records.
        """
        cb = self._callback(
            objective_metrics=["cal_error", "nrmse"],
            early_stopping_monitor="cal_error",
        )
        assert cb.early_stopping_monitor == "calibration_error"
        assert cb.early_stopping_monitor in cb.objective_metrics

    def test_the_objective_mean_sentinel_is_not_a_metric(self) -> None:
        """It names no metric, so it must pass through untouched."""
        cb = self._callback(
            objective_metrics=["cal_error"],
            early_stopping_monitor="objective_mean",
        )
        assert cb.early_stopping_monitor == "objective_mean"
