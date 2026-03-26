"""Tests for QMC warm-up sampler wrapper and create_study() integration."""

from __future__ import annotations

import logging

import optuna
import pytest
from optuna.trial import TrialState

from bayesflow_hpo.optimization.study import (
    QMCWarmupSampler,
    _is_power_of_two,
    _resolve_n_startup_trials,
    create_study,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Throwaway study used only to satisfy the ``after_trial`` signature.
_DUMMY_STUDY = optuna.create_study(storage=None)


def _make_frozen_trial(
    number: int,
    state: TrialState = TrialState.COMPLETE,
    user_attrs: dict | None = None,
) -> optuna.trial.FrozenTrial:
    """Create a minimal FrozenTrial for testing after_trial()."""
    return optuna.trial.FrozenTrial(
        number=number,
        state=state,
        value=None,
        datetime_start=None,
        datetime_complete=None,
        params={},
        distributions={},
        user_attrs=user_attrs or {},
        system_attrs={},
        intermediate_values={},
        trial_id=number,
        values=None,
    )


# ---------------------------------------------------------------------------
# _is_power_of_two
# ---------------------------------------------------------------------------


class TestIsPowerOfTwo:
    def test_powers_of_two(self):
        for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            assert _is_power_of_two(n), f"{n} should be power of 2"

    def test_non_powers_of_two(self):
        for n in (3, 5, 6, 7, 9, 10, 15, 17, 24, 100):
            assert not _is_power_of_two(n), f"{n} should not be power of 2"

    def test_zero_is_not_power_of_two(self):
        assert not _is_power_of_two(0)

    def test_negative_is_not_power_of_two(self):
        assert not _is_power_of_two(-4)


# ---------------------------------------------------------------------------
# QMCWarmupSampler unit tests
# ---------------------------------------------------------------------------


class TestQMCWarmupSampler:
    def test_negative_raises(self):
        with pytest.raises(ValueError, match="qmc_startup_trials must be >= 0"):
            QMCWarmupSampler(optuna.samplers.RandomSampler(), -1)

    def test_zero_is_valid(self):
        wrapper = QMCWarmupSampler(optuna.samplers.RandomSampler(), 0)
        # With 0 QMC trials, should immediately be past QMC phase
        assert not wrapper._is_qmc_phase

    def test_is_qmc_phase_initially(self):
        wrapper = QMCWarmupSampler(optuna.samplers.RandomSampler(), 4)
        assert wrapper._is_qmc_phase

    def test_active_sampler_is_qmc_during_warmup(self):
        wrapper = QMCWarmupSampler(optuna.samplers.RandomSampler(), 4)
        assert wrapper._active_sampler is wrapper._qmc_sampler

    def test_active_sampler_switches_after_warmup(self):
        main = optuna.samplers.RandomSampler()
        wrapper = QMCWarmupSampler(main, 2)
        # Simulate 2 non-rejected completions
        wrapper._n_qmc_completed = 2
        assert wrapper._active_sampler is main
        assert not wrapper._is_qmc_phase

    def test_delegates_to_qmc_during_warmup(self):
        """sample_independent delegates to QMCSampler during QMC phase."""
        main = optuna.samplers.RandomSampler(seed=1)
        wrapper = QMCWarmupSampler(main, 4)
        study = optuna.create_study(storage=None)
        trial = study.ask()
        dist = optuna.distributions.FloatDistribution(0.0, 1.0)
        # Should not raise — delegates to QMCSampler
        val = wrapper.sample_independent(study, trial, "x", dist)
        assert 0.0 <= val <= 1.0

    def test_switches_to_main_after_warmup(self):
        """After N completions, sample_independent uses main sampler."""
        main = optuna.samplers.RandomSampler(seed=42)
        wrapper = QMCWarmupSampler(main, 1)
        wrapper._n_qmc_completed = 1  # Already past QMC phase
        study = optuna.create_study(storage=None)
        trial = study.ask()
        dist = optuna.distributions.FloatDistribution(0.0, 1.0)
        val = wrapper.sample_independent(study, trial, "x", dist)
        assert 0.0 <= val <= 1.0
        # Trial number should NOT be in pending set (not QMC phase)
        assert trial.number not in wrapper._pending_qmc_trials

    def test_non_rejected_counting(self):
        """Budget-rejected trials don't count toward QMC quota."""
        wrapper = QMCWarmupSampler(optuna.samplers.RandomSampler(), 2)

        # Trial 0: QMC-sampled, completed but rejected
        wrapper._pending_qmc_trials.add(0)
        rejected_trial = _make_frozen_trial(
            0, user_attrs={"rejected_reason": "too large"}
        )
        wrapper.after_trial(
            _DUMMY_STUDY, rejected_trial, TrialState.COMPLETE, [1.0]
        )
        assert wrapper._n_qmc_completed == 0
        assert wrapper._is_qmc_phase

        # Trial 1: QMC-sampled, completed, NOT rejected
        wrapper._pending_qmc_trials.add(1)
        good_trial = _make_frozen_trial(1)
        wrapper.after_trial(
            _DUMMY_STUDY, good_trial, TrialState.COMPLETE, [0.5]
        )
        assert wrapper._n_qmc_completed == 1
        assert wrapper._is_qmc_phase  # Need 2, only got 1

        # Trial 2: QMC-sampled, completed, NOT rejected
        wrapper._pending_qmc_trials.add(2)
        good_trial2 = _make_frozen_trial(2)
        wrapper.after_trial(
            _DUMMY_STUDY, good_trial2, TrialState.COMPLETE, [0.3]
        )
        assert wrapper._n_qmc_completed == 2
        assert not wrapper._is_qmc_phase  # Done!

    def test_after_trial_ignores_non_pending_trials(self):
        """Trials not in pending set are not counted."""
        wrapper = QMCWarmupSampler(optuna.samplers.RandomSampler(), 2)
        trial = _make_frozen_trial(99)
        wrapper.after_trial(_DUMMY_STUDY, trial, TrialState.COMPLETE, [0.5])
        assert wrapper._n_qmc_completed == 0

    def test_after_trial_failed_trial_not_counted(self):
        """Failed QMC trials are removed from pending but not counted."""
        wrapper = QMCWarmupSampler(optuna.samplers.RandomSampler(), 2)
        wrapper._pending_qmc_trials.add(0)
        trial = _make_frozen_trial(0, state=TrialState.FAIL)
        wrapper.after_trial(_DUMMY_STUDY, trial, TrialState.FAIL, None)
        assert wrapper._n_qmc_completed == 0
        assert 0 not in wrapper._pending_qmc_trials

    def test_before_trial_delegates_to_active(self):
        """before_trial delegates to the active sub-sampler."""
        main = optuna.samplers.RandomSampler()
        wrapper = QMCWarmupSampler(main, 4)
        study = optuna.create_study(storage=None)
        trial = study.ask()
        # Should not raise
        wrapper.before_trial(study, trial)

    def test_after_trial_delegates_to_active(self):
        """after_trial delegates to the active sub-sampler."""
        main = optuna.samplers.NSGAIISampler()
        wrapper = QMCWarmupSampler(main, 2)
        study = optuna.create_study(
            directions=["minimize", "minimize"], storage=None
        )
        trial = study.ask()
        # Should not raise — NSGA-II uses after_trial for population mgmt
        wrapper.after_trial(study, trial, TrialState.COMPLETE, [0.5, 0.5])

    def test_n_startup_trials_property(self):
        """n_startup_trials returns max(qmc, main_sampler_startup)."""
        main = optuna.samplers.TPESampler(n_startup_trials=25)
        wrapper = QMCWarmupSampler(main, 8)
        # max(8, 25) = 25
        assert wrapper.n_startup_trials == 25

    def test_n_startup_trials_qmc_larger(self):
        """When QMC count exceeds main startup, QMC count wins."""
        main = optuna.samplers.GPSampler(n_startup_trials=10)
        wrapper = QMCWarmupSampler(main, 32)
        # max(32, 10) = 32
        assert wrapper.n_startup_trials == 32

    def test_infer_relative_search_space_delegates(self):
        """infer_relative_search_space delegates to active sampler."""
        main = optuna.samplers.RandomSampler()
        wrapper = QMCWarmupSampler(main, 4)
        study = optuna.create_study(storage=None)
        trial = study.ask()
        # Should return empty dict for RandomSampler (no relative space)
        space = wrapper.infer_relative_search_space(study, trial)
        assert isinstance(space, dict)

    def test_sample_relative_tracks_pending_during_qmc(self):
        """sample_relative adds trial to pending set during QMC phase."""
        wrapper = QMCWarmupSampler(optuna.samplers.RandomSampler(), 4)
        study = optuna.create_study(storage=None)
        trial = study.ask()
        wrapper.sample_relative(study, trial, {})
        assert trial.number in wrapper._pending_qmc_trials

    def test_sample_relative_does_not_track_after_qmc(self):
        """sample_relative does NOT add to pending after QMC phase."""
        wrapper = QMCWarmupSampler(optuna.samplers.RandomSampler(), 1)
        wrapper._n_qmc_completed = 1  # Past QMC phase
        study = optuna.create_study(storage=None)
        trial = study.ask()
        wrapper.sample_relative(study, trial, {})
        assert trial.number not in wrapper._pending_qmc_trials

    def test_phase_transition_boundary(self):
        """The trial that completes the QMC quota transitions correctly."""
        wrapper = QMCWarmupSampler(optuna.samplers.RandomSampler(), 1)
        assert wrapper._is_qmc_phase

        # Simulate one QMC trial completing
        wrapper._pending_qmc_trials.add(0)
        trial = _make_frozen_trial(0)
        wrapper.after_trial(_DUMMY_STUDY, trial, TrialState.COMPLETE, [0.5])

        # Now past QMC phase
        assert not wrapper._is_qmc_phase
        assert wrapper._active_sampler is wrapper._main_sampler

    def test_n_startup_trials_with_random_sampler(self):
        """Fallback path: RandomSampler has no n_startup_trials attr."""
        main = optuna.samplers.RandomSampler()
        wrapper = QMCWarmupSampler(main, 16)
        # RandomSampler falls back to 10; max(16, 10) = 16
        assert wrapper.n_startup_trials == 16


# ---------------------------------------------------------------------------
# create_study() QMC integration
# ---------------------------------------------------------------------------


class TestCreateStudyQMC:
    def test_qmc_zero_no_wrapper(self):
        """Default (0) produces no wrapper, existing sampler type preserved."""
        study = create_study(qmc_startup_trials=0, storage=None)
        assert isinstance(study.sampler, optuna.samplers.TPESampler)
        assert not isinstance(study.sampler, QMCWarmupSampler)

    def test_qmc_positive_wraps_sampler(self):
        """qmc_startup_trials=8 wraps the sampler in QMCWarmupSampler."""
        study = create_study(qmc_startup_trials=8, storage=None)
        assert isinstance(study.sampler, QMCWarmupSampler)
        assert isinstance(study.sampler._main_sampler, optuna.samplers.TPESampler)
        assert study.sampler._qmc_startup_trials == 8

    def test_qmc_with_string_preset_tpe(self):
        study = create_study(sampler="tpe", qmc_startup_trials=16, storage=None)
        assert isinstance(study.sampler, QMCWarmupSampler)
        assert isinstance(study.sampler._main_sampler, optuna.samplers.TPESampler)

    def test_qmc_with_string_preset_gp(self):
        study = create_study(sampler="gp", qmc_startup_trials=8, storage=None)
        assert isinstance(study.sampler, QMCWarmupSampler)
        assert isinstance(study.sampler._main_sampler, optuna.samplers.GPSampler)

    def test_qmc_with_string_preset_nsga2(self):
        study = create_study(sampler="nsga2", qmc_startup_trials=16, storage=None)
        assert isinstance(study.sampler, QMCWarmupSampler)
        assert isinstance(study.sampler._main_sampler, optuna.samplers.NSGAIISampler)

    def test_qmc_with_custom_sampler(self):
        """Wraps a custom BaseSampler instance."""
        custom = optuna.samplers.RandomSampler(seed=123)
        study = create_study(
            sampler=custom, qmc_startup_trials=4, storage=None
        )
        assert isinstance(study.sampler, QMCWarmupSampler)
        assert study.sampler._main_sampler is custom

    def test_qmc_negative_raises(self):
        with pytest.raises(ValueError, match="qmc_startup_trials must be >= 0"):
            create_study(qmc_startup_trials=-1, storage=None)

    def test_qmc_power_of_two_no_warning(self, caplog):
        """No warning for power-of-2 values like 8, 16, 32."""
        with caplog.at_level(logging.WARNING):
            create_study(qmc_startup_trials=16, storage=None)
        assert "not a power of 2" not in caplog.text

    def test_qmc_non_power_of_two_warning(self, caplog):
        """Logs warning for non-power-of-2 values."""
        with caplog.at_level(logging.WARNING):
            create_study(qmc_startup_trials=10, storage=None)
        assert "not a power of 2" in caplog.text


# ---------------------------------------------------------------------------
# _resolve_n_startup_trials with QMCWarmupSampler
# ---------------------------------------------------------------------------


class TestResolveNStartupTrialsQMC:
    def test_qmc_wrapper_returns_n_startup_trials(self):
        """_resolve_n_startup_trials works with QMCWarmupSampler."""
        main = optuna.samplers.TPESampler(n_startup_trials=25)
        wrapper = QMCWarmupSampler(main, 16)
        # The wrapper exposes n_startup_trials as a public property,
        # so _resolve_n_startup_trials should find it.
        result = _resolve_n_startup_trials(wrapper)
        assert result == 25  # max(16, 25)

    def test_qmc_wrapper_with_small_main_startup(self):
        main = optuna.samplers.GPSampler(n_startup_trials=10)
        wrapper = QMCWarmupSampler(main, 32)
        result = _resolve_n_startup_trials(wrapper)
        assert result == 32  # max(32, 10)
