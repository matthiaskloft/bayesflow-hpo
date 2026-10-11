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


# ---------------------------------------------------------------------------
# Conditional (define-by-run) search spaces
# ---------------------------------------------------------------------------


def _conditional_objective(trial: optuna.Trial) -> float:
    family = trial.suggest_categorical("family", ["x", "y"])
    if family == "x":
        a = trial.suggest_float("x_a", 0.0, 1.0)
        b = trial.suggest_float("x_b", 0.0, 1.0)
        return (a - 0.3) ** 2 + (b - 0.6) ** 2
    c = trial.suggest_float("y_c", 0.0, 1.0)
    return (c - 0.8) ** 2


def _unconditional_objective(trial: optuna.Trial) -> float:
    a = trial.suggest_float("a", 0.0, 1.0)
    b = trial.suggest_float("b", 1e-3, 1.0, log=True)
    k = trial.suggest_categorical("k", ["u", "v", "w"])
    return (a - 0.3) ** 2 + (b - 0.1) ** 2 + (k == "v") * 0.1


class TestConditionalSpaces:
    def test_tpe_preset_sets_group_and_multivariate(self):
        from bayesflow_hpo.optimization.study import _resolve_sampler

        sampler = _resolve_sampler("tpe")
        assert sampler._multivariate is True
        assert sampler._group is True

    def test_tpe_runs_on_conditional_space(self):
        from bayesflow_hpo.optimization.study import _resolve_sampler

        sampler = _resolve_sampler("tpe", n_startup_trials=6)
        study = optuna.create_study(storage=None, sampler=sampler)
        study.optimize(_conditional_objective, n_trials=30)
        fams = {t.params["family"] for t in study.trials}
        assert fams == {"x", "y"}
        # After startup, family-specific dims are sampled (no exception, in range).
        for t in study.trials:
            for v in t.params.values():
                assert v in ("x", "y") or 0.0 <= v <= 1.0

    def test_group_is_noop_on_unconditional_space(self):
        def run(group: bool) -> list[dict]:
            sampler = optuna.samplers.TPESampler(
                seed=42,
                multivariate=True,
                group=group,
                n_startup_trials=5,
                warn_independent_sampling=False,
            )
            study = optuna.create_study(storage=None, sampler=sampler)
            study.optimize(_unconditional_objective, n_trials=25)
            return [t.params for t in study.trials]

        assert run(True) == run(False)

    def test_warmup_covers_both_families(self):
        main = optuna.samplers.RandomSampler(seed=0)
        sampler = QMCWarmupSampler(main, 8)
        study = optuna.create_study(storage=None, sampler=sampler)
        study.optimize(_conditional_objective, n_trials=8)
        fams = [t.params["family"] for t in study.trials]
        # Balanced family choice: 4 + 4.
        assert fams.count("x") == 4
        assert fams.count("y") == 4
        for t in study.trials:
            names = set(t.params) - {"family"}
            assert names == (
                {"x_a", "x_b"} if t.params["family"] == "x" else {"y_c"}
            )

    def test_balanced_with_many_choices(self):
        # Optuna >= 5.0 samples categoricals in QMCSampler's relative space;
        # with two choices Sobol happens to balance them, with six it does not.
        def obj(trial: optuna.Trial) -> float:
            trial.suggest_categorical("family", list("abcdef"))
            trial.suggest_float("u", 0.0, 1.0)
            return 0.0

        sampler = QMCWarmupSampler(optuna.samplers.RandomSampler(seed=0), 8)
        study = optuna.create_study(storage=None, sampler=sampler)
        study.optimize(obj, n_trials=8)
        fams = [t.params["family"] for t in study.trials]
        counts = [fams.count(c) for c in "abcdef"]
        assert max(counts) - min(counts) <= 1

    def test_same_seed_reproduces_warmup(self):
        def run() -> list[dict]:
            sampler = QMCWarmupSampler(
                optuna.samplers.RandomSampler(seed=0), 8, seed=7
            )
            study = optuna.create_study(storage=None, sampler=sampler)
            study.optimize(_conditional_objective, n_trials=8)
            return [t.params for t in study.trials]

        assert run() == run()

    def test_balanced_nested_categorical_within_family(self):
        def obj(trial: optuna.Trial) -> float:
            fam = trial.suggest_categorical("family", ["x", "y"])
            if fam == "x":
                trial.suggest_categorical("act", ["relu", "gelu"])
            return 0.0

        sampler = QMCWarmupSampler(optuna.samplers.RandomSampler(seed=0), 8)
        study = optuna.create_study(storage=None, sampler=sampler)
        study.optimize(obj, n_trials=8)
        acts = [t.params["act"] for t in study.trials if "act" in t.params]
        assert acts.count("relu") == acts.count("gelu") == 2

    def test_conditional_dim_outside_sobol_space_is_logged_once(self, caplog):
        sampler = QMCWarmupSampler(optuna.samplers.RandomSampler(seed=0), 8)
        study = optuna.create_study(storage=None, sampler=sampler)
        with caplog.at_level(logging.WARNING, logger="bayesflow_hpo"):
            study.optimize(_conditional_objective, n_trials=8)
        texts = [r.getMessage() for r in caplog.records]
        msgs = [m for m in texts if "QMC warm-up" in m]
        # First trial is family x (balanced tie-break aside); the other
        # family's dimensions are the conditional ones, each logged once.
        assert len(msgs) == len(set(msgs))
        assert msgs

    def test_unconditional_space_unchanged_dimensions_stay_sobol(self, caplog):
        sampler = QMCWarmupSampler(optuna.samplers.RandomSampler(seed=0), 8)
        study = optuna.create_study(storage=None, sampler=sampler)
        with caplog.at_level(logging.WARNING, logger="bayesflow_hpo"):
            study.optimize(_unconditional_objective, n_trials=8)
        assert not [r for r in caplog.records if "QMC warm-up" in r.getMessage()]
        ks = [t.params["k"] for t in study.trials]
        assert max(ks.count(c) for c in "uvw") - min(ks.count(c) for c in "uvw") <= 1

    def test_create_study_end_to_end_conditional(self):
        study = create_study(
            storage=None,
            directions=["minimize"],
            metric_names=["loss"],
            qmc_startup_trials=4,
            sampler_n_startup_trials=4,
            has_cost=False,
        )
        study.optimize(_conditional_objective, n_trials=12)
        assert {t.params["family"] for t in study.trials} == {"x", "y"}
