"""Tests for pruner and sampler string presets in create_study()."""

from __future__ import annotations

from unittest.mock import patch

import optuna
import pytest

from bayesflow_hpo.optimization.study import (
    _count_non_rejected,
    _make_constraints_func,
    _resolve_n_startup_trials,
    _resolve_pruner,
    _resolve_sampler,
    create_study,
)


class TestResolvePruner:
    """Tests for _resolve_pruner() string preset resolution."""

    def test_median_creates_median_pruner(self):
        pruner = _resolve_pruner("median")
        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_hyperband_creates_hyperband_pruner(self):
        pruner = _resolve_pruner("hyperband")
        assert isinstance(pruner, optuna.pruners.HyperbandPruner)

    def test_none_creates_nop_pruner(self):
        pruner = _resolve_pruner("none")
        assert isinstance(pruner, optuna.pruners.NopPruner)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unknown pruner preset"):
            _resolve_pruner("invalid")


class TestResolveSampler:
    """Tests for _resolve_sampler() string preset resolution."""

    def test_tpe_creates_tpe_sampler(self):
        sampler = _resolve_sampler("tpe")
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_tpe_has_correct_defaults(self):
        sampler = _resolve_sampler("tpe")
        assert sampler._multivariate is True
        assert sampler._n_startup_trials == 25
        assert sampler._warn_independent_sampling is False
        assert sampler._constraints_func(_mk_frozen_trial()) == [0.0]

    def test_gp_creates_gp_sampler(self):
        sampler = _resolve_sampler("gp")
        assert isinstance(sampler, optuna.samplers.GPSampler)

    def test_gp_has_correct_defaults(self):
        sampler = _resolve_sampler("gp")
        # GPSampler stores n_startup_trials as private _n_startup_trials.
        assert getattr(sampler, "_n_startup_trials", None) == 10

    def test_nsga2_creates_nsga2_sampler(self):
        sampler = _resolve_sampler("nsga2")
        assert isinstance(sampler, optuna.samplers.NSGAIISampler)

    def test_nsga2_has_correct_defaults(self):
        sampler = _resolve_sampler("nsga2")
        assert sampler.population_size == 50

    def test_nsga3_creates_nsga3_sampler(self):
        sampler = _resolve_sampler("nsga3")
        assert isinstance(sampler, optuna.samplers.NSGAIIISampler)

    def test_nsga3_has_correct_defaults(self):
        sampler = _resolve_sampler("nsga3")
        assert sampler.population_size == 50

    def test_random_creates_random_sampler(self):
        sampler = _resolve_sampler("random")
        assert isinstance(sampler, optuna.samplers.RandomSampler)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unknown sampler preset"):
            _resolve_sampler("invalid")

    def test_budget_aware_true_wires_constraints(self):
        sampler = _resolve_sampler("tpe", budget_aware=True)
        assert sampler._constraints_func(_mk_frozen_trial()) == [0.0]

    def test_budget_aware_false_no_constraints(self):
        sampler = _resolve_sampler("tpe", budget_aware=False)
        assert sampler._constraints_func is None

    def test_nsga2_budget_aware_true_wires_constraints(self):
        sampler = _resolve_sampler("nsga2", budget_aware=True)
        assert sampler._constraints_func(_mk_frozen_trial()) == [0.0]

    def test_nsga2_budget_aware_false_no_constraints(self):
        sampler = _resolve_sampler("nsga2", budget_aware=False)
        assert sampler._constraints_func is None

    def test_random_has_no_constraints_func(self):
        """RandomSampler doesn't accept constraints_func."""
        sampler = _resolve_sampler("random", budget_aware=True)
        assert not hasattr(sampler, "_constraints_func")

    def test_botorch_missing_raises_import_error(self):
        with patch.dict("sys.modules", {"optuna.integration": None}):
            with pytest.raises(ImportError, match="optuna-integration\\[botorch\\]"):
                _resolve_sampler("botorch")

    def test_auto_missing_raises_import_error(self):
        """AutoSampler is not available in Optuna 4.7.0."""
        with pytest.raises(ImportError, match="AutoSampler"):
            _resolve_sampler("auto")


class TestCreateStudySampler:
    """Tests for sampler parameter in create_study()."""

    def test_default_creates_tpe(self):
        """sampler=None should create default TPESampler."""
        study = create_study(storage=None)
        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_string_tpe(self):
        study = create_study(sampler="tpe", storage=None)
        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_string_gp(self):
        study = create_study(sampler="gp", storage=None)
        assert isinstance(study.sampler, optuna.samplers.GPSampler)

    def test_string_nsga2(self):
        study = create_study(sampler="nsga2", storage=None)
        assert isinstance(study.sampler, optuna.samplers.NSGAIISampler)

    def test_string_nsga3(self):
        study = create_study(sampler="nsga3", storage=None)
        assert isinstance(study.sampler, optuna.samplers.NSGAIIISampler)

    def test_string_random(self):
        study = create_study(sampler="random", storage=None)
        assert isinstance(study.sampler, optuna.samplers.RandomSampler)

    def test_custom_sampler_passed_through(self):
        """Custom BaseSampler instance should be passed through unchanged."""
        custom = optuna.samplers.RandomSampler(seed=123)
        study = create_study(sampler=custom, storage=None)
        assert study.sampler is custom

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unknown sampler preset"):
            create_study(sampler="invalid", storage=None)

    def test_budget_aware_false_creates_sampler_without_constraints(self):
        study = create_study(sampler="tpe", budget_aware=False, storage=None)
        assert study.sampler._constraints_func is None


class TestCreateStudyPruner:
    """Tests for pruner parameter in create_study()."""

    def test_default_creates_median_pruner(self):
        """pruner=None should create default MedianPruner."""
        study = create_study(storage=None)
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_string_median(self):
        study = create_study(pruner="median", storage=None)
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_string_hyperband(self):
        study = create_study(pruner="hyperband", storage=None)
        assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)

    def test_string_none(self):
        study = create_study(pruner="none", storage=None)
        assert isinstance(study.pruner, optuna.pruners.NopPruner)

    def test_custom_pruner_passed_through(self):
        """Custom BasePruner instance should be passed through unchanged."""
        custom = optuna.pruners.PercentilePruner(percentile=50.0)
        study = create_study(pruner=custom, storage=None)
        assert study.pruner is custom

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unknown pruner preset"):
            create_study(pruner="invalid", storage=None)


class TestResolveNStartupTrials:
    """Tests for _resolve_n_startup_trials()."""

    def test_tpe_sampler_returns_25(self):
        sampler = optuna.samplers.TPESampler(n_startup_trials=25)
        assert _resolve_n_startup_trials(sampler) == 25

    def test_nsga2_returns_population_size(self):
        sampler = optuna.samplers.NSGAIISampler(population_size=100)
        assert _resolve_n_startup_trials(sampler) == 100

    def test_nsga3_returns_population_size(self):
        sampler = optuna.samplers.NSGAIIISampler(population_size=75)
        assert _resolve_n_startup_trials(sampler) == 75

    def test_random_sampler_fallback_10(self):
        sampler = optuna.samplers.RandomSampler(seed=42)
        assert _resolve_n_startup_trials(sampler) == 10

    def test_gp_sampler_returns_10_via_private_attr(self):
        """GPSampler(n_startup_trials=10) stores it as _n_startup_trials."""
        sampler = optuna.samplers.GPSampler(seed=42, n_startup_trials=10)
        assert _resolve_n_startup_trials(sampler) == 10


def _mk_frozen_trial(
    *,
    user_attrs: dict | None = None,
) -> optuna.trial.FrozenTrial:
    return optuna.trial.create_trial(
        params={"x": 0.5},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 1.0)},
        values=(0.5, 0.5),
        user_attrs=user_attrs or {},
    )


def test_make_constraints_func_budget_only():
    fn = _make_constraints_func(budget_aware=True, soft_thresholds=None)
    assert fn(_mk_frozen_trial()) == [0.0]
    assert fn(_mk_frozen_trial(user_attrs={"rejected_reason": "param_budget"})) == [1.0]


def test_make_constraints_func_soft_violation_above():
    fn = _make_constraints_func(
        budget_aware=False,
        soft_thresholds=[("calibration_error", 0.2, "above")],
    )
    vals = fn(_mk_frozen_trial(user_attrs={"calibration_error": 0.3}))
    assert vals[0] == pytest.approx(0.1)


def test_make_constraints_func_soft_violation_below():
    fn = _make_constraints_func(
        budget_aware=False,
        soft_thresholds=[("coverage", 0.8, "below")],
    )
    vals = fn(_mk_frozen_trial(user_attrs={"coverage": 0.7}))
    assert vals[0] == pytest.approx(0.1)


def test_make_constraints_func_missing_metric_is_feasible():
    fn = _make_constraints_func(
        budget_aware=False,
        soft_thresholds=[("sbc_ks", 0.1, "above")],
    )
    assert fn(_mk_frozen_trial()) == [0.0]


def test_make_constraints_func_combined_budget_and_soft():
    fn = _make_constraints_func(
        budget_aware=True,
        soft_thresholds=[("calibration_error", 0.2, "above")],
    )
    vals = fn(
        _mk_frozen_trial(
            user_attrs={
                "rejected_reason": "memory_budget",
                "calibration_error": 0.3,
            }
        )
    )
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(0.1)


def test_resolve_sampler_budget_false_without_soft_has_no_constraints():
    sampler = _resolve_sampler("tpe", budget_aware=False, soft_thresholds=None)
    assert sampler._constraints_func is None


def test_resolve_sampler_budget_false_with_soft_has_constraints():
    sampler = _resolve_sampler(
        "tpe",
        budget_aware=False,
        soft_thresholds=[("calibration_error", 0.2, "above")],
    )
    fn = sampler._constraints_func
    vals = fn(_mk_frozen_trial(user_attrs={"calibration_error": 0.3}))
    assert vals[0] == pytest.approx(0.1)


def test_count_non_rejected_includes_metric_rejected_trials():
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.add_trial(_mk_frozen_trial(user_attrs={"rejected_reason": "param_budget"}))
    study.add_trial(
        _mk_frozen_trial(user_attrs={"rejected_reason": "metric_constraint"})
    )
    study.add_trial(_mk_frozen_trial())
    assert _count_non_rejected(study) == 2
