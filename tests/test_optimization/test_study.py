"""Tests for pruner string presets in create_study()."""

from __future__ import annotations

import optuna
import pytest

from bayesflow_hpo.optimization.study import _resolve_pruner, create_study


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
