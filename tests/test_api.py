"""Tests for the high-level optimize() API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from conftest import FakeRename, canonical_adapter, make_adapter

from bayesflow_hpo.api import (
    _build_objective,
    _create_and_run_study,
    _derive_directions,
    _infer_and_validate_keys,
    _setup_validation_data,
    optimize,
)


def _make_fake_search_space():
    """Return a mock search space for tests."""
    space = MagicMock()
    space.inference_space = MagicMock()
    space.summary_space = None
    return space


def _patched_optimize(adapter=None, **extra_kwargs):
    """Call optimize() with heavy internals mocked out."""
    if adapter is None:
        adapter = canonical_adapter()

    with (
        patch("bayesflow_hpo.api.GenericObjective") as mock_obj_cls,
        patch("bayesflow_hpo.api.create_study"),
        patch("bayesflow_hpo.api.optimize_until"),
        patch("bayesflow_hpo.api.check_pipeline"),
        patch("bayesflow_hpo.api.generate_validation_dataset") as mock_gen,
    ):
        mock_instance = MagicMock()
        mock_instance.n_objectives = 3  # pareto default: 2 metrics + cost
        mock_obj_cls.return_value = mock_instance

        # Return a mock validation dataset
        mock_gen.return_value = MagicMock()

        kwargs = {"storage": None, "search_space": _make_fake_search_space()}
        kwargs.update(extra_kwargs)

        optimize(
            simulator=MagicMock(),
            adapter=adapter,
            **kwargs,
        )

        # ObjectiveConfig passed to GenericObjective
        return mock_obj_cls.call_args[0][0]


# ---------------------------------------------------------------------------
# Keys derived from adapter
# ---------------------------------------------------------------------------


def test_keys_derived_from_adapter():
    """param_keys and data_keys are derived from adapter transforms."""
    # Keys are inferred internally, not stored on config anymore.
    # The test verifies optimize() doesn't raise with a canonical adapter.
    _patched_optimize()


def test_missing_param_keys_raises_type_error():
    """TypeError when adapter lacks inference_variables."""
    adapter = make_adapter(
        [
            FakeRename("x", "summary_variables"),
        ]
    )
    with pytest.raises(TypeError, match="param_keys"):
        _patched_optimize(adapter=adapter)


def test_missing_data_keys_raises_type_error():
    """TypeError when adapter lacks both summary_variables and inference_conditions."""
    adapter = make_adapter(
        [
            FakeRename("theta", "inference_variables"),
        ]
    )
    with pytest.raises(TypeError, match="data_keys"):
        _patched_optimize(adapter=adapter)


def test_inference_conditions_fallback_for_data_keys():
    """data_keys falls back to inference_conditions when summary_variables is absent."""
    adapter = make_adapter(
        [
            FakeRename("theta", "inference_variables"),
            FakeRename("x", "inference_conditions"),
        ]
    )
    # Should not raise — inference_conditions provides data_keys.
    _patched_optimize(adapter=adapter)


# ---------------------------------------------------------------------------
# Early stopping parameters
# ---------------------------------------------------------------------------


def test_optimize_forwards_early_stopping_params_to_objective_config():
    """optimize() forwards early_stopping_patience/window to ObjectiveConfig."""
    config = _patched_optimize(
        early_stopping_patience=10,
        early_stopping_window=5,
    )
    assert config.early_stopping_patience == 10
    assert config.early_stopping_window == 5


def test_optimize_early_stopping_default_values():
    """optimize() applies default patience=5, window=7 when not specified."""
    config = _patched_optimize()
    assert config.early_stopping_patience == 5
    assert config.early_stopping_window == 7


# ---------------------------------------------------------------------------
# Hook forwarding
# ---------------------------------------------------------------------------


def test_optimize_forwards_build_approximator_fn():
    """build_approximator_fn is forwarded to ObjectiveConfig."""
    sentinel = lambda hp: None  # noqa: E731
    config = _patched_optimize(build_approximator_fn=sentinel)
    assert config.build_approximator_fn is sentinel


def test_optimize_forwards_train_fn():
    """train_fn is forwarded to ObjectiveConfig."""
    sentinel = lambda approx, sim, hp, cb: None  # noqa: E731
    config = _patched_optimize(train_fn=sentinel)
    assert config.train_fn is sentinel


def test_optimize_forwards_validate_fn():
    """validate_fn is forwarded to ObjectiveConfig."""
    sentinel = lambda approx, vd, n: {}  # noqa: E731
    config = _patched_optimize(validate_fn=sentinel)
    assert config.validate_fn is sentinel


def test_optimize_forwards_n_posterior_samples():
    """n_posterior_samples is forwarded to ObjectiveConfig."""
    config = _patched_optimize(n_posterior_samples=1000)
    assert config.n_posterior_samples == 1000


def test_optimize_default_objective_metrics():
    """Default objective_metrics is ["calibration_error", "nrmse"]."""
    config = _patched_optimize()
    assert config.objective_metrics == ["calibration_error", "nrmse"]


def test_optimize_default_objective_mode():
    """Default objective_mode is "pareto"."""
    config = _patched_optimize()
    assert config.objective_mode == "pareto"


def test_optimize_forwards_report_frequency():
    """report_frequency is forwarded to ObjectiveConfig."""
    config = _patched_optimize(report_frequency=25)
    assert config.report_frequency == 25


def test_optimize_default_report_frequency():
    """Default report_frequency is 10."""
    config = _patched_optimize()
    assert config.report_frequency == 10


def test_optimize_rejects_invalid_report_frequency():
    """optimize() fails fast on report_frequency < 1 before any setup."""
    from conftest import canonical_adapter

    with pytest.raises(ValueError, match="report_frequency must be >= 1"):
        optimize(
            simulator=MagicMock(),
            adapter=canonical_adapter(),
            search_space=_make_fake_search_space(),
            storage=None,
            report_frequency=0,
        )


def test_optimize_uses_validation_simulator_for_dataset():
    """validation_simulator is passed to generate_validation_dataset."""
    val_sim = MagicMock(name="val_simulator")
    train_sim = MagicMock(name="train_simulator")
    adapter = canonical_adapter()

    with (
        patch("bayesflow_hpo.api.GenericObjective") as mock_obj_cls,
        patch("bayesflow_hpo.api.create_study"),
        patch("bayesflow_hpo.api.optimize_until"),
        patch("bayesflow_hpo.api.check_pipeline"),
        patch("bayesflow_hpo.api.generate_validation_dataset") as mock_gen,
    ):
        mock_instance = MagicMock()
        mock_instance.n_objectives = 3
        mock_obj_cls.return_value = mock_instance
        mock_gen.return_value = MagicMock()

        optimize(
            simulator=train_sim,
            adapter=adapter,
            search_space=_make_fake_search_space(),
            storage=None,
            validation_simulator=val_sim,
        )

        # generate_validation_dataset should receive val_sim, not train_sim
        call_kwargs = mock_gen.call_args
        assert call_kwargs[1]["simulator"] is val_sim


def test_optimize_defaults_to_training_simulator_for_dataset():
    """Without validation_simulator, training simulator is used."""
    train_sim = MagicMock(name="train_simulator")
    adapter = canonical_adapter()

    with (
        patch("bayesflow_hpo.api.GenericObjective") as mock_obj_cls,
        patch("bayesflow_hpo.api.create_study"),
        patch("bayesflow_hpo.api.optimize_until"),
        patch("bayesflow_hpo.api.check_pipeline"),
        patch("bayesflow_hpo.api.generate_validation_dataset") as mock_gen,
    ):
        mock_instance = MagicMock()
        mock_instance.n_objectives = 3
        mock_obj_cls.return_value = mock_instance
        mock_gen.return_value = MagicMock()

        optimize(
            simulator=train_sim,
            adapter=adapter,
            search_space=_make_fake_search_space(),
            storage=None,
        )

        call_kwargs = mock_gen.call_args
        assert call_kwargs[1]["simulator"] is train_sim


def test_optimize_calls_check_pipeline():
    """check_pipeline() is called at start of optimize()."""
    adapter = canonical_adapter()

    with (
        patch("bayesflow_hpo.api.GenericObjective") as mock_obj_cls,
        patch("bayesflow_hpo.api.create_study"),
        patch("bayesflow_hpo.api.optimize_until"),
        patch("bayesflow_hpo.api.check_pipeline") as mock_check,
        patch("bayesflow_hpo.api.generate_validation_dataset") as mock_gen,
    ):
        mock_instance = MagicMock()
        mock_instance.n_objectives = 3
        mock_obj_cls.return_value = mock_instance
        mock_gen.return_value = MagicMock()

        optimize(
            simulator=MagicMock(),
            adapter=adapter,
            search_space=_make_fake_search_space(),
            storage=None,
        )

        mock_check.assert_called_once()


# ---------------------------------------------------------------------------
# Unit tests for extracted private helpers
# ---------------------------------------------------------------------------


class TestInferAndValidateKeys:
    """Tests for _infer_and_validate_keys()."""

    def test_canonical_adapter_returns_keys(self):
        adapter = canonical_adapter()
        param_keys, data_keys = _infer_and_validate_keys(adapter)
        assert param_keys == ["theta"]
        assert data_keys == ["x"]

    def test_missing_param_keys_raises(self):
        adapter = make_adapter([FakeRename("x", "summary_variables")])
        with pytest.raises(TypeError, match="param_keys"):
            _infer_and_validate_keys(adapter)

    def test_missing_data_keys_raises(self):
        adapter = make_adapter([FakeRename("theta", "inference_variables")])
        with pytest.raises(TypeError, match="data_keys"):
            _infer_and_validate_keys(adapter)

    def test_inference_conditions_fallback(self):
        adapter = make_adapter([
            FakeRename("theta", "inference_variables"),
            FakeRename("x", "inference_conditions"),
        ])
        param_keys, data_keys = _infer_and_validate_keys(adapter)
        assert param_keys == ["theta"]
        assert data_keys == ["x"]


class TestSetupValidationData:
    """Tests for _setup_validation_data()."""

    def test_uses_validation_simulator(self):
        val_sim = MagicMock(name="val_sim")
        train_sim = MagicMock(name="train_sim")
        with patch("bayesflow_hpo.api.generate_validation_dataset") as mock_gen:
            mock_gen.return_value = MagicMock()
            _setup_validation_data(
                simulator=train_sim,
                validation_simulator=val_sim,
                param_keys=["theta"],
                data_keys=["x"],
                validation_conditions=None,
                sims_per_condition=100,
            )
            assert mock_gen.call_args[1]["simulator"] is val_sim

    def test_falls_back_to_simulator(self):
        train_sim = MagicMock(name="train_sim")
        with patch("bayesflow_hpo.api.generate_validation_dataset") as mock_gen:
            mock_gen.return_value = MagicMock()
            _setup_validation_data(
                simulator=train_sim,
                validation_simulator=None,
                param_keys=["theta"],
                data_keys=["x"],
                validation_conditions=None,
                sims_per_condition=100,
            )
            assert mock_gen.call_args[1]["simulator"] is train_sim

    def test_passes_condition_grid(self):
        conds = {"N": [50, 100]}
        with patch("bayesflow_hpo.api.generate_validation_dataset") as mock_gen:
            mock_gen.return_value = MagicMock()
            _setup_validation_data(
                simulator=MagicMock(),
                validation_simulator=None,
                param_keys=["theta"],
                data_keys=["x"],
                validation_conditions=conds,
                sims_per_condition=200,
            )
            assert mock_gen.call_args[1]["condition_grid"] is conds


class TestBuildObjective:
    """Tests for _build_objective()."""

    def test_constructs_objective_with_config(self):
        with (
            patch("bayesflow_hpo.api.GenericObjective") as mock_cls,
            patch("bayesflow_hpo.api.ObjectiveConfig") as mock_config_cls,
        ):
            mock_cls.return_value = MagicMock()
            _build_objective(
                simulator=MagicMock(),
                adapter=MagicMock(),
                search_space=MagicMock(),
                validation_data=MagicMock(),
                epochs=100,
                num_batches=50,
                early_stopping_patience=5,
                early_stopping_window=7,
                max_param_count=1_000_000,
                max_memory_mb=None,
                n_posterior_samples=500,
                objective_metrics=["calibration_error"],
                objective_mode="pareto",
                cost_metric="inference_time",
                report_frequency=10,
                build_approximator_fn=None,
                train_fn=None,
                validate_fn=None,
                checkpoint_pool=None,
            )
            mock_config_cls.assert_called_once()
            mock_cls.assert_called_once()


class TestDeriveDirections:
    """Tests for _derive_directions()."""

    def test_auto_derive_directions(self):
        obj = MagicMock()
        obj.n_objectives = 3
        directions, metric_names = _derive_directions(
            objective=obj,
            directions=None,
            objective_metrics=["calibration_error", "nrmse"],
            objective_mode="pareto",
            cost_metric="inference_time",
        )
        assert directions == ["minimize", "minimize", "minimize"]
        assert metric_names == ["calibration_error", "nrmse", "inference_time"]

    def test_explicit_directions_passthrough(self):
        obj = MagicMock()
        obj.n_objectives = 3
        directions, _ = _derive_directions(
            objective=obj,
            directions=["minimize", "minimize", "maximize"],
            objective_metrics=["calibration_error", "nrmse"],
            objective_mode="pareto",
            cost_metric="inference_time",
        )
        assert directions == ["minimize", "minimize", "maximize"]

    def test_wrong_direction_count_raises(self):
        obj = MagicMock()
        obj.n_objectives = 3
        with pytest.raises(ValueError, match="directions has 2 entries"):
            _derive_directions(
                objective=obj,
                directions=["minimize", "minimize"],
                objective_metrics=["calibration_error", "nrmse"],
                objective_mode="pareto",
                cost_metric="inference_time",
            )

    def test_mean_mode_metric_names(self):
        obj = MagicMock()
        obj.n_objectives = 2
        _, metric_names = _derive_directions(
            objective=obj,
            directions=None,
            objective_metrics=["calibration_error", "nrmse"],
            objective_mode="mean",
            cost_metric="param_count",
        )
        assert metric_names == ["mean(calibration_error+nrmse)", "param_count"]


class TestCreateAndRunStudy:
    """Tests for _create_and_run_study()."""

    def _call(self, **overrides):
        """Call _create_and_run_study with sensible defaults."""
        defaults = {
            "objective": MagicMock(),
            "study_name": "test",
            "directions": ["minimize"],
            "metric_names": ["m"],
            "storage": "sqlite:///test.db",
            "resume": False,
            "warm_start_from": None,
            "warm_start_top_k": 25,
            "n_trials": 10,
            "max_total_trials": None,
            "show_progress_bar": False,
        }
        defaults.update(overrides)
        return _create_and_run_study(**defaults)

    def test_deletes_existing_study_when_not_resuming(self):
        with (
            patch("bayesflow_hpo.api.optuna.delete_study") as mock_del,
            patch("bayesflow_hpo.api.create_study") as mock_create,
            patch("bayesflow_hpo.api.optimize_until"),
        ):
            mock_create.return_value = MagicMock()
            self._call(resume=False)
            mock_del.assert_called_once_with(
                study_name="test", storage="sqlite:///test.db"
            )

    def test_skips_delete_when_resuming(self):
        with (
            patch("bayesflow_hpo.api.optuna.delete_study") as mock_del,
            patch("bayesflow_hpo.api.create_study") as mock_create,
            patch("bayesflow_hpo.api.optimize_until"),
        ):
            mock_create.return_value = MagicMock()
            self._call(resume=True)
            mock_del.assert_not_called()

    def test_passes_load_if_exists_correctly(self):
        with (
            patch("bayesflow_hpo.api.create_study") as mock_create,
            patch("bayesflow_hpo.api.optimize_until"),
        ):
            mock_create.return_value = MagicMock()
            self._call(storage=None, resume=False)
            # storage=None → load_if_exists=True (resume or storage is None)
            assert mock_create.call_args[1]["load_if_exists"] is True
