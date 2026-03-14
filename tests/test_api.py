"""Tests for the high-level optimize() API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from conftest import FakeRename, canonical_adapter, make_adapter

from bayesflow_hpo.api import optimize


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
    """TypeError when adapter lacks summary_variables."""
    adapter = make_adapter(
        [
            FakeRename("theta", "inference_variables"),
        ]
    )
    with pytest.raises(TypeError, match="data_keys"):
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
