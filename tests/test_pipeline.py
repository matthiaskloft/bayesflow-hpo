"""Tests for check_pipeline() pre-flight validation."""

import pytest
from conftest import canonical_adapter

from bayesflow_hpo.pipeline import PipelineError, check_pipeline


class _FakeSearchSpace:
    class _InferenceSpace:
        def build(self, params):
            return object()

    def __init__(self):
        self.inference_space = self._InferenceSpace()
        self.summary_space = None

    def sample(self, trial):
        return {"initial_lr": 1e-3}


class _FakeApproximator:
    def fit(self, **kwargs):
        pass

    def compile(self, *args, **kwargs):
        pass

    def compute_loss(self, data):
        pass

    def sample(self, conditions=None, num_samples=1):
        return None


class _FakeSimulator:
    def sample(self, n_sims, conditions=None, seed=None):
        import numpy as np

        rng = np.random.default_rng(seed)
        n = n_sims if isinstance(n_sims, int) else n_sims[0]
        return {
            "theta": rng.normal(size=(n, 1)),
            "x": rng.normal(size=(n, 1)),
        }


def test_check_pipeline_build_failure_raises():
    """PipelineError when builder fails."""

    def bad_builder(hparams):
        raise ValueError("broken builder")

    with pytest.raises(PipelineError, match="Build step failed"):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=bad_builder,
        )


def test_check_pipeline_missing_fit_raises():
    """PipelineError when builder returns object without fit."""

    def no_fit_builder(hparams):
        return object()  # no fit method

    with pytest.raises(PipelineError, match="no 'fit' method"):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=no_fit_builder,
        )


def test_check_pipeline_validate_fn_missing_keys_raises():
    """PipelineError when validate_fn returns wrong keys."""

    def bad_validate(approx, vd, n):
        return {"wrong_key": 0.5}

    with pytest.raises(PipelineError, match="missing required metric keys"):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=lambda hp: _FakeApproximator(),
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=bad_validate,
            objective_metrics=["calibration_error"],
        )


def test_check_pipeline_validate_fn_non_finite_raises():
    """PipelineError when validate_fn returns NaN."""

    def nan_validate(approx, vd, n):
        return {"calibration_error": float("nan")}

    with pytest.raises(PipelineError, match="non-finite"):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=lambda hp: _FakeApproximator(),
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=nan_validate,
            objective_metrics=["calibration_error"],
        )


def test_check_pipeline_valid_custom_hooks():
    """No error when all custom hooks work correctly."""

    def good_validate(approx, vd, n):
        return {"calibration_error": 0.05, "nrmse": 0.1}

    check_pipeline(
        simulator=_FakeSimulator(),
        adapter=canonical_adapter(),
        search_space=_FakeSearchSpace(),
        build_approximator_fn=lambda hp: _FakeApproximator(),
        train_fn=lambda approx, sim, hp, cb: None,
        validate_fn=good_validate,
        objective_metrics=["calibration_error", "nrmse"],
    )


def test_check_pipeline_warns_unused_hparams(caplog):
    """Warning when builder doesn't read all sampled hparams."""

    def selective_builder(hparams):
        _ = hparams["initial_lr"]  # only reads one key
        return _FakeApproximator()

    class _ExtraParamSpace:
        class _InferenceSpace:
            def build(self, params):
                return object()

        def __init__(self):
            self.inference_space = self._InferenceSpace()
            self.summary_space = None

        def sample(self, trial):
            return {"initial_lr": 1e-3, "hidden_dim": 64, "depth": 4}

    import logging

    with caplog.at_level(logging.WARNING):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_ExtraParamSpace(),
            build_approximator_fn=selective_builder,
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=lambda approx, vd, n: {"calibration_error": 0.05, "nrmse": 0.1},
        )

    assert "never read" in caplog.text
