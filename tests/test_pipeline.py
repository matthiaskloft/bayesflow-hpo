"""Tests for check_pipeline() pre-flight validation."""

import pytest
from conftest import canonical_adapter

from bayesflow_hpo.pipeline import PipelineError, _TrackingDict, check_pipeline


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


def test_check_pipeline_missing_initial_lr_raises():
    """PipelineError when search space doesn't sample initial_lr and no train_fn."""

    class _NoLrSpace:
        class _InferenceSpace:
            def build(self, params):
                return object()

        def __init__(self):
            self.inference_space = self._InferenceSpace()
            self.summary_space = None

        def sample(self, trial):
            return {"hidden_dim": 64}  # no initial_lr

    with pytest.raises(PipelineError, match="initial_lr"):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_NoLrSpace(),
            build_approximator_fn=lambda hp: _FakeApproximator(),
        )


def test_check_pipeline_missing_initial_lr_ok_with_train_fn():
    """No error when initial_lr missing but custom train_fn is provided."""

    class _NoLrSpace:
        class _InferenceSpace:
            def build(self, params):
                return object()

        def __init__(self):
            self.inference_space = self._InferenceSpace()
            self.summary_space = None

        def sample(self, trial):
            return {"hidden_dim": 64}  # no initial_lr

    check_pipeline(
        simulator=_FakeSimulator(),
        adapter=canonical_adapter(),
        search_space=_NoLrSpace(),
        build_approximator_fn=lambda hp: _FakeApproximator(),
        train_fn=lambda approx, sim, hp, cb: None,
        validate_fn=lambda approx, vd, n: {"calibration_error": 0.05, "nrmse": 0.1},
    )


def test_check_pipeline_train_fn_wrong_arity_raises():
    """PipelineError when train_fn has wrong number of parameters."""

    def bad_train(approx, sim):  # only 2 args, should be 4
        pass

    with pytest.raises(PipelineError, match="train_fn must accept exactly 4"):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=lambda hp: _FakeApproximator(),
            train_fn=bad_train,
        )


def test_check_pipeline_validate_fn_wrong_arity_raises():
    """PipelineError when validate_fn has wrong number of parameters."""

    def bad_validate(approx):  # only 1 arg, should be 3
        return {}

    with pytest.raises(PipelineError, match="validate_fn must accept exactly 3"):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=lambda hp: _FakeApproximator(),
            validate_fn=bad_validate,
        )


def test_check_pipeline_build_fn_wrong_arity_raises():
    """PipelineError when build_approximator_fn has wrong number of parameters."""

    def bad_builder(a, b):  # 2 args, should be 1
        return _FakeApproximator()

    match = "build_approximator_fn must accept exactly 1"
    with pytest.raises(PipelineError, match=match):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=bad_builder,
        )


def test_check_pipeline_train_fn_error_propagates():
    """PipelineError wraps exceptions from custom train_fn."""

    def exploding_train(approx, sim, hp, cb):
        raise RuntimeError("train exploded")

    with pytest.raises(PipelineError, match="Training step failed.*train exploded"):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=lambda hp: _FakeApproximator(),
            train_fn=exploding_train,
            validate_fn=lambda approx, vd, n: {"calibration_error": 0.05, "nrmse": 0.1},
        )


def test_check_pipeline_validate_fn_error_propagates():
    """PipelineError wraps exceptions from custom validate_fn."""

    def exploding_validate(approx, vd, n):
        raise RuntimeError("validate exploded")

    with pytest.raises(PipelineError, match="Validation step failed"):
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=lambda hp: _FakeApproximator(),
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=exploding_validate,
        )


# ---------------------------------------------------------------------------
# _TrackingDict tests
# ---------------------------------------------------------------------------


class TestTrackingDict:
    """Tests for _TrackingDict key-access tracking."""

    def test_items_marks_all_keys_accessed(self):
        td = _TrackingDict({"a": 1, "b": 2, "c": 3})
        list(td.items())
        assert td.accessed_keys == {"a", "b", "c"}

    def test_values_marks_all_keys_accessed(self):
        td = _TrackingDict({"x": 10, "y": 20})
        list(td.values())
        assert td.accessed_keys == {"x", "y"}

    def test_dict_copy_does_not_mark_keys(self):
        """dict(td) uses __iter__, which must NOT mark keys as accessed."""
        td = _TrackingDict({"a": 1, "b": 2})
        _ = dict(td)
        assert td.accessed_keys == set()

    def test_items_returns_correct_pairs(self):
        td = _TrackingDict({"a": 1, "b": 2})
        assert sorted(td.items()) == [("a", 1), ("b", 2)]

    def test_values_returns_correct_values(self):
        td = _TrackingDict({"a": 1, "b": 2})
        assert sorted(td.values()) == [1, 2]


class TestPreflightCollisionResolutionIsOrderFree:
    """`check_pipeline` re-keys the hook's summary, and it was the fourth
    such boundary — the other three were made collision-aware together and
    this one stayed a last-write-wins comprehension.

    Its failure mode is a rejection rather than a misranking. A hook emitting
    both spellings of one metric, one finite and one not, made pre-flight
    accept or refuse the whole run on nothing but the insertion order of a
    dict the hook happened to build.

    This drives `check_pipeline` rather than `canonical_summary`, so a revert
    to the comprehension fails here.
    """

    @pytest.mark.parametrize("order", ["canonical_first", "alias_first"])
    def test_a_finite_canonical_value_is_accepted_in_either_order(
        self, order: str
    ) -> None:
        pairs = [("calibration_error", 0.02), ("cal_error", float("nan"))]
        if order == "alias_first":
            pairs.reverse()

        def colliding_validate(approx, vd, n):
            return dict(pairs)

        # No PipelineError: the canonical entry wins regardless of order.
        check_pipeline(
            simulator=_FakeSimulator(),
            adapter=canonical_adapter(),
            search_space=_FakeSearchSpace(),
            build_approximator_fn=lambda hp: _FakeApproximator(),
            train_fn=lambda approx, sim, hp, cb: None,
            validate_fn=colliding_validate,
            objective_metrics=["calibration_error"],
        )

    @pytest.mark.parametrize("order", ["canonical_first", "alias_first"])
    def test_a_non_finite_canonical_value_is_refused_in_either_order(
        self, order: str
    ) -> None:
        """The rule is "canonical wins", not "the finite one wins"."""
        pairs = [("calibration_error", float("nan")), ("cal_error", 0.02)]
        if order == "alias_first":
            pairs.reverse()

        def colliding_validate(approx, vd, n):
            return dict(pairs)

        with pytest.raises(PipelineError, match="non-finite"):
            check_pipeline(
                simulator=_FakeSimulator(),
                adapter=canonical_adapter(),
                search_space=_FakeSearchSpace(),
                build_approximator_fn=lambda hp: _FakeApproximator(),
                train_fn=lambda approx, sim, hp, cb: None,
                validate_fn=colliding_validate,
                objective_metrics=["calibration_error"],
            )
