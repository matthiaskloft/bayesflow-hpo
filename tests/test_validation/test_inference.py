"""Tests for make_bayesflow_infer_fn data_keys validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn
from bayesflow_hpo.validation.pipeline import run_validation_pipeline


def _make_approximator(param_keys: list[str]) -> MagicMock:
    """Create a mock approximator that returns dummy posterior draws."""
    approx = MagicMock()
    approx.sample.return_value = {
        k: np.random.randn(2, 10, 1) for k in param_keys
    }
    return approx


class TestDataKeysValidation:
    def test_missing_data_keys_raises_keyerror(self):
        """Passing available_keys that miss a required data_key raises KeyError."""
        with pytest.raises(KeyError, match="missing_key"):
            make_bayesflow_infer_fn(
                approximator=_make_approximator(["theta"]),
                param_keys=["theta"],
                data_keys=["x", "missing_key"],
                available_keys={"x", "y"},
            )

    def test_available_keys_none_skips_check(self):
        """When available_keys is None, no upfront validation occurs."""
        fn = make_bayesflow_infer_fn(
            approximator=_make_approximator(["theta"]),
            param_keys=["theta"],
            data_keys=["x"],
            available_keys=None,
        )
        assert callable(fn)

    def test_all_keys_present_succeeds(self):
        """When all data_keys are in available_keys, construction succeeds."""
        fn = make_bayesflow_infer_fn(
            approximator=_make_approximator(["theta"]),
            param_keys=["theta"],
            data_keys=["x", "y"],
            available_keys={"x", "y", "z"},
        )
        assert callable(fn)

    def test_closure_raises_on_missing_key(self):
        """The infer_fn closure raises KeyError when sim_data lacks a data_key."""
        fn = make_bayesflow_infer_fn(
            approximator=_make_approximator(["theta"]),
            param_keys=["theta"],
            data_keys=["x", "y"],
            available_keys=None,
        )
        with pytest.raises(KeyError, match="y"):
            fn({"x": np.ones((2, 5))}, n_posterior_samples=10)


class TestPipelinePassesAvailableKeys:
    def test_pipeline_raises_on_mismatched_data_keys(self):
        """run_validation_pipeline detects data_keys missing from simulations."""
        vdata = ValidationDataset(
            simulations=[{"x": np.ones((5, 3))}],
            condition_labels=[{"cond": "a"}],
            param_keys=["theta"],
            data_keys=["x", "missing_key"],
            seed=0,
        )
        with pytest.raises(KeyError, match="missing_key"):
            run_validation_pipeline(
                approximator=_make_approximator(["theta"]),
                validation_data=vdata,
            )


class TestChunkedSampling:
    """Chunking bounds the peak allocation of validation inference (#101)."""

    @staticmethod
    def _chunking_approximator(param_keys: list[str], n_params_axis: bool = True):
        """Approximator whose draws depend on the rows it was handed."""
        approx = MagicMock()
        calls: list[int] = []

        def _sample(*, conditions, num_samples):
            rows = int(np.asarray(next(iter(conditions.values()))).shape[0])
            calls.append(rows)
            # Values encode the row index so the assembled array can be
            # checked for order, not merely for shape.
            base = np.asarray(conditions["x"])[:, :1]
            draws = np.repeat(base, num_samples, axis=1)
            if n_params_axis:
                return {k: draws[..., None] for k in param_keys}
            return {k: draws for k in param_keys}

        approx.sample.side_effect = _sample
        approx.calls = calls
        return approx

    def test_batch_is_split_into_slices_under_the_cap(self):
        approx = self._chunking_approximator(["theta"])
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["theta"],
            data_keys=["x"],
            max_samples_per_call=40,
        )
        sim_data = {"x": np.arange(10, dtype=float).reshape(10, 1)}

        draws = fn(sim_data, n_posterior_samples=10)

        # 40 // 10 = 4 rows per call over 10 rows.
        assert approx.calls == [4, 4, 2]
        assert draws.shape == (10, 10)
        # Rows come back in their original order.
        assert np.array_equal(draws[:, 0], np.arange(10, dtype=float))

    def test_chunked_and_unchunked_agree(self):
        sim_data = {"x": np.arange(12, dtype=float).reshape(12, 1)}
        whole = make_bayesflow_infer_fn(
            approximator=self._chunking_approximator(["theta"]),
            param_keys=["theta"],
            data_keys=["x"],
            max_samples_per_call=None,
        )(sim_data, n_posterior_samples=5)
        chunked = make_bayesflow_infer_fn(
            approximator=self._chunking_approximator(["theta"]),
            param_keys=["theta"],
            data_keys=["x"],
            max_samples_per_call=10,
        )(sim_data, n_posterior_samples=5)
        assert np.array_equal(whole, chunked)

    def test_multi_parameter_draws_keep_their_layout(self):
        approx = self._chunking_approximator(["a", "b"], n_params_axis=False)
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["a", "b"],
            data_keys=["x"],
            max_samples_per_call=12,
        )
        sim_data = {"x": np.arange(8, dtype=float).reshape(8, 1)}

        draws = fn(sim_data, n_posterior_samples=4)

        assert approx.calls == [3, 3, 2]
        assert draws.shape == (8, 4, 2)
        assert np.array_equal(draws[:, 0, 0], np.arange(8, dtype=float))

    def test_sample_count_above_the_cap_is_honoured(self):
        """One row per call rather than a silently reduced sample count."""
        approx = self._chunking_approximator(["theta"])
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["theta"],
            data_keys=["x"],
            max_samples_per_call=10,
        )
        draws = fn({"x": np.arange(3, dtype=float).reshape(3, 1)}, 50)

        assert approx.calls == [1, 1, 1]
        assert draws.shape == (3, 50)

    def test_single_call_when_the_batch_already_fits(self):
        approx = self._chunking_approximator(["theta"])
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["theta"],
            data_keys=["x"],
            max_samples_per_call=1000,
        )
        fn({"x": np.arange(4, dtype=float).reshape(4, 1)}, 10)

        assert approx.calls == [4]

    @pytest.mark.parametrize("cap", [20_000.0, 2e4, 1.5])
    def test_float_cap_is_rejected(self, cap):
        """A float cap fails only on the CHUNKED path, after training.

        `20_000.0 // 500` is `40.0`, which `range()` refuses -- but a
        pre-flight batch small enough to take the single-call path never
        reaches that line, so the TypeError arrives at final validation.
        """
        with pytest.raises(TypeError, match="max_samples_per_call must be an int"):
            make_bayesflow_infer_fn(
                approximator=_make_approximator(["theta"]),
                param_keys=["theta"],
                data_keys=["x"],
                max_samples_per_call=cap,
            )

    def test_non_positive_cap_is_rejected(self):
        with pytest.raises(ValueError, match="max_samples_per_call"):
            make_bayesflow_infer_fn(
                approximator=_make_approximator(["theta"]),
                param_keys=["theta"],
                data_keys=["x"],
                max_samples_per_call=0,
            )

    def test_broadcast_row_does_not_disable_chunking(self):
        """A leading dim of 1 is broadcast, not a one-row batch.

        Counting it as the batch size made ``rows_per_call >= n_rows``
        trivially true, so a condition carrying any broadcast covariate
        went through in a single unchunked call -- the exact allocation
        the cap exists to bound, with the cap set and looking effective.
        """
        approx = self._chunking_approximator(["theta"])
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["theta"],
            data_keys=["x", "ctx"],
            max_samples_per_call=40,
        )
        sim_data = {
            "x": np.arange(10, dtype=float).reshape(10, 1),
            "ctx": np.ones((1, 3)),
        }

        draws = fn(sim_data, n_posterior_samples=10)

        assert approx.calls == [4, 4, 2]
        # The broadcast value reaches every chunk whole; slicing it would
        # have handed chunks 2 and 3 an empty array.
        ctx_rows = [
            np.asarray(call.kwargs["conditions"]["ctx"]).shape[0]
            for call in approx.sample.call_args_list
        ]
        assert ctx_rows == [1, 1, 1]
        assert np.array_equal(draws[:, 0], np.arange(10, dtype=float))

    def test_zero_dim_condition_does_not_disable_chunking(self):
        approx = self._chunking_approximator(["theta"])
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["theta"],
            data_keys=["x", "n_obs"],
            max_samples_per_call=40,
        )
        sim_data = {
            "x": np.arange(10, dtype=float).reshape(10, 1),
            "n_obs": np.float64(50.0),
        }

        fn(sim_data, n_posterior_samples=10)

        assert approx.calls == [4, 4, 2]

    def test_mismatched_batch_sizes_raise(self):
        """Slicing to the shorter array would silently drop the tail."""
        fn = make_bayesflow_infer_fn(
            approximator=self._chunking_approximator(["theta"]),
            param_keys=["theta"],
            data_keys=["x", "y"],
            max_samples_per_call=40,
        )
        sim_data = {
            "x": np.arange(10, dtype=float).reshape(10, 1),
            "y": np.zeros((6, 2)),
        }

        with pytest.raises(ValueError, match="disagree on their batch size"):
            fn(sim_data, n_posterior_samples=10)

    def test_unreadable_shape_falls_back_to_one_call(self):
        """Shape probing is new and must not reject what used to work."""

        class _NoShape:
            """A conditioning value NumPy cannot describe."""

            def __array__(self, *args, **kwargs):
                raise TypeError("not arrayable")

        approx = MagicMock()
        approx.sample.return_value = {"theta": np.zeros((3, 5, 1))}
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["theta"],
            data_keys=["x"],
            max_samples_per_call=10,
        )

        draws = fn({"x": _NoShape()}, n_posterior_samples=5)

        assert approx.sample.call_count == 1
        assert draws.shape == (3, 5)

    def test_zero_posterior_samples_does_not_divide_by_zero(self):
        approx = self._chunking_approximator(["theta"])
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["theta"],
            data_keys=["x"],
            max_samples_per_call=10,
        )

        draws = fn({"x": np.arange(4, dtype=float).reshape(4, 1)}, 0)

        assert draws.shape == (4, 0)

    def test_empty_condition_batch(self):
        approx = self._chunking_approximator(["theta"])
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["theta"],
            data_keys=["x"],
            max_samples_per_call=10,
        )

        fn({"x": np.zeros((0, 1))}, n_posterior_samples=5)

        assert approx.sample.call_count == 1


    def test_assembly_preserves_dtype_without_concatenating(self):
        """Chunks are written into one preallocated array.

        `np.concatenate` would hold every chunk and the finished result
        live at once, peaking at twice the returned array -- a transient
        the per-call cap does not bound.
        """
        approx = MagicMock()

        def _sample(*, conditions, num_samples):
            rows = int(np.asarray(conditions["x"]).shape[0])
            return {"theta": np.zeros((rows, num_samples, 1), dtype=np.float32)}

        approx.sample.side_effect = _sample
        fn = make_bayesflow_infer_fn(
            approximator=approx,
            param_keys=["theta"],
            data_keys=["x"],
            max_samples_per_call=20,
        )

        draws = fn({"x": np.zeros((7, 1))}, n_posterior_samples=10)

        assert draws.shape == (7, 10)
        assert draws.dtype == np.float32
        assert approx.sample.call_count == 4  # 2 rows per call, last is 1


class TestPipelineForwardsTheCap:
    """The knob has to reach the closure, not merely be accepted."""

    def test_run_validation_pipeline_forwards_max_samples_per_call(self):
        approx = TestChunkedSampling._chunking_approximator(["theta"])
        vdata = ValidationDataset(
            simulations=[
                {
                    "x": np.arange(8, dtype=float).reshape(8, 1),
                    "theta": np.arange(8, dtype=float),
                }
            ],
            condition_labels=[{"cond": "a"}],
            param_keys=["theta"],
            data_keys=["x"],
            seed=0,
        )

        run_validation_pipeline(
            approximator=approx,
            validation_data=vdata,
            n_posterior_samples=4,
            metrics=["nrmse"],
            max_samples_per_call=12,
        )

        assert approx.calls == [3, 3, 2]
