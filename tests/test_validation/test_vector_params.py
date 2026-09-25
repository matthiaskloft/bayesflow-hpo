"""Vector-valued parameters: one row per (simulation, element) (#114).

Each ``sim_batch[key]`` of shape ``(n_sims, W)`` is pooled by folding its
elements into rows, for the joint and the marginal path alike.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bayesflow_hpo.validation.c2st import make_lc2st_joint_metric
from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.pipeline import (
    _fold_vector_draws,
    run_validation_pipeline,
)
from bayesflow_hpo.validation.registry import (
    REQUIRES_SCALAR_PARAMETERS,
    JointMetricConfigurationError,
    JointMetricInputs,
)

N_SIMS, N_ITEMS, N_SAMPLES = 6, 4, 32


class _TruthApproximator:
    """Draws centred on the truth for ``good`` keys and far off for others.

    Returns per-key ``(n_sims, n_samples, W)`` arrays, the shape BayesFlow's
    sampler gives a vector-valued key, so the real ``make_bayesflow_infer_fn``
    concatenation runs.
    """

    def __init__(self, truths: dict[str, np.ndarray], good: set[str]) -> None:
        self.truths = truths
        self.good = good

    def sample(self, *, conditions: Any, num_samples: int) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        out = {}
        for key, truth in self.truths.items():
            truth = truth.reshape(truth.shape[0], -1)
            noise = rng.normal(scale=0.1, size=(truth.shape[0], num_samples,
                                                truth.shape[1]))
            centre = truth if key in self.good else truth + 50.0
            out[key] = centre[:, None, :] + noise
        return out


def _dataset(params: dict[str, np.ndarray]) -> ValidationDataset:
    sim = {**params, "x": np.zeros((N_SIMS, 3))}
    return ValidationDataset(
        simulations=[sim],
        condition_labels=[{"c": 0}],
        param_keys=list(params),
        data_keys=["x"],
        seed=0,
    )


def test_fold_layout_pins_rows_and_columns() -> None:
    """Row s*W+i is (sim s, element i); column k is key k."""
    n_sims, n_samples, n_keys, width = 3, 2, 2, 4
    # value = 1000*sim + 100*sample + 10*key + element
    s, n, k, i = np.meshgrid(
        np.arange(n_sims), np.arange(n_samples), np.arange(n_keys),
        np.arange(width), indexing="ij",
    )
    values = 1000 * s + 100 * n + 10 * k + i
    # Key-by-key concatenation, as make_bayesflow_infer_fn produces it.
    draws = np.concatenate([values[:, :, kk, :] for kk in range(n_keys)], -1)

    folded = _fold_vector_draws(draws, n_keys, width)

    assert folded.shape == (n_sims * width, n_samples, n_keys)
    for ss in range(n_sims):
        for ii in range(width):
            for nn in range(n_samples):
                for kk in range(n_keys):
                    assert folded[ss * width + ii, nn, kk] == (
                        1000 * ss + 100 * nn + 10 * kk + ii
                    )
    # And the truths the pipeline builds use the same row order.
    truth = np.arange(n_sims * width).reshape(n_sims, width)
    assert list(truth.reshape(-1)) == [
        ss * width + ii for ss in range(n_sims) for ii in range(width)
    ]


def test_fold_is_identity_at_width_one() -> None:
    draws = np.random.default_rng(0).normal(size=(5, 7, 2))
    assert _fold_vector_draws(draws, 2, 1) is draws


def test_marginal_metrics_score_each_keys_own_columns() -> None:
    rng = np.random.default_rng(3)
    truths = {
        "a": rng.normal(size=(N_SIMS, N_ITEMS)),
        "b": rng.normal(size=(N_SIMS, N_ITEMS)),
    }
    result = run_validation_pipeline(
        approximator=_TruthApproximator(truths, good={"a"}),
        validation_data=_dataset(truths),
        n_posterior_samples=N_SAMPLES,
        metrics=["rmse"],
    )
    rmse_a = result.per_parameter["a"].summary["rmse"]
    rmse_b = result.per_parameter["b"].summary["rmse"]
    assert rmse_a < 0.1
    assert rmse_b > 40.0


def test_single_vector_key_marginal() -> None:
    rng = np.random.default_rng(4)
    truths = {"a": rng.normal(size=(N_SIMS, N_ITEMS))}
    result = run_validation_pipeline(
        approximator=_TruthApproximator(truths, good={"a"}),
        validation_data=_dataset(truths),
        n_posterior_samples=N_SAMPLES,
        metrics=["rmse"],
    )
    assert result.summary["rmse"] < 0.1


def test_joint_metric_runs_on_vector_params() -> None:
    rng = np.random.default_rng(5)
    truths = {
        "a": rng.normal(size=(N_SIMS, N_ITEMS)),
        "b": rng.normal(size=(N_SIMS, N_ITEMS)),
    }
    seen: dict[str, tuple[int, ...]] = {}

    result = run_validation_pipeline(
        approximator=_TruthApproximator(truths, good={"a", "b"}),
        validation_data=_dataset(truths),
        n_posterior_samples=N_SAMPLES,
        metrics=["rmse", "tarp_error_random"],
    )
    assert result.failed_joint_metrics == {}
    assert np.isfinite(result.summary["tarp_error_random"])

    def _probe(inputs: JointMetricInputs) -> dict[str, float]:
        seen["draws"] = inputs.draws.shape
        seen["true"] = inputs.true_values.shape
        # Draws centred on the truth: row r column k matches truth row r.
        err = np.abs(inputs.draws.mean(axis=1) - inputs.true_values).max()
        return {"tarp_error": float(err)}

    result = run_validation_pipeline(
        approximator=_TruthApproximator(truths, good={"a", "b"}),
        validation_data=_dataset(truths),
        n_posterior_samples=N_SAMPLES,
        metrics=["rmse"],
        joint_metrics={"tarp_error": _probe},
    )
    assert result.failed_joint_metrics == {}
    assert seen["draws"] == (N_SIMS * N_ITEMS, N_SAMPLES, 2)
    assert seen["true"] == (N_SIMS * N_ITEMS, 2)
    assert result.summary["tarp_error"] < 0.1


def test_mixed_widths_raise_before_inference() -> None:
    rng = np.random.default_rng(6)
    truths = {
        "a": rng.normal(size=(N_SIMS, N_ITEMS)),
        "theta": rng.normal(size=(N_SIMS, 3)),
    }

    class _NeverSample:
        def sample(self, **_: Any) -> Any:
            raise AssertionError("inference ran before the config check")

    with pytest.raises(JointMetricConfigurationError, match="different numbers"):
        run_validation_pipeline(
            approximator=_NeverSample(),
            validation_data=_dataset(truths),
            n_posterior_samples=N_SAMPLES,
            metrics=["rmse"],
        )


def test_lc2st_refused_for_vector_params() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    truths = {"a": rng.normal(size=(N_SIMS, N_ITEMS))}
    with pytest.raises(JointMetricConfigurationError, match="L-C2ST"):
        run_validation_pipeline(
            approximator=_TruthApproximator(truths, good={"a"}),
            validation_data=_dataset(truths),
            n_posterior_samples=N_SAMPLES,
            metrics=["rmse"],
            joint_metrics={"lc2st": make_lc2st_joint_metric()},
        )


@pytest.mark.parametrize("shape", [(N_SIMS,), (N_SIMS, 1)])
def test_scalar_params_unaffected(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(8)
    truths = {
        "a": rng.normal(size=shape),
        "b": rng.normal(size=shape),
    }
    result = run_validation_pipeline(
        approximator=_TruthApproximator(truths, good={"a"}),
        validation_data=_dataset(truths),
        n_posterior_samples=N_SAMPLES,
        metrics=["rmse"],
    )
    assert result.per_parameter["a"].summary["rmse"] < 0.1
    assert result.per_parameter["b"].summary["rmse"] > 40.0


def test_scalar_only_joint_metric_refused_for_vector_params() -> None:
    """The marker is generic, not tied to L-C2ST (no sklearn needed)."""
    rng = np.random.default_rng(9)
    truths = {"a": rng.normal(size=(N_SIMS, N_ITEMS))}

    def _rowwise(inputs: JointMetricInputs) -> dict[str, float]:
        raise AssertionError("must be refused before it runs")

    setattr(_rowwise, REQUIRES_SCALAR_PARAMETERS, "pairs data by row.")
    with pytest.raises(JointMetricConfigurationError, match="pairs data"):
        run_validation_pipeline(
            approximator=_TruthApproximator(truths, good={"a"}),
            validation_data=_dataset(truths),
            n_posterior_samples=N_SAMPLES,
            metrics=["rmse"],
            joint_metrics={"tarp_error": _rowwise},
        )


def _grid(conditions: list[dict[str, np.ndarray]]) -> ValidationDataset:
    sims = [{**c, "x": np.zeros((N_SIMS, 3))} for c in conditions]
    return ValidationDataset(
        simulations=sims,
        condition_labels=[{"c": i} for i in range(len(sims))],
        param_keys=list(conditions[0]),
        data_keys=["x"],
        seed=0,
    )


class _GridApproximator:
    """Like `_TruthApproximator`, but serves each condition's own truths.

    Conditions are told apart by the id stored in ``x``.
    """

    def __init__(self, grid: list[dict[str, np.ndarray]], good: set[str]):
        self.inner = [_TruthApproximator(c, good) for c in grid]

    def sample(self, *, conditions: Any, num_samples: int) -> dict[str, Any]:
        cond = int(np.asarray(conditions["x"])[0, 0])
        return self.inner[cond].sample(
            conditions=conditions, num_samples=num_samples
        )


def _grid_with_ids(grid: list[dict[str, np.ndarray]]) -> ValidationDataset:
    data = _grid(grid)
    for i, sim in enumerate(data.simulations):
        sim["x"] = np.full((N_SIMS, 3), float(i))
    return data


def test_width_may_vary_across_conditions() -> None:
    rng = np.random.default_rng(10)
    grid = [
        {"a": rng.normal(size=(N_SIMS,)), "b": rng.normal(size=(N_SIMS,))},
        {"a": rng.normal(size=(N_SIMS, 3)), "b": rng.normal(size=(N_SIMS, 3))},
        {"a": rng.normal(size=(N_SIMS, 5)), "b": rng.normal(size=(N_SIMS, 5))},
    ]
    result = run_validation_pipeline(
        approximator=_GridApproximator(grid, good={"a"}),
        validation_data=_grid_with_ids(grid),
        n_posterior_samples=N_SAMPLES,
        metrics=["rmse", "tarp_error_random"],
    )
    assert result.failed_joint_metrics == {}
    rows_a = result.per_parameter["a"].condition_metrics
    rows_b = result.per_parameter["b"].condition_metrics
    assert (rows_a["rmse"] < 0.1).all()
    assert (rows_b["rmse"] > 40.0).all()
    # The column counts simulations, not the pooled rows.
    assert list(rows_a["n_sims"]) == [N_SIMS] * 3


def test_scalar_only_metric_refused_if_a_later_condition_is_vector() -> None:
    rng = np.random.default_rng(11)
    grid = [
        {"a": rng.normal(size=(N_SIMS,))},
        {"a": rng.normal(size=(N_SIMS, 4))},
    ]

    def _rowwise(inputs: JointMetricInputs) -> dict[str, float]:
        raise AssertionError("must be refused before it runs")

    setattr(_rowwise, REQUIRES_SCALAR_PARAMETERS, "pairs data by row.")

    class _NeverSample:
        def sample(self, **_: Any) -> Any:
            raise AssertionError("inference ran before the config check")

    with pytest.raises(JointMetricConfigurationError, match="pairs data"):
        run_validation_pipeline(
            approximator=_NeverSample(),
            validation_data=_grid(grid),
            n_posterior_samples=N_SAMPLES,
            metrics=["rmse"],
            joint_metrics={"tarp_error": _rowwise},
        )


def test_mixed_widths_in_a_later_condition_raise_before_inference() -> None:
    rng = np.random.default_rng(12)
    grid = [
        {"a": rng.normal(size=(N_SIMS, 2)), "b": rng.normal(size=(N_SIMS, 2))},
        {"a": rng.normal(size=(N_SIMS, 2)), "b": rng.normal(size=(N_SIMS, 3))},
    ]

    class _NeverSample:
        def sample(self, **_: Any) -> Any:
            raise AssertionError("inference ran before the config check")

    with pytest.raises(JointMetricConfigurationError, match="different numbers"):
        run_validation_pipeline(
            approximator=_NeverSample(),
            validation_data=_grid(grid),
            n_posterior_samples=N_SAMPLES,
            metrics=["rmse"],
        )


class _MatrixApproximator:
    """Per-key draws ``(S, N, W1, W2)``: several trailing axes per key."""

    def __init__(self, truths: dict[str, np.ndarray], good: set[str]) -> None:
        self.truths = truths
        self.good = good

    def sample(self, *, conditions: Any, num_samples: int) -> dict[str, Any]:
        out = {}
        for key, truth in self.truths.items():
            centre = truth if key in self.good else truth + 50.0
            out[key] = np.repeat(centre[:, None], num_samples, axis=1)
        return out


def test_multi_key_matrix_valued_params_fold_correctly() -> None:
    rng = np.random.default_rng(13)
    truths = {
        "a": rng.normal(size=(N_SIMS, 2, 3)),
        "b": rng.normal(size=(N_SIMS, 2, 3)),
    }
    seen: dict[str, np.ndarray] = {}

    def _probe(inputs: JointMetricInputs) -> dict[str, float]:
        seen["draws"] = inputs.draws
        seen["true"] = inputs.true_values
        return {"tarp_error": 0.0}

    result = run_validation_pipeline(
        approximator=_MatrixApproximator(truths, good={"a"}),
        validation_data=_dataset(truths),
        n_posterior_samples=N_SAMPLES,
        metrics=["rmse"],
        joint_metrics={"tarp_error": _probe},
    )
    assert result.failed_joint_metrics == {}
    assert result.per_parameter["a"].summary["rmse"] < 1e-9
    assert result.per_parameter["b"].summary["rmse"] > 40.0
    assert seen["draws"].shape == (N_SIMS * 6, N_SAMPLES, 2)
    np.testing.assert_allclose(seen["draws"][:, 0, 0], truths["a"].reshape(-1))
    np.testing.assert_allclose(
        seen["draws"][:, 0, 1], truths["b"].reshape(-1) + 50.0
    )
    np.testing.assert_allclose(seen["true"][:, 1], truths["b"].reshape(-1))
