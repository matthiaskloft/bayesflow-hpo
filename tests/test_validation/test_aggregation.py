"""Tests for per-metric validation-grid aggregation."""

import numpy as np
import pytest

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.metrics import (
    aggregate_condition_rows,
    normalize_aggregate,
)
from bayesflow_hpo.validation.pipeline import run_validation_pipeline
from bayesflow_hpo.validation.registry import (
    JointMetricInputs,
    register_joint_metric,
    register_metric,
    unregister_metric,
)


def _rows() -> list[dict[str, float]]:
    return [
        {"id_cond": 0, "n_sims": 10, "nrmse": 1.0, "mae": 2.0},
        {"id_cond": 1, "n_sims": 10, "nrmse": 4.0, "mae": 6.0},
    ]


def test_normalize_aggregate_accepts_scalar_and_metric_mapping() -> None:
    assert normalize_aggregate("mean") == "mean"
    assert normalize_aggregate({"nrmse": "geometric"}) == {
        "nrmse": "geometric"
    }


def test_aggregate_condition_rows_keeps_mean_as_default() -> None:
    summary = aggregate_condition_rows(_rows())
    assert summary["nrmse"] == pytest.approx(2.5)
    assert summary["mae"] == pytest.approx(4.0)


def test_aggregate_condition_rows_supports_worst_case_reduction() -> None:
    summary = aggregate_condition_rows(_rows(), aggregate="worst")
    assert summary["nrmse"] == pytest.approx(4.0)
    assert summary["mae"] == pytest.approx(6.0)


def test_aggregate_condition_rows_supports_geometric_reduction() -> None:
    summary = aggregate_condition_rows(_rows(), aggregate="geometric")
    assert summary["nrmse"] == pytest.approx(2.0)
    assert summary["mae"] == pytest.approx(np.sqrt(12.0))


def test_metric_mapping_uses_mean_for_unspecified_metrics() -> None:
    summary = aggregate_condition_rows(
        _rows(), aggregate={"nrmse": "geometric"}
    )
    assert summary["nrmse"] == pytest.approx(2.0)
    assert summary["mae"] == pytest.approx(4.0)


def test_aggregation_skips_nan_values() -> None:
    rows = [
        {"id_cond": 0, "n_sims": 10, "nrmse": 1.0},
        {"id_cond": 1, "n_sims": 10, "nrmse": float("nan")},
    ]
    assert aggregate_condition_rows(rows)["nrmse"] == pytest.approx(1.0)


def test_geometric_aggregation_rejects_non_positive_values() -> None:
    rows = [
        {"id_cond": 0, "n_sims": 10, "nrmse": 0.0},
        {"id_cond": 1, "n_sims": 10, "nrmse": 1.0},
    ]
    with pytest.raises(ValueError, match="positive"):
        aggregate_condition_rows(rows, aggregate="geometric")


@pytest.mark.parametrize(
    ("bad", "error"),
    [("median", ValueError), ("", ValueError), (1, TypeError),
     ({"nrmse": "median"}, ValueError)],
)
def test_normalize_aggregate_rejects_unknown_reducers(bad: object, error) -> None:
    with pytest.raises(error):
        normalize_aggregate(bad)


class _GridApproximator:
    """Return correctly shaped draws; custom metrics read the truths."""

    def sample(self, *, conditions, num_samples):
        n_sims = len(next(iter(conditions.values())))
        return {
            "a": np.zeros((n_sims, num_samples, 1)),
            "b": np.zeros((n_sims, num_samples, 1)),
        }


def _grid_dataset() -> ValidationDataset:
    simulations = [
        {"a": np.ones(4), "b": np.full(4, 5.0), "x": np.zeros((4, 1))},
        {"a": np.full(4, 3.0), "b": np.full(4, 7.0), "x": np.ones((4, 1))},
    ]
    return ValidationDataset(
        simulations=simulations,
        condition_labels=[{"c": 0}, {"c": 1}],
        param_keys=["a", "b"],
        data_keys=["x"],
        seed=0,
    )


def test_mapped_worst_reduces_full_parameter_condition_grid() -> None:
    def probe(draws, true_values):
        return {"grid_probe": float(np.mean(true_values))}

    register_metric("grid_probe", probe, overwrite=True)
    try:
        data = _grid_dataset()
        mapped = run_validation_pipeline(
            _GridApproximator(), data, n_posterior_samples=3,
            metrics=["grid_probe"], aggregate={"grid_probe": "worst"},
        )
        scalar = run_validation_pipeline(
            _GridApproximator(), data, n_posterior_samples=3,
            metrics=["grid_probe"], aggregate="worst",
        )
        assert mapped.summary["grid_probe"] == pytest.approx(7.0)
        assert scalar.summary["grid_probe"] == pytest.approx(5.0)
    finally:
        unregister_metric("grid_probe")


def test_mapped_geometric_reduces_full_parameter_condition_grid() -> None:
    def probe(draws, true_values):
        return {"grid_probe": float(np.mean(true_values))}

    register_metric("grid_probe", probe, overwrite=True)
    try:
        result = run_validation_pipeline(
            _GridApproximator(), _grid_dataset(), n_posterior_samples=3,
            metrics=["grid_probe"], aggregate={"grid_probe": "geometric"},
        )
        assert result.summary["grid_probe"] == pytest.approx(105 ** 0.25)
    finally:
        unregister_metric("grid_probe")


def test_joint_metric_reduces_once_per_condition_and_failed_key_is_absent() -> None:
    def joint(inputs: JointMetricInputs):
        return {"joint_probe": float(inputs.cond_id + 2)}

    def flaky(inputs: JointMetricInputs):
        if inputs.cond_id == 1:
            raise RuntimeError("bad condition")
        return {"joint_flaky": 0.1}

    register_joint_metric("joint_probe", joint, overwrite=True)
    register_joint_metric("joint_flaky", flaky, overwrite=True)
    try:
        result = run_validation_pipeline(
            _GridApproximator(), _grid_dataset(), n_posterior_samples=3,
            metrics=[],
            joint_metrics={"joint_probe": joint, "joint_flaky": flaky},
            aggregate={"joint_probe": "worst", "joint_flaky": "worst"},
        )
        assert result.summary["joint_probe"] == pytest.approx(3.0)
        assert "joint_flaky" not in result.summary
    finally:
        unregister_metric("joint_probe")
        unregister_metric("joint_flaky")


def test_worst_uses_min_for_higher_is_better_correlation_alias() -> None:
    rows = [
        {"id_cond": 0, "n_sims": 4, "correlation": 0.9},
        {"id_cond": 1, "n_sims": 4, "correlation": 0.2},
    ]
    assert aggregate_condition_rows(rows, aggregate={"corr": "worst"})[
        "correlation"
    ] == pytest.approx(0.2)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_geometric_domain_error_names_metric(value):
    with pytest.raises(ValueError, match="nrmse.*positive"):
        aggregate_condition_rows([{"nrmse": value}], {"nrmse": "geometric"})


@pytest.mark.parametrize("reduction", ["mean", "worst", "geometric"])
def test_all_nan_reductions_return_nan(reduction):
    summary = aggregate_condition_rows([{"nrmse": float("nan")}], reduction)
    assert np.isnan(summary["nrmse"])


def test_conflicting_alias_reductions_are_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_aggregate({"corr": "worst", "correlation": "mean"})


def test_geometric_mean_does_not_overflow_intermediate_product():
    summary = aggregate_condition_rows(
        [{"nrmse": 1e200}, {"nrmse": 1e200}], "geometric"
    )
    assert summary["nrmse"] == pytest.approx(1e200)


def test_intermediate_and_final_validation_use_same_grid_reduction():
    import optuna

    from bayesflow_hpo.optimization.objective import default_validate_fn
    from bayesflow_hpo.optimization.validation_callback import (
        PeriodicValidationCallback,
    )

    def probe(draws, true_values):
        return {"grid_probe": float(np.mean(true_values))}

    register_metric("grid_probe", probe, overwrite=True)
    try:
        data = _grid_dataset()
        aggregate = {"grid_probe": "worst"}
        callback = PeriodicValidationCallback(
            trial=optuna.create_study().ask(),
            approximator=_GridApproximator(), validation_data=data,
            n_posterior_samples=3, objective_metrics=["grid_probe"],
            aggregate=aggregate,
        )
        intermediate = callback._run_lightweight_validation()
        final = default_validate_fn(
            _GridApproximator(), data, 3, objective_metrics=["grid_probe"],
            aggregate=aggregate,
        )
        assert intermediate == {"grid_probe": 7.0}
        assert final["grid_probe"] == 7.0
    finally:
        unregister_metric("grid_probe")


def test_intermediate_validation_does_not_swallow_geometric_domain_errors():
    import optuna

    from bayesflow_hpo.optimization.validation_callback import (
        PeriodicValidationCallback,
    )
    from bayesflow_hpo.validation.metrics import AggregationError

    def probe(draws, true_values):
        return {"grid_probe": 0.0}

    register_metric("grid_probe", probe, overwrite=True)
    try:
        callback = PeriodicValidationCallback(
            trial=optuna.create_study().ask(),
            approximator=_GridApproximator(), validation_data=_grid_dataset(),
            n_posterior_samples=3, objective_metrics=["grid_probe"],
            aggregate={"grid_probe": "geometric"},
        )
        with pytest.raises(AggregationError, match="positive"):
            callback._run_lightweight_validation()
    finally:
        unregister_metric("grid_probe")
