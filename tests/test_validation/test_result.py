"""Tests for ValidationResult dataclass."""

import pandas as pd

from bayesflow_hpo.validation.result import ValidationResult


def _make_result() -> ValidationResult:
    cond_df = pd.DataFrame(
        [
            {"id_cond": 0, "n_sims": 100, "calibration_error": 0.05, "rmse": 0.2},
            {"id_cond": 1, "n_sims": 100, "calibration_error": 0.07, "rmse": 0.3},
        ]
    )
    summary = {"calibration_error": 0.06, "rmse": 0.25}
    return ValidationResult(
        condition_metrics=cond_df,
        summary=summary,
        timing={"inference": 1.0, "metrics": 0.5},
        n_conditions=2,
        n_posterior_samples=500,
        metric_names=["calibration_error", "rmse"],
    )


def test_summary_table_single_row():
    result = _make_result()
    table = result.summary_table()
    assert len(table) == 1
    assert "calibration_error" in table.columns


def test_condition_table_full():
    result = _make_result()
    table = result.condition_table()
    assert len(table) == 2


def test_condition_table_filtered():
    result = _make_result()
    table = result.condition_table(metric="rmse")
    assert "rmse" in table.columns
    assert "calibration_error" not in table.columns


def test_objective_scalar_default():
    result = _make_result()
    assert result.objective_scalar("calibration_error") == 0.06


def test_objective_scalar_fallback():
    result = ValidationResult(
        condition_metrics=pd.DataFrame(),
        summary={"mean_cal_error": 0.1},
        n_conditions=0,
        n_posterior_samples=0,
    )
    assert result.objective_scalar("missing_key") == 0.1


def test_objective_scalar_final_fallback():
    result = ValidationResult(
        condition_metrics=pd.DataFrame(),
        summary={},
        n_conditions=0,
        n_posterior_samples=0,
    )
    assert result.objective_scalar("missing") == 1.0


def test_parameter_table_none_when_single_param():
    result = _make_result()
    assert result.parameter_table() is None


def test_parameter_table_multi_param():
    child = ValidationResult(
        condition_metrics=pd.DataFrame(),
        summary={"calibration_error": 0.03},
        n_conditions=1,
        n_posterior_samples=100,
    )
    result = ValidationResult(
        condition_metrics=pd.DataFrame(),
        summary={"calibration_error": 0.03},
        per_parameter={"theta": child, "sigma": child},
        n_conditions=1,
        n_posterior_samples=100,
    )
    table = result.parameter_table()
    assert table is not None
    assert len(table) == 2
    assert "parameter" in table.columns


def test_repr_contains_summary():
    result = _make_result()
    text = repr(result)
    assert "n_conditions=2" in text
    assert "calibration_error" in text
