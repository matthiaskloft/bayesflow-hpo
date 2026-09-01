"""Tests for objective extraction and param normalization."""

import math

import numpy as np
import pytest

from bayesflow_hpo.objectives import (
    HIGHER_IS_BETTER,
    MAX_PARAM_COUNT,
    METRIC_DIRECTIONS,
    MIN_PARAM_COUNT,
    _metric_to_minimize,
    compute_inference_time_per_dataset,
    denormalize_param_count,
    extract_multi_objective_values,
    extract_objective_values,
    mean_objective_score,
    normalize_param_count,
    worst_objective_value,
)


def test_metric_to_minimize_lower_is_better():
    assert _metric_to_minimize("calibration_error", 0.05) == 0.05
    assert _metric_to_minimize("nrmse", 0.2) == 0.2


def test_metric_to_minimize_higher_is_better():
    assert np.isclose(_metric_to_minimize("correlation", 0.8), 0.2)
    assert np.isclose(_metric_to_minimize("contraction", 0.8), 0.2)


def test_extract_multi_mean_mode():
    metrics = {
        "summary": {
            "calibration_error": 0.04,
            "nrmse": 0.10,
        }
    }
    values = extract_multi_objective_values(
        metrics,
        cost_score=0.5,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="mean",
    )
    assert len(values) == 2
    assert np.isclose(values[0], np.mean([0.04, 0.10]))
    assert values[1] == 0.5  # cost_score passed through


def test_extract_multi_pareto_mode():
    metrics = {
        "summary": {
            "calibration_error": 0.04,
            "nrmse": 0.10,
        }
    }
    values = extract_multi_objective_values(
        metrics,
        cost_score=0.5,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="pareto",
    )
    assert len(values) == 3
    assert values[0] == 0.04  # calibration_error
    assert values[1] == 0.10  # nrmse
    assert values[2] == 0.5  # cost_score passed through


def test_extract_multi_with_correlation():
    """Correlation is higher-is-better, so objective = 1 - corr."""
    metrics = {
        "summary": {
            "calibration_error": 0.05,
            "nrmse": 0.15,
            "correlation": 0.9,
        }
    }
    values = extract_multi_objective_values(
        metrics,
        cost_score=0.5,
        objective_metrics=["calibration_error", "nrmse", "correlation"],
        objective_mode="pareto",
    )
    assert len(values) == 4
    assert values[0] == 0.05
    assert values[1] == 0.15
    assert np.isclose(values[2], 0.1)  # 1 - 0.9


def test_extract_multi_missing_metric_returns_worst():
    """Missing lower-is-better metric defaults to 1.0 (worst)."""
    metrics = {"summary": {"nrmse": 0.1}}
    values = extract_multi_objective_values(
        metrics,
        cost_score=0.5,
        objective_metrics=["nrmse", "nonexistent_key"],
        objective_mode="pareto",
    )
    assert values[1] == 1.0  # fallback for missing lower-is-better key


def test_extract_multi_missing_correlation_returns_worst():
    """Missing higher-is-better metric should default to worst (1.0 after inversion)."""
    metrics = {"summary": {"calibration_error": 0.05}}
    values = extract_multi_objective_values(
        metrics,
        cost_score=0.5,
        objective_metrics=["calibration_error", "correlation"],
        objective_mode="pareto",
    )
    # correlation missing -> default 0.0 -> _metric_to_minimize: 1.0 - 0.0 = 1.0
    assert values[1] == 1.0


def test_extract_legacy_applies_metric_to_minimize():
    """Legacy extract_objective_values inverts higher-is-better metrics."""
    metrics = {"summary": {"correlation": 0.9}}
    obj_val, _ = extract_objective_values(
        metrics,
        cost_score=0.5,
        objective_metric="correlation",
    )
    assert np.isclose(obj_val, 0.1)  # 1 - 0.9


def test_extract_legacy_lower_is_better_unchanged():
    """Legacy path passes lower-is-better metrics through unchanged."""
    metrics = {"summary": {"calibration_error": 0.05}}
    obj_val, _ = extract_objective_values(
        metrics,
        cost_score=0.5,
        objective_metric="calibration_error",
    )
    assert obj_val == 0.05


def test_extract_multi_rejects_unknown_mode():
    metrics = {"summary": {"calibration_error": 0.05}}
    with pytest.raises(ValueError, match="Unknown objective_mode"):
        extract_multi_objective_values(
            metrics,
            cost_score=0.5,
            objective_metrics=["calibration_error"],
            objective_mode="weighted",
        )


# --- mean_objective_score tests ---


def test_mean_objective_score_multi_value_pareto_shape():
    """Pareto-shaped values: mean of all metrics, excluding the last (cost)."""
    values = (0.10, 0.30, 0.99)  # cal_err, nrmse, cost
    assert np.isclose(mean_objective_score(values), 0.20)


def test_mean_objective_score_two_value_mean_shape():
    """Mean-mode shape (metric, cost) reduces to the metric itself."""
    values = (0.30, 0.80)
    assert mean_objective_score(values) == 0.30


def test_mean_objective_score_single_value():
    """Single-element tuple returns that element directly."""
    values = (0.42,)
    assert mean_objective_score(values) == 0.42


# --- normalize_param_count tests ---


def test_normalize_default_range():
    """Endpoints of the default [1K, 1M] range map to 0.0 and 1.0."""
    assert normalize_param_count(MIN_PARAM_COUNT) == 0.0
    assert normalize_param_count(MAX_PARAM_COUNT) == 1.0


def test_normalize_mid_range():
    """A value inside the range should produce a score in (0, 1)."""
    score = normalize_param_count(100_000)
    assert 0.0 < score < 1.0


def test_normalize_auto_tightens_with_small_max():
    """When max_count < default MAX, min is auto-tightened to max/100."""
    # With max=100K, min auto-tightens to 1K (same as default here).
    # But with max=500K, min auto-tightens to 5K.
    low = normalize_param_count(10_000, max_count=500_000)
    high = normalize_param_count(400_000, max_count=500_000)
    assert low < 0.3, f"Low param count should map near 0, got {low}"
    assert high > 0.8, f"High param count should map near 1, got {high}"


def test_normalize_auto_tighten_spreads_values():
    """Auto-tightening should spread normalized values across [0, 1]."""
    # Without auto-tightening (max=1M default), all values in [70K, 150K]
    # cluster near 0.93-1.0.  With max=200K, they should spread out.
    scores = [
        normalize_param_count(p, max_count=200_000)
        for p in [5_000, 50_000, 100_000, 180_000]
    ]
    spread = max(scores) - min(scores)
    assert spread > 0.5, f"Expected spread > 0.5, got {spread}"


def test_normalize_zero_or_negative_returns_worst():
    assert normalize_param_count(0) == 1.0
    assert normalize_param_count(-10) == 1.0


def test_normalize_explicit_min_skips_auto_tighten():
    """When caller passes a non-default min_count, auto-tightening is skipped."""
    # min=500 is not the default 1000, so no auto-tightening
    score = normalize_param_count(500, min_count=500, max_count=100_000)
    assert score == 0.0  # at the lower bound


def test_normalize_raises_on_max_le_min():
    """Contradictory bounds (max < min) raise ValueError."""
    with pytest.raises(ValueError, match="max_count.*must be greater"):
        normalize_param_count(500, min_count=100, max_count=10)


def test_normalize_raises_on_max_eq_min():
    """Equal bounds raise ValueError."""
    with pytest.raises(ValueError, match="max_count.*must be greater"):
        normalize_param_count(100, min_count=100, max_count=100)


def test_normalize_raises_after_auto_tightening():
    """max_count=1 auto-tightens min_count to max(1, 0)=1, producing equal bounds."""
    with pytest.raises(ValueError, match="max_count.*must be greater"):
        normalize_param_count(1, max_count=1)


def test_denormalize_raises_on_max_eq_min():
    """denormalize_param_count raises ValueError when max_count equals min_count."""
    with pytest.raises(ValueError, match="max_count.*must be greater"):
        denormalize_param_count(0.5, min_count=100, max_count=100)


def test_denormalize_raises_on_max_lt_min():
    """denormalize_param_count raises ValueError when max_count < min_count."""
    with pytest.raises(ValueError, match="max_count.*must be greater"):
        denormalize_param_count(0.5, min_count=100, max_count=10)


def test_denormalize_round_trips_with_auto_tightened_max_count():
    """denormalize_param_count must mirror normalize_param_count's
    auto-tightening so round-tripping a custom max_count recovers the
    original param count."""
    normalized = normalize_param_count(3000, max_count=5000)
    round_tripped = denormalize_param_count(normalized, max_count=5000)
    assert abs(round_tripped - 3000) <= 1


# --- compute_inference_time_per_dataset tests ---


def test_inference_time_per_dataset_normal():
    """Average = total / n_datasets."""
    result = compute_inference_time_per_dataset(10.0, n_datasets=5)
    assert np.isclose(result, 2.0)


def test_inference_time_per_dataset_single():
    """Single dataset returns the total time."""
    result = compute_inference_time_per_dataset(3.5, n_datasets=1)
    assert np.isclose(result, 3.5)


def test_inference_time_per_dataset_zero_datasets():
    """n_datasets=0 uses max(0, 1)=1 to avoid division by zero."""
    result = compute_inference_time_per_dataset(5.0, n_datasets=0)
    assert np.isclose(result, 5.0)


class TestLogGammaDirection:
    """`log_gamma` runs opposite to every other calibration metric.

    BayesFlow's `calibration_log_gamma` returns log(gamma/gamma_null) and
    documents `log_gamma < 0` as rejecting the hypothesis of uniform ranks at
    the 5% level. Larger is better. Before this was recorded, `log_gamma`
    passed through `_metric_to_minimize` unchanged, so Optuna minimized it and
    would have selected the most miscalibrated model in the study -- silently,
    since nothing in the output looks wrong.
    """

    def test_a_well_calibrated_model_beats_a_miscalibrated_one(self):
        good = _metric_to_minimize("log_gamma", 1.5)
        bad = _metric_to_minimize("log_gamma", -25.5)
        assert good < bad, (
            "Optuna minimizes, so the better-calibrated model must map to the "
            "SMALLER objective value"
        )

    def test_conversion_is_negation_not_one_minus(self):
        """`1 - v` is meaningless for an unbounded log-ratio."""
        assert _metric_to_minimize("log_gamma", -25.5) == 25.5
        assert _metric_to_minimize("log_gamma", 1.5) == -1.5

    def test_bounded_metrics_keep_the_one_minus_conversion(self):
        """The scales differ; correlation must not change behaviour."""
        assert np.isclose(_metric_to_minimize("correlation", 0.8), 0.2)
        assert np.isclose(_metric_to_minimize("contraction", 0.8), 0.2)

    def test_log_gamma_is_reported_as_higher_is_better(self):
        assert "log_gamma" in HIGHER_IS_BETTER
        assert METRIC_DIRECTIONS["log_gamma"].higher_is_better is True

    def test_unknown_metrics_pass_through_unchanged(self):
        """Error-style metrics have no entry and must stay as they are."""
        assert _metric_to_minimize("some_custom_error", 0.3) == 0.3


class TestMissingMetricDefaults:
    def test_a_missing_log_gamma_does_not_look_excellent(self):
        """The old cross-metric fallback made a missing metric a winner.

        `extract_objective_values` used to substitute `calibration_error` for
        any missing objective. For `log_gamma` that is catastrophic: a good
        calibration_error of 0.05 becomes an objective of -0.05, which under
        minimization beats every real log_gamma a trial could report. A trial
        that failed to produce the metric would outrank every trial that did.
        """
        metrics = {"summary": {"calibration_error": 0.05, "nrmse": 0.2}}
        value, _ = extract_objective_values(metrics, 1.0, "log_gamma")
        assert value == worst_objective_value("log_gamma")
        assert value > _metric_to_minimize("log_gamma", -50.0)

    def test_the_fallback_survives_for_same_direction_metrics(self):
        """An unknown lower-is-better metric keeps the historical behaviour."""
        metrics = {"summary": {"calibration_error": 0.05}}
        value, _ = extract_objective_values(metrics, 1.0, "some_custom_error")
        assert value == pytest.approx(0.05)

    def test_missing_worst_case_is_not_converted_twice(self):
        """`worst_objective_value` is already in minimize space."""
        metrics = {"summary": {"nrmse": 0.2}}
        values = extract_multi_objective_values(
            metrics, 1.0, ["log_gamma", "nrmse"], objective_mode="pareto"
        )
        assert values[0] == worst_objective_value("log_gamma")
        assert values[0] > 0, "a double negation would flip this negative"

    def test_missing_correlation_still_maps_to_its_worst(self):
        metrics = {"summary": {"nrmse": 0.2}}
        values = extract_multi_objective_values(
            metrics, 1.0, ["correlation", "nrmse"], objective_mode="pareto"
        )
        assert values[0] == pytest.approx(1.0)

    def test_a_catastrophic_value_still_ranks_better_than_a_missing_one(self):
        """A finite penalty can be beaten, which inverts the intent.

        log_gamma is unbounded below, so no finite constant is provably its
        worst value. With a penalty of 1e3, a real log_gamma of -5000 would
        score worse than a missing one -- i.e. failing to report the metric
        would look better than reporting a catastrophic value.
        """
        assert _metric_to_minimize("log_gamma", -5000.0) < worst_objective_value(
            "log_gamma"
        )

    @pytest.mark.parametrize("metric", ["sbc_ks", "sbc_chi2"])
    def test_sbc_tests_take_their_own_penalty_not_calibration_errors(
        self, metric
    ):
        """Built-in metrics must not silently borrow another's value."""
        metrics = {"summary": {"calibration_error": 0.05, "nrmse": 0.2}}
        value, _ = extract_objective_values(metrics, 1.0, metric)
        assert value == worst_objective_value(metric)
        assert value != pytest.approx(0.05)

    @pytest.mark.parametrize("metric", ["sbc_ks", "sbc_chi2"])
    def test_sbc_tests_pass_through_unchanged_when_present(self, metric):
        """They are lower-is-better; the entry must not flip them."""
        assert _metric_to_minimize(metric, 0.3) == 0.3

    @pytest.mark.parametrize("metric", ["sbc_ks", "sbc_chi2"])
    def test_sbc_tests_missing_in_multi_objective(self, metric):
        metrics = {"summary": {"nrmse": 0.2}}
        values = extract_multi_objective_values(
            metrics, 1.0, [metric, "nrmse"], objective_mode="pareto"
        )
        assert values[0] == worst_objective_value(metric)

    def test_only_the_bounded_metrics_get_a_finite_worst_case(self):
        """KS is a sup of a CDF difference; chi-squared is unbounded above."""
        assert worst_objective_value("sbc_ks") == 1.0
        assert math.isinf(worst_objective_value("sbc_chi2"))
        assert math.isinf(worst_objective_value("log_gamma"))
        assert worst_objective_value("correlation") == 1.0
