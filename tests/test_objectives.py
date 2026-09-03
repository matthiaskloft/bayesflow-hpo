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
    register_metric_direction,
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
    """An unregistered missing metric defaults to +inf, not 1.0.

    Re-baselined with reasoning rather than to make the suite green. Nothing
    is known about an unregistered metric's scale, so no finite constant is
    defensible: with 1.0, a custom RMSE-like metric reporting 100.0 lost to a
    trial that reported nothing at all -- the same "missing beats bad"
    inversion found separately for `log_gamma` and `correlation`. Registering
    the metric is how a caller supplies a tighter bound.
    """
    metrics = {"summary": {"nrmse": 0.1}}
    values = extract_multi_objective_values(
        metrics,
        cost_score=0.5,
        objective_metrics=["nrmse", "nonexistent_key"],
        objective_mode="pareto",
    )
    assert values[1] == math.inf


def test_extract_multi_missing_correlation_returns_worst():
    """A missing correlation must be worse than any value it could report.

    This previously asserted 1.0, from a default raw correlation of 0.0. That
    was wrong: Pearson correlation runs [-1, 1], so a reported -0.5 maps to
    1.5 -- worse than the penalty. A missing value beat a genuinely negative
    one. The worst raw correlation is -1.0, giving 2.0.
    """
    metrics = {"summary": {"calibration_error": 0.05}}
    values = extract_multi_objective_values(
        metrics,
        cost_score=0.5,
        objective_metrics=["calibration_error", "correlation"],
        objective_mode="pareto",
    )
    assert values[1] == 2.0
    assert values[1] > _metric_to_minimize("correlation", -0.99)


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

    def test_an_unknown_metric_does_not_borrow_calibration_error(self) -> None:
        """Absence from the table establishes neither direction nor scale.

        This previously substituted `calibration_error` for any unregistered
        objective. A missing `custom_rmse` would take calibration_error's
        0.05, while a genuinely reported custom RMSE can be 100 -- so the
        missing metric wins. An earlier version of this test asserted the
        substitution, which enshrined the assumption rather than checking it.
        """
        metrics = {"summary": {"calibration_error": 0.05}}
        value, _ = extract_objective_values(metrics, 1.0, "custom_rmse")
        assert value == worst_objective_value("custom_rmse")
        assert value != pytest.approx(0.05)

    def test_legacy_higher_is_better_removal_is_honoured(self) -> None:
        """The old set controlled conversion by content, both ways.

        Supporting only additions would silently ignore a consumer that
        removed a built-in to make it pass through.
        """
        assert _metric_to_minimize("contraction", 0.8) == pytest.approx(0.2)
        HIGHER_IS_BETTER.discard("contraction")
        try:
            assert _metric_to_minimize("contraction", 0.8) == pytest.approx(0.8)
            # Removing it makes the metric unknown, and an unknown scale now
            # takes +inf rather than a finite 1.0 -- the conversion claim
            # above is what this test is about, and it is unchanged.
            assert worst_objective_value("contraction") == math.inf
        finally:
            HIGHER_IS_BETTER.add("contraction")
        assert _metric_to_minimize("contraction", 0.8) == pytest.approx(0.2)

    def test_register_metric_direction_round_trips(self) -> None:
        register_metric_direction(
            "my_unbounded_score",
            higher_is_better=True,
            worst_raw=-math.inf,
            to_minimize=lambda v: -v,
        )
        try:
            assert _metric_to_minimize("my_unbounded_score", 3.0) == -3.0
            assert math.isinf(worst_objective_value("my_unbounded_score"))
            assert "my_unbounded_score" in HIGHER_IS_BETTER
        finally:
            METRIC_DIRECTIONS.pop("my_unbounded_score", None)
            HIGHER_IS_BETTER.discard("my_unbounded_score")

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
        assert values[0] == pytest.approx(2.0)

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
        # Pearson runs [-1, 1], so its worst minimize-space value is 2.0.
        assert worst_objective_value("correlation") == 2.0
        assert worst_objective_value("contraction") == 1.0


class TestEncodingChangeSetIsDerived:
    """`ENCODING_CHANGED_AT_V2` must be computed, not remembered.

    Its first version listed `log_gamma` alone. That came from diffing the
    conversion function and stopping there -- but a trial stores a number for
    two reasons, and the *penalty* substituted for a missing metric moved as
    well. `correlation` went from an objective penalty of 1.0 to 2.0, so a
    legacy study's old failed trials outranked new ones while the guard waved
    it through. Reading the diff is exactly what failed; this computes it.
    """

    # The pre-change rule, reproduced from `git show d2e1d47`, the last commit
    # carrying the RELEASED 0.1.0 behaviour: conversion was `1 - value` for the
    # names in `HIGHER_IS_BETTER` and pass-through otherwise, and the
    # missing-metric substitute was `0.0 if key in HIGHER_IS_BETTER else 1.0`.
    #
    # This previously read `{"correlation", "contraction"}`, taken from
    # `b93501b~1` -- a commit that is NOT an ancestor of `main` and that sits
    # PART-WAY through the correction series, after `contraction` had already
    # been given its direction. Anchoring here made the derivation model
    # contraction as already-converted, compute "unchanged", and file it under
    # `ENCODING_UNCHANGED_AT_V2` -- so a real 0.1.0 contraction study passed
    # the resume guard and mixed raw values with `1 - value` in one column.
    # The baseline has to be the released version, not a waypoint.
    _OLD_HIGHER_IS_BETTER = {"correlation"}

    def _old_penalty(self, name: str) -> float:
        raw = 0.0 if name in self._OLD_HIGHER_IS_BETTER else 1.0
        return (1.0 - raw) if name in self._OLD_HIGHER_IS_BETTER else raw

    def _old_conversion(self, name: str, value: float) -> float:
        return (1.0 - value) if name in self._OLD_HIGHER_IS_BETTER else value

    def test_the_declared_set_matches_the_computed_one(self) -> None:
        from bayesflow_hpo.objectives import (
            ENCODING_CHANGED_AT_V2,
            METRIC_DIRECTIONS,
            _metric_to_minimize,
            worst_objective_value,
        )
        from bayesflow_hpo.validation.registry import _KINDS

        # Every metric a study could actually store an objective column for.
        candidates = {
            name for name, kind in _KINDS.items() if kind == "objective"
        } | set(METRIC_DIRECTIONS)

        computed = set()
        for name in candidates:
            penalty_moved = (
                self._old_penalty(name) != worst_objective_value(name)
            )
            probe = 0.3
            conversion_moved = abs(
                self._old_conversion(name, probe)
                - _metric_to_minimize(name, probe)
            ) > 1e-12
            if penalty_moved or conversion_moved:
                computed.add(name)

        assert computed == set(ENCODING_CHANGED_AT_V2), (
            "ENCODING_CHANGED_AT_V2 no longer matches the metrics whose "
            "stored values actually changed. Missing: "
            f"{sorted(computed - set(ENCODING_CHANGED_AT_V2))}; stale: "
            f"{sorted(set(ENCODING_CHANGED_AT_V2) - computed)}."
        )

    def test_the_audited_sets_cover_every_objective_capable_builtin(
        self,
    ) -> None:
        """Disjointness alone let an omission pass unnoticed.

        The round-8 test asserted only that the two sets do not overlap, so a
        built-in in NEITHER set was silent -- and anything omitted is treated
        as encoding-sensitive, blocking legacy studies that are in fact
        comparable. This asserts coverage over the same candidate set the
        derivation above uses: names a study could store a column for.
        """
        from bayesflow_hpo.objectives import (
            ENCODING_CHANGED_AT_V2,
            ENCODING_UNCHANGED_AT_V2,
            METRIC_DIRECTIONS,
        )
        from bayesflow_hpo.validation.registry import _KINDS

        candidates = {
            name for name, kind in _KINDS.items() if kind == "objective"
        } | set(METRIC_DIRECTIONS)
        audited = ENCODING_CHANGED_AT_V2 | ENCODING_UNCHANGED_AT_V2

        assert not (candidates - audited), (
            "These objective-capable built-ins are in neither audited set, so "
            "they are silently treated as encoding-sensitive: "
            f"{sorted(candidates - audited)}."
        )

    def test_the_deprecated_sbc_shim_is_deliberately_unaudited(self) -> None:
        """It produces no column of its own, so it needs no verdict.

        `sbc` is registered but delegates to `sbc_ks` and `sbc_chi2`, which
        are audited separately (and land in opposite sets). Pinning its kind
        keeps the coverage test above honest: if `sbc` ever became an
        objective, that test would start demanding a verdict for it.
        """
        from bayesflow_hpo.objectives import (
            ENCODING_CHANGED_AT_V2,
            ENCODING_UNCHANGED_AT_V2,
        )
        from bayesflow_hpo.validation.registry import _KINDS

        assert _KINDS["sbc"] == "diagnostic"
        assert "sbc" not in (ENCODING_CHANGED_AT_V2 | ENCODING_UNCHANGED_AT_V2)
        assert "sbc_ks" in ENCODING_UNCHANGED_AT_V2
        assert "sbc_chi2" in ENCODING_CHANGED_AT_V2

    def test_contraction_is_included_for_the_right_reason(self) -> None:
        """It was excluded because the baseline was a mid-series waypoint.

        Against released 0.1.0, `contraction` had no direction: it was
        minimized raw. Converting it to `1 - value` moves every stored number,
        so a legacy study's column is not comparable with a new one.
        """
        from bayesflow_hpo.objectives import (
            ENCODING_CHANGED_AT_V2,
            _metric_to_minimize,
        )

        assert "contraction" in ENCODING_CHANGED_AT_V2
        assert self._old_conversion("contraction", 0.3) == pytest.approx(0.3)
        assert _metric_to_minimize("contraction", 0.3) == pytest.approx(0.7)


class TestAliasesReachTheSummary:
    """The public extractors are keyed by canonical name.

    Found by giving `canonical_metric_name` a `NewType` return and letting
    mypy report every site that fed it a bare `str`. Both functions are
    exported in `__all__`, and both looked an alias up directly in a summary
    that is emitted under canonical names: the lookup missed and the
    worst-case penalty stood in for the real value, with no error raised.
    For an unregistered spelling that penalty is `+inf`, so every trial tied
    at the worst possible score and the objective went flat.
    """

    _SUMMARY = {"calibration_error": 0.02, "nrmse": 0.1}

    def test_the_single_extractor_resolves_an_alias(self) -> None:
        from bayesflow_hpo.objectives import extract_objective_values

        canonical = extract_objective_values(
            self._SUMMARY, 1.0, "calibration_error"
        )
        alias = extract_objective_values(self._SUMMARY, 1.0, "cal_error")
        assert alias == canonical
        assert math.isfinite(alias[0])

    def test_the_multi_extractor_resolves_an_alias(self) -> None:
        from bayesflow_hpo.objectives import extract_multi_objective_values

        canonical = extract_multi_objective_values(
            self._SUMMARY, 1.0, ["calibration_error", "nrmse"], "pareto"
        )
        alias = extract_multi_objective_values(
            self._SUMMARY, 1.0, ["cal_error", "nrmse"], "pareto"
        )
        assert alias == canonical
        assert all(math.isfinite(v) for v in alias)

    def test_an_unknown_name_still_takes_the_penalty(self) -> None:
        """Resolving aliases must not turn into accepting anything."""
        from bayesflow_hpo.objectives import extract_objective_values

        value, _ = extract_objective_values(
            self._SUMMARY, 1.0, "not_a_metric_at_all"
        )
        assert value == math.inf


class TestTheScoreSpacesCannotBeMixed:
    """The raw/minimize distinction is enforced statically, not by review.

    Eight review rounds caught instances of this pair being confused by
    reading the code. The two spaces are indistinguishable at runtime -- both
    are plain floats -- so nothing raises when they are swapped; a
    higher-is-better metric simply ranks backwards. These `NewType`s make the
    swap a type error, and this test is what fails if either signature is
    later widened back to a bare `float`.
    """

    _SNIPPET = """
from bayesflow_hpo.objectives import (
    _metric_to_minimize,
    worst_objective_value,
    worst_raw_value,
)
from bayesflow_hpo.validation.registry import canonical_metric_name

k = canonical_metric_name("log_gamma")
_metric_to_minimize(k, worst_raw_value(k))
_metric_to_minimize(k, worst_objective_value(k))
"""

    def _run_mypy(self, tmp_path):
        import subprocess
        import sys

        src = tmp_path / "snippet.py"
        src.write_text(self._SNIPPET, encoding="utf-8")
        # `--python-version` must match the interpreter running the tests, NOT
        # the project's pinned 3.11. Inheriting the pin made mypy parse this
        # interpreter's own numpy stubs under 3.11 rules, and on 3.12+ those
        # use `type` statements: mypy aborted with a syntax error before it
        # reached the snippet, and the test read that empty result as "the bad
        # call was accepted".
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        return subprocess.run(
            [
                sys.executable, "-m", "mypy",
                "--no-error-summary", f"--python-version={version}", str(src),
            ],
            capture_output=True, text=True,
        )

    def test_a_minimize_value_in_a_raw_slot_is_a_type_error(
        self, tmp_path
    ) -> None:
        """The round-2 P1 on PR #72, now unrepresentable.

        `worst_objective_value` is minimize-space; the second argument of
        `_metric_to_minimize` is consumed before conversion. Feeding one to
        the other double-converts it, which for `log_gamma` turned a penalty
        into a value that beat every genuine result.
        """
        pytest.importorskip("mypy")
        result = self._run_mypy(tmp_path)

        # Exit 2 is a mypy crash or config error, not a clean verdict. Without
        # this the test passes vacuously whenever mypy fails to run at all.
        assert result.returncode == 1, (
            "expected mypy to report type errors (exit 1), not to fail to "
            f"run (exit {result.returncode}):\n{result.stdout}{result.stderr}"
        )
        assert "[syntax]" not in result.stdout, (
            f"mypy could not parse its inputs:\n{result.stdout}"
        )

        # Line 10 is the worst_raw_value call (correct); line 11 the
        # worst_objective_value one (the bug).
        assert "snippet.py:11" in result.stdout, (
            "expected the minimize-space argument to be rejected; got:\n"
            f"{result.stdout}{result.stderr}"
        )
        assert "snippet.py:10" not in result.stdout, (
            "the raw-space call is correct and must still type-check; got:\n"
            f"{result.stdout}{result.stderr}"
        )


class TestBothSpellingsOfASummaryResolve:
    """Canonicalizing one side alone fixes one shape and breaks another.

    The pipeline emits canonical names, but a caller's own `validate_fn`
    emits whatever spelling its author used -- and both reach these
    extractors. Resolving only the requested name made an alias-keyed summary
    read with that same alias miss, which had worked before.
    """

    _CANONICAL = {"calibration_error": 0.02, "nrmse": 0.1}
    _ALIASED = {"cal_error": 0.02, "nrmse": 0.1}

    @pytest.mark.parametrize("summary_style", ["canonical", "aliased"])
    @pytest.mark.parametrize("requested", ["calibration_error", "cal_error"])
    def test_every_spelling_combination_finds_the_value(
        self, summary_style: str, requested: str
    ) -> None:
        from bayesflow_hpo.objectives import extract_objective_values

        summary = (
            self._CANONICAL if summary_style == "canonical" else self._ALIASED
        )
        value, _ = extract_objective_values(summary, 1.0, requested)
        assert value == 0.02

    @pytest.mark.parametrize("requested", ["calibration_error", "cal_error"])
    def test_the_multi_extractor_too(self, requested: str) -> None:
        from bayesflow_hpo.objectives import extract_multi_objective_values

        values = extract_multi_objective_values(
            self._ALIASED, 1.0, [requested, "nrmse"], "pareto"
        )
        assert values[0] == 0.02

    def test_a_summary_holding_both_keeps_the_canonical_value(self) -> None:
        """The pipeline's own measurement wins over a caller's spelling.

        Re-keying by canonical name collides when a summary carries both. The
        canonical entry is what the pipeline wrote, so letting the alias
        overwrite it would let a spelling override a measurement.
        """
        from bayesflow_hpo.objectives import extract_objective_values

        both = {"calibration_error": 0.02, "cal_error": 0.99, "nrmse": 0.1}
        for requested in ("calibration_error", "cal_error"):
            value, _ = extract_objective_values(both, 1.0, requested)
            assert value == 0.02

    def test_non_metric_entries_survive_the_rekeying(self) -> None:
        """Unknown names pass through, so a summary is not filtered."""
        from bayesflow_hpo.objectives import canonical_summary

        out = canonical_summary(
            {"cal_error": 0.02, "n_datasets": 500, "some_note": "x"}
        )
        assert out == {
            "calibration_error": 0.02, "n_datasets": 500, "some_note": "x"
        }


class TestCollisionResolutionIsUniformAndOrderFree:
    """Every boundary that re-keys must use the same rule.

    `canonical_summary` resolved a both-spellings collision to the canonical
    entry, but two earlier boundaries -- `_validate_metric_keys` and the
    periodic-validation callback -- used a plain comprehension, which is
    last-write-wins. In the real optimization path those run FIRST, so the
    advertised rule did not hold and the trial's score depended on the
    insertion order of a hook's output dict.
    """

    _CANONICAL_FIRST = {
        "calibration_error": 0.02, "cal_error": 0.99, "nrmse": 0.1,
    }
    _ALIAS_FIRST = {
        "cal_error": 0.99, "calibration_error": 0.02, "nrmse": 0.1,
    }

    @pytest.mark.parametrize("order", ["canonical_first", "alias_first"])
    def test_validate_metric_keys_is_order_independent(
        self, order: str
    ) -> None:
        from bayesflow_hpo.optimization.objective import _validate_metric_keys

        raw = (
            self._CANONICAL_FIRST if order == "canonical_first"
            else self._ALIAS_FIRST
        )
        cleaned = _validate_metric_keys(
            dict(raw), ["calibration_error", "nrmse"]
        )
        assert cleaned["calibration_error"] == 0.02

    @pytest.mark.parametrize("order", ["canonical_first", "alias_first"])
    def test_the_extractor_agrees_with_it(self, order: str) -> None:
        """The two boundaries must not disagree about the same input."""
        from bayesflow_hpo.objectives import canonical_summary

        raw = (
            self._CANONICAL_FIRST if order == "canonical_first"
            else self._ALIAS_FIRST
        )
        assert canonical_summary(raw)["calibration_error"] == 0.02

    # The fourth boundary, `check_pipeline`, is covered in
    # tests/test_pipeline.py::TestPreflightCollisionResolutionIsOrderFree
    # instead. Asserting on `canonical_summary` here would pass even if that
    # boundary reverted to its last-write-wins comprehension, so the test has
    # to drive the pre-flight path itself and needs that module's doubles.
