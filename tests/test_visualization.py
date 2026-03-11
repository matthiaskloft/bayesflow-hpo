"""Tests for result visualization functions."""

from __future__ import annotations

import datetime

import matplotlib
import matplotlib.pyplot as plt
import optuna
import pytest

matplotlib.use("Agg")

from bayesflow_hpo.results.visualization import (
    _pareto_front_2d,
    _trained_trials,
    plot_metric_panels,
    plot_metric_scatter,
    plot_optimization_history,
    plot_param_importance,
    plot_pareto_front,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Ensure all matplotlib figures are closed after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trial(
    number: int,
    values: list[float],
    user_attrs: dict | None = None,
    state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE,
) -> optuna.trial.FrozenTrial:
    """Build a FrozenTrial for testing."""
    now = datetime.datetime.now()
    return optuna.trial.FrozenTrial(
        number=number,
        state=state,
        value=None,
        values=values,
        datetime_start=now,
        datetime_complete=now,
        params={"lr": 1e-3, "depth": 4},
        distributions={
            "lr": optuna.distributions.FloatDistribution(1e-5, 1e-1, log=True),
            "depth": optuna.distributions.IntDistribution(1, 8),
        },
        user_attrs=user_attrs or {},
        system_attrs={},
        intermediate_values={},
        trial_id=number,
    )


@pytest.fixture
def multi_objective_study() -> optuna.Study:
    """Multi-objective study with 5 trials (1 rejected) and metric_names."""
    study = optuna.create_study(
        directions=["minimize", "minimize"],
    )
    # Set metric names (Optuna >=4.x attribute)
    study._metric_names = ["mean(calibration_error+nrmse)", "param_count_norm"]

    trials = [
        _make_trial(0, [0.10, 0.3], {
            "param_count": 50_000,
            "calibration_error": 0.08,
            "nrmse": 0.12,
        }),
        _make_trial(1, [0.08, 0.5], {
            "param_count": 100_000,
            "calibration_error": 0.06,
            "nrmse": 0.10,
        }),
        _make_trial(2, [0.15, 0.2], {
            "param_count": 30_000,
            "calibration_error": 0.12,
            "nrmse": 0.18,
        }),
        _make_trial(3, [0.12, 0.4], {
            "param_count": 80_000,
            "calibration_error": 0.10,
            "nrmse": 0.14,
        }),
        # Rejected trial
        _make_trial(4, [0.50, 0.9], {
            "param_count": 200_000,
            "rejected_reason": "exceeded param budget",
        }),
    ]
    for t in trials:
        study.add_trial(t)
    return study


@pytest.fixture
def single_objective_study() -> optuna.Study:
    """Single-objective study."""
    study = optuna.create_study(direction="minimize")
    study.add_trial(_make_trial(0, [0.05], {"param_count": 50_000}))
    return study


@pytest.fixture
def empty_study() -> optuna.Study:
    """Study with no completed trials."""
    return optuna.create_study(directions=["minimize", "minimize"])


# ---------------------------------------------------------------------------
# _pareto_front_2d tests
# ---------------------------------------------------------------------------

class TestParetoFront2D:
    def test_basic(self):
        xs = [1.0, 2.0, 3.0, 1.5]
        ys = [3.0, 1.0, 2.0, 2.0]
        front = _pareto_front_2d(xs, ys)
        front_points = sorted((xs[i], ys[i]) for i in front)
        # Non-dominated front: (1.0,3.0), (1.5,2.0), (2.0,1.0)
        # (3.0, 2.0) is dominated by (2.0, 1.0)
        assert front_points == [(1.0, 3.0), (1.5, 2.0), (2.0, 1.0)]

    def test_empty(self):
        assert _pareto_front_2d([], []) == []

    def test_single_point(self):
        assert _pareto_front_2d([1.0], [2.0]) == [0]

    def test_all_dominated_by_one(self):
        # Point 0 dominates all others
        xs = [1.0, 2.0, 3.0]
        ys = [1.0, 2.0, 3.0]
        front = _pareto_front_2d(xs, ys)
        assert front == [0]

    def test_no_dominance(self):
        # Perfect trade-off: as x increases, y decreases
        xs = [1.0, 2.0, 3.0]
        ys = [3.0, 2.0, 1.0]
        front = _pareto_front_2d(xs, ys)
        assert len(front) == 3


# ---------------------------------------------------------------------------
# _trained_trials tests
# ---------------------------------------------------------------------------

class TestTrainedTrials:
    def test_filters_rejected(self, multi_objective_study):
        trained = _trained_trials(multi_objective_study)
        assert len(trained) == 4  # 5 total, 1 rejected

    def test_empty_study(self, empty_study):
        assert _trained_trials(empty_study) == []


# ---------------------------------------------------------------------------
# plot_pareto_front tests
# ---------------------------------------------------------------------------

class TestPlotParetoFront:
    def test_returns_axes(self, multi_objective_study):
        ax = plot_pareto_front(multi_objective_study)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_xlabel_auto_derived(self, multi_objective_study):
        ax = plot_pareto_front(multi_objective_study)
        assert ax.get_xlabel() == "mean(calibration_error+nrmse)"

    def test_xlabel_override(self, multi_objective_study):
        ax = plot_pareto_front(multi_objective_study, xlabel="Custom X")
        assert ax.get_xlabel() == "Custom X"

    def test_single_objective(self, single_objective_study):
        ax = plot_pareto_front(single_objective_study)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_no_trained_trials(self, empty_study):
        ax = plot_pareto_front(empty_study)
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# plot_optimization_history tests
# ---------------------------------------------------------------------------

class TestPlotOptimizationHistory:
    def test_returns_axes(self, multi_objective_study):
        ax = plot_optimization_history(multi_objective_study)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_ylabel_auto_derived(self, multi_objective_study):
        ax = plot_optimization_history(multi_objective_study)
        assert ax.get_ylabel() == "mean(calibration_error+nrmse)"

    def test_empty_study(self, empty_study):
        ax = plot_optimization_history(empty_study)
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# plot_metric_scatter tests
# ---------------------------------------------------------------------------

class TestPlotMetricScatter:
    def test_returns_axes(self, multi_objective_study):
        ax = plot_metric_scatter(
            multi_objective_study, "calibration_error", "nrmse",
        )
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_labels(self, multi_objective_study):
        ax = plot_metric_scatter(
            multi_objective_study, "calibration_error", "nrmse",
        )
        assert ax.get_xlabel() == "calibration_error"
        assert ax.get_ylabel() == "nrmse"

    def test_iso_lines_auto_detected(self, multi_objective_study):
        # metric_names[0] starts with "mean(" -> iso lines should appear
        ax = plot_metric_scatter(
            multi_objective_study, "calibration_error", "nrmse",
        )
        # Check that grey lines were drawn (iso-mean)
        lines = [line for line in ax.get_lines() if line.get_color() == "grey"]
        assert len(lines) > 0

    def test_iso_lines_disabled(self, multi_objective_study):
        ax = plot_metric_scatter(
            multi_objective_study, "calibration_error", "nrmse",
            show_iso_lines=False,
        )
        lines = [line for line in ax.get_lines() if line.get_color() == "grey"]
        assert len(lines) == 0

    def test_missing_metrics(self, multi_objective_study):
        ax = plot_metric_scatter(
            multi_objective_study, "nonexistent_x", "nonexistent_y",
        )
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# plot_metric_panels tests
# ---------------------------------------------------------------------------

class TestPlotMetricPanels:
    def test_returns_axes_array(self, multi_objective_study):
        axes = plot_metric_panels(
            multi_objective_study,
            metrics=["calibration_error", "nrmse"],
        )
        assert hasattr(axes, "__len__")
        assert len(axes) == 2

    def test_auto_detect_metrics(self, multi_objective_study):
        axes = plot_metric_panels(multi_objective_study)
        # Should auto-detect calibration_error and nrmse
        assert hasattr(axes, "__len__")
        assert len(axes) >= 2

    def test_empty_study(self, empty_study):
        ax = plot_metric_panels(empty_study)
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# plot_param_importance tests
# ---------------------------------------------------------------------------

class TestPlotParamImportance:
    def test_returns_axes(self, multi_objective_study):
        ax = plot_param_importance(multi_objective_study)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_target_name_title(self, multi_objective_study):
        ax = plot_param_importance(
            multi_objective_study, target_name="calibration_error",
        )
        # Title shows metric name when importance succeeds, or fallback text
        title = ax.get_title()
        assert "calibration_error" in title or title == ""

    def test_default_title(self, multi_objective_study):
        ax = plot_param_importance(multi_objective_study)
        # Title is "Parameter importance" when importance succeeds,
        # empty when Optuna falls back to "Importance unavailable" text
        title = ax.get_title()
        assert title in ("Parameter importance", "")
