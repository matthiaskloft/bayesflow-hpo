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
    plot_parallel_coordinates,
    plot_param_importance,
    plot_pareto_3d,
    plot_pareto_front,
    plot_pareto_projections,
    plot_study,
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


@pytest.fixture
def three_objective_study() -> optuna.Study:
    """3-objective study with 6 trials and realistic values."""
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
    )
    study._metric_names = [
        "calibration_error", "nrmse", "param_count_norm",
    ]

    trials = [
        # Trial 0: good calibration, moderate nrmse, small model
        _make_trial(0, [0.05, 0.12, 0.2], {
            "param_count": 30_000,
            "calibration_error": 0.05,
            "nrmse": 0.12,
        }),
        # Trial 1: best calibration, worst nrmse, large model
        _make_trial(1, [0.03, 0.18, 0.6], {
            "param_count": 120_000,
            "calibration_error": 0.03,
            "nrmse": 0.18,
        }),
        # Trial 2: worst calibration, best nrmse, smallest model
        _make_trial(2, [0.10, 0.08, 0.1], {
            "param_count": 15_000,
            "calibration_error": 0.10,
            "nrmse": 0.08,
        }),
        # Trial 3: balanced
        _make_trial(3, [0.06, 0.11, 0.3], {
            "param_count": 50_000,
            "calibration_error": 0.06,
            "nrmse": 0.11,
        }),
        # Trial 4: dominated by trial 3 on all objectives
        _make_trial(4, [0.08, 0.15, 0.5], {
            "param_count": 90_000,
            "calibration_error": 0.08,
            "nrmse": 0.15,
        }),
        # Trial 5: trade-off between obj0 and obj2
        _make_trial(5, [0.04, 0.14, 0.4], {
            "param_count": 70_000,
            "calibration_error": 0.04,
            "nrmse": 0.14,
        }),
    ]
    for t in trials:
        study.add_trial(t)
    return study


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
        # Check that grey/gray lines were drawn (iso-mean)
        lines = [
            line for line in ax.get_lines()
            if line.get_color() in ("grey", "gray")
        ]
        assert len(lines) > 0

    def test_iso_lines_disabled(self, multi_objective_study):
        ax = plot_metric_scatter(
            multi_objective_study, "calibration_error", "nrmse",
            show_iso_lines=False,
        )
        lines = [
            line for line in ax.get_lines()
            if line.get_color() in ("grey", "gray")
        ]
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


# ---------------------------------------------------------------------------
# plot_pareto_3d tests
# ---------------------------------------------------------------------------

class TestPlotPareto3D:
    def test_returns_axes3d(self, three_objective_study):
        from mpl_toolkits.mplot3d import Axes3D

        ax = plot_pareto_3d(three_objective_study)
        assert isinstance(ax, Axes3D)

    def test_labels_auto_derived(self, three_objective_study):
        ax = plot_pareto_3d(three_objective_study)
        assert ax.get_xlabel() == "calibration_error"
        assert ax.get_ylabel() == "nrmse"
        assert ax.get_zlabel() == "param_count_norm"

    def test_label_overrides(self, three_objective_study):
        ax = plot_pareto_3d(
            three_objective_study,
            xlabel="X", ylabel="Y", zlabel="Z",
        )
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_zlabel() == "Z"

    def test_cost_display_size(self, three_objective_study):
        ax = plot_pareto_3d(three_objective_study, cost_display="size")
        assert ax.get_title() == "3D Pareto front"

    def test_fewer_than_3_objectives(self, multi_objective_study):
        """Gracefully handles <3 objectives with a text message."""
        from mpl_toolkits.mplot3d import Axes3D

        ax = plot_pareto_3d(multi_objective_study)
        assert isinstance(ax, Axes3D)

    def test_incomplete_trial_filtered(self):
        """Trials with fewer than 3 values are excluded."""
        from mpl_toolkits.mplot3d import Axes3D

        study = optuna.create_study(
            directions=["minimize", "minimize", "minimize"],
        )
        study._metric_names = ["a", "b", "c"]
        study.add_trial(_make_trial(0, [0.1, 0.2, 0.3], {"param_count": 100}))
        # Add a failed trial (no values) — should be filtered out
        study.add_trial(_make_trial(
            1, None, {"param_count": 200},
            state=optuna.trial.TrialState.FAIL,
        ))
        ax = plot_pareto_3d(study)
        assert isinstance(ax, Axes3D)

    def test_empty_study(self, empty_study):
        from mpl_toolkits.mplot3d import Axes3D

        ax = plot_pareto_3d(empty_study)
        assert isinstance(ax, Axes3D)


# ---------------------------------------------------------------------------
# Pareto correctness test
# ---------------------------------------------------------------------------

class TestParetoCorrectness3Obj:
    def test_best_trials_are_non_dominated(self, three_objective_study):
        """Verify study.best_trials forms a valid Pareto set."""
        pareto = three_objective_study.best_trials
        for a in pareto:
            for b in pareto:
                if a.number == b.number:
                    continue
                # b should NOT dominate a (all objectives strictly better)
                dominated = all(
                    bv <= av for bv, av in zip(b.values, a.values)
                ) and any(
                    bv < av for bv, av in zip(b.values, a.values)
                )
                assert not dominated, (
                    f"Trial {b.number} dominates {a.number} "
                    f"in the Pareto set"
                )


# ---------------------------------------------------------------------------
# plot_pareto_projections tests
# ---------------------------------------------------------------------------

class TestPlotParetoProjections:
    def test_returns_3_axes(self, three_objective_study):
        axes = plot_pareto_projections(three_objective_study)
        assert hasattr(axes, "__len__")
        assert len(axes) == 3

    def test_2obj_produces_1_panel(self, multi_objective_study):
        axes = plot_pareto_projections(multi_objective_study)
        assert hasattr(axes, "__len__")
        assert len(axes) == 1

    def test_cost_display_size(self, three_objective_study):
        axes = plot_pareto_projections(
            three_objective_study, cost_display="size",
        )
        assert len(axes) == 3

    def test_labels_correct(self, three_objective_study):
        axes = plot_pareto_projections(three_objective_study)
        assert axes[0].get_xlabel() == "calibration_error"
        assert axes[0].get_ylabel() == "nrmse"
        assert axes[1].get_xlabel() == "calibration_error"
        assert axes[1].get_ylabel() == "param_count_norm"
        assert axes[2].get_xlabel() == "nrmse"
        assert axes[2].get_ylabel() == "param_count_norm"


# ---------------------------------------------------------------------------
# plot_parallel_coordinates tests
# ---------------------------------------------------------------------------

class TestPlotParallelCoordinates:
    def test_returns_axes(self, three_objective_study):
        ax = plot_parallel_coordinates(three_objective_study)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_title(self, three_objective_study):
        ax = plot_parallel_coordinates(three_objective_study)
        assert ax.get_title() == "Parallel coordinates"

    def test_top_k_clamped(self, three_objective_study):
        """top_k larger than trial count doesn't error."""
        ax = plot_parallel_coordinates(three_objective_study, top_k=100)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_metric_order_override(self, three_objective_study):
        ax = plot_parallel_coordinates(
            three_objective_study,
            metric_order=["nrmse", "calibration_error", "param_count_norm"],
        )
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels[0] == "nrmse"

    def test_empty_study(self, empty_study):
        ax = plot_parallel_coordinates(empty_study)
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# plot_study tests
# ---------------------------------------------------------------------------

class TestPlotStudy:
    def test_2obj_returns_figure_with_4_axes(self, multi_objective_study):
        fig = plot_study(multi_objective_study)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 4

    def test_3obj_returns_figure(self, three_objective_study):
        fig = plot_study(three_objective_study)
        assert isinstance(fig, matplotlib.figure.Figure)
        # 3D Pareto + parallel coords + 3 projections = 5 main axes
        # (plus potential colorbar axes)
        assert len(fig.axes) >= 5

    def test_single_obj_raises(self, single_objective_study):
        with pytest.raises(ValueError, match="2 or 3 objectives"):
            plot_study(single_objective_study)

    def test_4obj_raises(self):
        study = optuna.create_study(
            directions=["minimize"] * 4,
        )
        with pytest.raises(ValueError, match="2 or 3 objectives"):
            plot_study(study)

    def test_empty_2obj_study(self, empty_study):
        """Empty study still returns a Figure with placeholder text."""
        fig = plot_study(empty_study)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 4


# ---------------------------------------------------------------------------
# Phase 2: Edge-case tests
# ---------------------------------------------------------------------------

class TestPlotPareto3DEdgeCases:
    def test_minimal_3_trials(self):
        """3D Pareto with exactly 3 trials (minimal Pareto)."""
        from mpl_toolkits.mplot3d import Axes3D

        study = optuna.create_study(
            directions=["minimize", "minimize", "minimize"],
        )
        study._metric_names = ["a", "b", "c"]
        study.add_trial(_make_trial(0, [0.1, 0.2, 0.3], {"param_count": 100}))
        study.add_trial(_make_trial(1, [0.3, 0.1, 0.2], {"param_count": 200}))
        study.add_trial(_make_trial(2, [0.2, 0.3, 0.1], {"param_count": 150}))
        ax = plot_pareto_3d(study)
        assert isinstance(ax, Axes3D)
        assert ax.get_title() == "3D Pareto front"


class TestPlotProjectionsEdgeCases:
    def test_2obj_produces_1_panel(self, multi_objective_study):
        """2-objective study produces exactly 1 projection panel."""
        axes = plot_pareto_projections(multi_objective_study)
        assert len(axes) == 1


class TestPlotParallelCoordinatesEdgeCases:
    def test_top_k_exceeds_trial_count(self, three_objective_study):
        """top_k >> trial count clamps gracefully."""
        ax = plot_parallel_coordinates(three_objective_study, top_k=1000)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert ax.get_title() == "Parallel coordinates"


class TestColorConstantsUsed:
    def test_pareto_front_uses_primary_color(self, multi_objective_study):
        """Spot-check that scatter uses PRIMARY color constant."""
        import matplotlib.colors as mcolors

        from bayesflow_hpo.results import _colors as colors

        ax = plot_pareto_front(multi_objective_study)
        collections = ax.collections
        assert len(collections) >= 1
        facecolors = collections[0].get_facecolors()
        expected = mcolors.to_rgba(colors.PRIMARY, alpha=colors.ALPHA_TRIAL)
        assert facecolors[0] == pytest.approx(expected, abs=0.02)

    def test_optimization_history_uses_best_line_color(
        self, multi_objective_study,
    ):
        """Best-so-far line should use BEST_LINE color."""
        import matplotlib.colors as mcolors

        from bayesflow_hpo.results import _colors as colors

        ax = plot_optimization_history(multi_objective_study)
        best_line_hex = mcolors.to_hex(colors.BEST_LINE)
        step_lines = [
            line for line in ax.get_lines()
            if mcolors.to_hex(line.get_color()) == best_line_hex
        ]
        assert len(step_lines) >= 1
