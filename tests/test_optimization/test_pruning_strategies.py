"""Tests for multi-objective pruning strategies.

Tests cover the three strategy functions in
``bayesflow_hpo.optimization.pruning_strategies`` and the
``_non_dominated_sort`` helper.
"""

from __future__ import annotations

import numpy as np
import optuna
from optuna.trial import TrialState

from bayesflow_hpo.optimization.pruning_strategies import (
    _non_dominated_sort,
    should_prune_dominance,
    should_prune_mo_sha,
    should_prune_primary,
)


def _make_study(n_objectives: int = 2) -> optuna.Study:
    """Create an in-memory multi-objective study."""
    return optuna.create_study(directions=["minimize"] * n_objectives)


def _add_completed_trial(
    study: optuna.Study,
    values: list[float],
    user_attrs: dict | None = None,
) -> None:
    """Add a synthetic completed trial to *study*."""
    trial = optuna.trial.create_trial(
        params={},
        distributions={},
        values=values,
        user_attrs=user_attrs or {},
        state=TrialState.COMPLETE,
    )
    study.add_trial(trial)


def _make_metric_attrs(
    metrics: dict[str, float],
    step: int,
) -> dict[str, float]:
    """Build ``val_{metric}_step_{step}`` user attrs from a metric dict."""
    return {f"val_{m}_step_{step}": v for m, v in metrics.items()}



class TestDominance:
    """Tests for should_prune_dominance()."""

    def test_no_pruning_below_startup(self):
        """Should not prune when fewer than n_startup_trials refs exist."""
        study = _make_study()
        for i in range(3):
            _add_completed_trial(
                study,
                [0.1 + i * 0.1, 0.5],
                _make_metric_attrs({"cal": 0.1 + i * 0.1, "nrmse": 0.5}, step=1),
            )
        trial = study.ask()
        assert not should_prune_dominance(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=1, n_startup_trials=5
        )

    def test_no_pruning_at_median(self):
        """Score equal to median on all metrics should NOT trigger pruning."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study,
                [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        trial = study.ask()
        # Median is 0.3 on both metrics — equal should not prune.
        assert not should_prune_dominance(
            trial, {"cal": 0.3, "nrmse": 0.3}, step=1, n_startup_trials=5
        )

    def test_prune_above_median_all_metrics(self):
        """Worse than median on ALL metrics → prune."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study,
                [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        trial = study.ask()
        assert should_prune_dominance(
            trial, {"cal": 0.4, "nrmse": 0.4}, step=1, n_startup_trials=5
        )

    def test_and_rule_no_prune_when_good_on_one(self):
        """Bad on one metric but good on another → NO prune (AND rule)."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study,
                [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        trial = study.ask()
        # Bad on cal (0.9) but good on nrmse (0.1) → no prune.
        assert not should_prune_dominance(
            trial, {"cal": 0.9, "nrmse": 0.1}, step=1, n_startup_trials=5
        )

    def test_nan_score_triggers_pruning(self):
        """NaN in current scores → immediate prune."""
        study = _make_study()
        trial = study.ask()
        assert should_prune_dominance(
            trial,
            {"cal": float("nan"), "nrmse": 0.1},
            step=1,
            n_startup_trials=5,
        )

    def test_inf_score_triggers_pruning(self):
        """Inf in current scores → immediate prune."""
        study = _make_study()
        trial = study.ask()
        assert should_prune_dominance(
            trial,
            {"cal": float("inf"), "nrmse": 0.1},
            step=1,
            n_startup_trials=5,
        )

    def test_nan_in_reference_scores_filtered(self):
        """NaN in completed trials' attrs → those trials skipped."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study,
                [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        # Add two trials with NaN — should be filtered.
        _add_completed_trial(
            study,
            [0.9, 0.9],
            _make_metric_attrs({"cal": float("nan"), "nrmse": 0.5}, step=1),
        )
        _add_completed_trial(
            study,
            [0.9, 0.9],
            _make_metric_attrs({"cal": 0.5, "nrmse": float("nan")}, step=1),
        )
        trial = study.ask()
        # Median of [0.1..0.5] is 0.3 — should prune at 0.4.
        assert should_prune_dominance(
            trial, {"cal": 0.4, "nrmse": 0.4}, step=1, n_startup_trials=5
        )

    def test_budget_rejected_excluded(self):
        """Trials with rejected_reason should not count as references."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            attrs = _make_metric_attrs({"cal": s, "nrmse": s}, step=1)
            if s >= 0.3:
                attrs["rejected_reason"] = "param_budget"
            _add_completed_trial(study, [s, s], attrs)
        trial = study.ask()
        # Only 2 non-rejected → below startup of 5.
        assert not should_prune_dominance(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=1, n_startup_trials=5
        )

    def test_step_independence(self):
        """Scores at step 1 should not affect step 2 decisions."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study,
                [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        trial = study.ask()
        # No step-2 data → below startup.
        assert not should_prune_dominance(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=2, n_startup_trials=5
        )

    def test_n_startup_zero_never_prunes(self):
        """n_startup_trials=0 should be treated as disabled."""
        study = _make_study()
        for s in [0.1, 0.2]:
            _add_completed_trial(
                study,
                [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        trial = study.ask()
        assert not should_prune_dominance(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=1, n_startup_trials=0
        )

    def test_scale_normalization(self):
        """Metrics on very different scales should be handled correctly."""
        study = _make_study()
        # cal ~ [0.01, 0.05], nrmse ~ [100, 500]
        for cal, nrmse in [
            (0.01, 100),
            (0.02, 200),
            (0.03, 300),
            (0.04, 400),
            (0.05, 500),
        ]:
            _add_completed_trial(
                study,
                [cal, nrmse],
                _make_metric_attrs({"cal": cal, "nrmse": nrmse}, step=1),
            )
        trial = study.ask()
        # Good on cal (0.01) but bad on nrmse (600) → no prune (AND rule).
        assert not should_prune_dominance(
            trial, {"cal": 0.01, "nrmse": 600}, step=1, n_startup_trials=5
        )
        # Bad on both (normalized) → prune.
        assert should_prune_dominance(
            trial, {"cal": 0.04, "nrmse": 400}, step=1, n_startup_trials=5
        )

    def test_degenerate_range_single_value(self):
        """All reference scores identical → skip normalization, no crash."""
        study = _make_study()
        for _ in range(5):
            _add_completed_trial(
                study,
                [0.3, 0.3],
                _make_metric_attrs({"cal": 0.3, "nrmse": 0.3}, step=1),
            )
        trial = study.ask()
        # Exactly at median → no prune.
        assert not should_prune_dominance(
            trial, {"cal": 0.3, "nrmse": 0.3}, step=1, n_startup_trials=5
        )
        # Above → prune.
        assert should_prune_dominance(
            trial, {"cal": 0.4, "nrmse": 0.4}, step=1, n_startup_trials=5
        )

    def test_cross_schema_migration(self):
        """Trials with old val_score_step_* format should be silently skipped."""
        study = _make_study()
        # 3 trials with new schema.
        for s in [0.1, 0.2, 0.3]:
            _add_completed_trial(
                study,
                [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        # 5 trials with old schema — should be skipped.
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study,
                [s, s],
                {"val_score_step_1": s},
            )
        trial = study.ask()
        # Only 3 new-schema refs → below startup of 5 → no prune.
        assert not should_prune_dominance(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=1, n_startup_trials=5
        )



class TestMoSha:
    """Tests for should_prune_mo_sha()."""

    def test_no_pruning_below_startup(self):
        """Should not prune when fewer than n_startup_trials refs exist."""
        study = _make_study()
        for i in range(3):
            _add_completed_trial(
                study,
                [0.1, 0.5],
                _make_metric_attrs({"cal": 0.1 + i * 0.1, "nrmse": 0.5}, step=1),
            )
        trial = study.ask()
        assert not should_prune_mo_sha(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=1, n_startup_trials=5
        )

    def test_nan_triggers_pruning(self):
        """NaN in current scores → immediate prune."""
        study = _make_study()
        trial = study.ask()
        assert should_prune_mo_sha(
            trial,
            {"cal": float("nan"), "nrmse": 0.1},
            step=1,
            n_startup_trials=5,
        )

    def test_inf_triggers_pruning(self):
        """Inf in current scores → immediate prune."""
        study = _make_study()
        trial = study.ask()
        assert should_prune_mo_sha(
            trial,
            {"cal": float("inf"), "nrmse": 0.1},
            step=1,
            n_startup_trials=5,
        )

    def test_pareto_optimal_not_pruned(self):
        """Trial that dominates all references should not be pruned."""
        study = _make_study()
        # 6 reference trials — all mediocre.
        for cal, nrmse in [
            (0.5, 0.5), (0.6, 0.4), (0.4, 0.6),
            (0.7, 0.3), (0.3, 0.7), (0.55, 0.55),
        ]:
            _add_completed_trial(
                study,
                [cal, nrmse],
                _make_metric_attrs({"cal": cal, "nrmse": nrmse}, step=1),
            )
        trial = study.ask()
        # Current trial dominates all refs → must be in front 0 → survives.
        assert not should_prune_mo_sha(
            trial, {"cal": 0.1, "nrmse": 0.1}, step=1, n_startup_trials=5
        )

    def test_dominated_point_pruned(self):
        """Trial dominated by many should be pruned."""
        study = _make_study()
        # 9 trials, all very good.
        for i in range(9):
            cal = 0.01 + i * 0.01
            nrmse = 0.01 + i * 0.01
            _add_completed_trial(
                study,
                [cal, nrmse],
                _make_metric_attrs({"cal": cal, "nrmse": nrmse}, step=1),
            )
        trial = study.ask()
        # A very bad trial should be pruned (bottom fraction with eta=3).
        assert should_prune_mo_sha(
            trial, {"cal": 0.99, "nrmse": 0.99}, step=1, n_startup_trials=5
        )

    def test_bottom_fraction_with_eta_3(self):
        """With eta=3, top 1/3 survive. Bottom 2/3 should be pruned."""
        study = _make_study()
        # 6 trials arranged by increasing badness (both metrics correlated).
        for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            _add_completed_trial(
                study,
                [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        trial = study.ask()
        # With 7 total (6 ref + 1 current), select top 7//3 = 2.
        # Best two are (0.1, 0.1) and (0.2, 0.2).
        # A bad trial at (0.55, 0.55) is dominated → should be pruned.
        assert should_prune_mo_sha(
            trial, {"cal": 0.55, "nrmse": 0.55}, step=1, n_startup_trials=5
        )

    def test_small_reference_set(self):
        """With fewer than eta trials, at least 1 should survive."""
        study = _make_study()
        # Only 2 reference trials.
        _add_completed_trial(
            study, [0.1, 0.1],
            _make_metric_attrs({"cal": 0.1, "nrmse": 0.1}, step=1),
        )
        _add_completed_trial(
            study, [0.5, 0.5],
            _make_metric_attrs({"cal": 0.5, "nrmse": 0.5}, step=1),
        )
        trial = study.ask()
        # 3 total, select max(1, 3//3) = 1. Best is (0.1, 0.1).
        # Current at (0.05, 0.05) is Pareto-optimal → should survive.
        assert not should_prune_mo_sha(
            trial, {"cal": 0.05, "nrmse": 0.05}, step=1, n_startup_trials=2
        )

    def test_n_startup_zero_never_prunes(self):
        """n_startup_trials=0 → disabled."""
        study = _make_study()
        trial = study.ask()
        assert not should_prune_mo_sha(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=1, n_startup_trials=0
        )

    def test_budget_rejected_excluded(self):
        """Rejected trials should not be in the reference set."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            attrs = _make_metric_attrs({"cal": s, "nrmse": s}, step=1)
            if s >= 0.3:
                attrs["rejected_reason"] = "param_budget"
            _add_completed_trial(study, [s, s], attrs)
        trial = study.ask()
        # Only 2 non-rejected → below startup of 5.
        assert not should_prune_mo_sha(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=1, n_startup_trials=5
        )

    def test_cross_schema_migration(self):
        """Old-schema trials silently skipped (conservative)."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3]:
            _add_completed_trial(
                study, [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(study, [s, s], {"val_score_step_1": s})
        trial = study.ask()
        # Only 3 new-schema → below startup of 5.
        assert not should_prune_mo_sha(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=1, n_startup_trials=5
        )

    def test_step_independence(self):
        """Step 1 data does not affect step 2 decisions."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study, [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        trial = study.ask()
        assert not should_prune_mo_sha(
            trial, {"cal": 0.9, "nrmse": 0.9}, step=2, n_startup_trials=5
        )

    def test_custom_reduction_factor(self):
        """Custom reduction_factor changes how many survive."""
        study = _make_study()
        # 5 refs arranged by increasing badness.
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study, [s, s],
                _make_metric_attrs({"cal": s, "nrmse": s}, step=1),
            )
        trial = study.ask()
        # eta=2: select 6//2 = 3 survivors. Trial at (0.25, 0.25) is in
        # front 2 (dominated by 0.1 and 0.2) → 3 selected from fronts
        # 0,1,2 → should survive.
        assert not should_prune_mo_sha(
            trial,
            {"cal": 0.25, "nrmse": 0.25},
            step=1,
            n_startup_trials=5,
            reduction_factor=2,
        )
        # eta=6: select 6//6 = 1. Only the best (0.1, 0.1) survives.
        # Trial at (0.25, 0.25) is pruned.
        assert should_prune_mo_sha(
            trial,
            {"cal": 0.25, "nrmse": 0.25},
            step=1,
            n_startup_trials=5,
            reduction_factor=6,
        )


class TestPrimary:
    """Tests for should_prune_primary()."""

    def test_no_pruning_below_startup(self):
        """Should not prune when fewer than n_startup_trials refs exist."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3]:
            _add_completed_trial(
                study,
                [s, 0.5],
                _make_metric_attrs({"cal": s}, step=1),
            )
        trial = study.ask()
        assert not should_prune_primary(
            trial, score=0.9, metric="cal", step=1, n_startup_trials=5
        )

    def test_no_pruning_at_median(self):
        """Equal to median → no prune."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study, [s, 0.5],
                _make_metric_attrs({"cal": s}, step=1),
            )
        trial = study.ask()
        assert not should_prune_primary(
            trial, score=0.3, metric="cal", step=1, n_startup_trials=5
        )

    def test_prune_above_median(self):
        """Score above median → prune."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study, [s, 0.5],
                _make_metric_attrs({"cal": s}, step=1),
            )
        trial = study.ask()
        assert should_prune_primary(
            trial, score=0.4, metric="cal", step=1, n_startup_trials=5
        )

    def test_nan_triggers_pruning(self):
        """NaN score → immediate prune."""
        study = _make_study()
        trial = study.ask()
        assert should_prune_primary(
            trial,
            score=float("nan"),
            metric="cal",
            step=1,
            n_startup_trials=5,
        )

    def test_inf_triggers_pruning(self):
        """Inf score → immediate prune."""
        study = _make_study()
        trial = study.ask()
        assert should_prune_primary(
            trial,
            score=float("inf"),
            metric="cal",
            step=1,
            n_startup_trials=5,
        )

    def test_n_startup_zero_never_prunes(self):
        """n_startup_trials=0 → disabled."""
        study = _make_study()
        trial = study.ask()
        assert not should_prune_primary(
            trial, score=0.9, metric="cal", step=1, n_startup_trials=0
        )

    def test_budget_rejected_excluded(self):
        """Rejected trials not counted."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            attrs = _make_metric_attrs({"cal": s}, step=1)
            if s >= 0.3:
                attrs["rejected_reason"] = "param_budget"
            _add_completed_trial(study, [s, 0.5], attrs)
        trial = study.ask()
        assert not should_prune_primary(
            trial, score=0.9, metric="cal", step=1, n_startup_trials=5
        )

    def test_step_independence(self):
        """Step 1 data does not affect step 2."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study, [s, 0.5],
                _make_metric_attrs({"cal": s}, step=1),
            )
        trial = study.ask()
        assert not should_prune_primary(
            trial, score=0.9, metric="cal", step=2, n_startup_trials=5
        )

    def test_nan_in_reference_filtered(self):
        """NaN reference scores are filtered out."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(
                study, [s, 0.5],
                _make_metric_attrs({"cal": s}, step=1),
            )
        _add_completed_trial(
            study, [0.9, 0.5],
            _make_metric_attrs({"cal": float("nan")}, step=1),
        )
        trial = study.ask()
        # Median of [0.1..0.5] is 0.3.
        assert not should_prune_primary(
            trial, score=0.2, metric="cal", step=1, n_startup_trials=5
        )
        assert should_prune_primary(
            trial, score=0.4, metric="cal", step=1, n_startup_trials=5
        )

    def test_cross_schema_migration(self):
        """Old val_score_step_* trials silently skipped."""
        study = _make_study()
        for s in [0.1, 0.2, 0.3]:
            _add_completed_trial(
                study, [s, 0.5],
                _make_metric_attrs({"cal": s}, step=1),
            )
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            _add_completed_trial(study, [s, 0.5], {"val_score_step_1": s})
        trial = study.ask()
        # Only 3 new-schema → below startup.
        assert not should_prune_primary(
            trial, score=0.9, metric="cal", step=1, n_startup_trials=5
        )



class TestNonDominatedSort:
    """Tests for the _non_dominated_sort() helper."""

    def test_single_front(self):
        """All non-dominated → single front."""
        # Pareto front: (0.1, 0.9), (0.5, 0.5), (0.9, 0.1)
        obj = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        fronts = _non_dominated_sort(obj)
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2}

    def test_two_fronts(self):
        """Two clear fronts."""
        obj = np.array([
            [0.1, 0.1],  # Front 1
            [0.2, 0.2],  # Front 2
            [0.5, 0.5],  # Front 3
        ])
        fronts = _non_dominated_sort(obj)
        assert len(fronts) == 3
        assert fronts[0] == [0]
        assert fronts[1] == [1]
        assert fronts[2] == [2]

    def test_identical_points_same_front(self):
        """Identical points are non-dominated w.r.t. each other."""
        obj = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        fronts = _non_dominated_sort(obj)
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2}

    def test_single_objective(self):
        """Works with M=1 (reduces to sorted order)."""
        obj = np.array([[0.3], [0.1], [0.2]])
        fronts = _non_dominated_sort(obj)
        assert len(fronts) == 3
        assert fronts[0] == [1]  # 0.1
        assert fronts[1] == [2]  # 0.2
        assert fronts[2] == [0]  # 0.3

    def test_three_objectives(self):
        """Works with M=3."""
        obj = np.array([
            [0.1, 0.1, 0.9],  # Non-dominated
            [0.9, 0.1, 0.1],  # Non-dominated
            [0.5, 0.5, 0.5],  # Dominated by neither
            [0.8, 0.8, 0.8],  # Dominated by all above
        ])
        fronts = _non_dominated_sort(obj)
        assert set(fronts[0]) == {0, 1, 2}
        assert fronts[1] == [3]

    def test_empty_after_first_front(self):
        """Single point → one front."""
        obj = np.array([[0.5, 0.5]])
        fronts = _non_dominated_sort(obj)
        assert len(fronts) == 1
        assert fronts[0] == [0]
