"""A real ``optimize()`` run, end to end.

This is the smoke test: tiny simulator, real approximator, real BayesFlow fit
loop, real validation pipeline, real Optuna study.  It asserts that the
pipeline *runs*, not that the resulting model is any good -- two epochs on 128
simulations is noise, and a threshold on ``calibration_error`` would be a
flake generator.

**What this file does not guard.**  It requests only metrics that are already
in ``DEFAULT_METRICS`` (``validation/registry.py:734``), so it does **not**
catch either of the #72 integration defects: remove objective-metric threading
from pre-flight, or constraint producers from final validation, and everything
here still passes.  Those live in ``test_optimize_metric_paths.py``, which
requests non-default metrics deliberately.

Must fail if: training is replaced by a no-op; a trial silently takes the
training-error or validation-error fallback; an objective direction is
flipped; the objective columns stop reaching the results frame.

Sources for the contracts asserted here, all recorded in
``docs/references.md``.

The ``log_gamma`` direction: the gamma discrepancy -- the probability, under
uniform ranks, of the most extreme point of the observed rank ECDF -- is
Sailynoja et al. (2022). Modrak et al. (2025) adopt it in Section 4.1 and
define the quantity BayesFlow reports, ``log(gamma / gamma_bar)`` with
``gamma_bar`` the 5th percentile of the null distribution, stating that
``log(gamma / gamma_bar) < 0`` implies rejection of uniform ranks at the 5%
level. Larger is therefore better, and its minimize-form is negation.

The ranking claims are Optuna's documented semantics: every objective whose
direction is ``minimize`` is minimized. The non-dominance rule a trial must
satisfy to sit on the Pareto front -- no other trial at least as good on every
objective and strictly better on one -- is Deb et al. (2002); it is why the
selection tests hold the cost coordinate equal.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesflow_hpo.results.extraction import trials_to_dataframe

from .conftest import (
    ATTR_ROUNDING_TOL,
    assert_all_minimize,
    assert_trials_succeeded,
)

pytestmark = pytest.mark.endtoend


def test_default_objectives_complete_successfully(run_study, training_spy):
    """Two trials, default objectives, nothing silently falling back."""
    study = run_study(
        objective_metrics=["calibration_error", "nrmse"],
        train_fn=training_spy,
    )

    assert_trials_succeeded(study, expected=2)
    # calibration_error, nrmse, cost
    assert_all_minimize(study, expected=3)

    for trial in study.trials:
        for key in ("calibration_error", "nrmse", "param_count", "inference_time_s"):
            assert key in trial.user_attrs, (
                f"trial {trial.number} is missing user attr {key!r}"
            )

    # Training actually ran. One call for check_pipeline()'s pre-flight
    # (pipeline.py:323) plus one per trial; every call took optimizer steps.
    assert training_spy.n_calls == 3, (
        f"expected 1 pre-flight + 2 trial training calls, "
        f"got {training_spy.n_calls}"
    )
    assert all(steps > 0 for steps in training_spy.iterations), (
        f"a training call took no optimizer steps: {training_spy.iterations}"
    )


def test_results_frame_carries_the_objective_columns(run_study):
    """The study is readable through the public results API."""
    study = run_study(objective_metrics=["calibration_error", "nrmse"])

    assert_trials_succeeded(study, expected=2)

    frame = trials_to_dataframe(study)
    assert len(frame) == 2
    for column in ("calibration_error", "nrmse"):
        assert column in frame.columns, (
            f"{column!r} missing from {sorted(frame.columns)}"
        )
        assert np.isfinite(frame[column]).all()


def test_mean_mode_collapses_to_a_single_quality_objective(run_study):
    """``objective_mode="mean"`` yields two directions, not three."""
    study = run_study(
        n_trials=1,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="mean",
    )

    assert_trials_succeeded(study, expected=1)
    assert_all_minimize(study, expected=2)


def test_cost_metric_none_optimizes_quality_alone(run_study):
    """``cost_metric=None`` drops the direction but keeps the measurement.

    The measurement is the point of the feature: post-hoc cost ranking needs
    ``param_count`` and ``inference_time_s`` on every trial, and only the
    Optuna objective column goes away.
    """
    study = run_study(
        objective_metrics=["calibration_error", "nrmse"],
        cost_metric=None,
    )

    assert_trials_succeeded(study, expected=2)
    # calibration_error, nrmse -- and no cost.
    assert_all_minimize(study, expected=2)
    assert list(study.metric_names or []) == ["calibration_error", "nrmse"]

    for trial in study.trials:
        for key in ("param_count", "inference_time_s"):
            assert key in trial.user_attrs, (
                f"trial {trial.number} is missing user attr {key!r}; cost must "
                f"still be measured when it is not an objective"
            )
        # The columns hold the quality metrics, not one of them plus a cost.
        for i, key in enumerate(("calibration_error", "nrmse")):
            assert trial.values[i] == pytest.approx(
                trial.user_attrs[key], abs=ATTR_ROUNDING_TOL
            )


def test_cost_metric_none_mean_mode_is_single_objective(run_study):
    """Mean mode without a cost column leaves exactly one direction."""
    study = run_study(
        n_trials=1,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="mean",
        cost_metric=None,
    )

    assert_trials_succeeded(study, expected=1)
    assert_all_minimize(study, expected=1)
