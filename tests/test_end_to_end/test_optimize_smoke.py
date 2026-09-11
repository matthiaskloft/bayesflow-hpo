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
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesflow_hpo.results.extraction import trials_to_dataframe

from .conftest import assert_all_minimize, assert_trials_succeeded

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
