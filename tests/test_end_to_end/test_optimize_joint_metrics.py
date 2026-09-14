"""`optimize()` with a joint metric, through the real public entry point.

Three review rounds each found a defect on the path through the pieces while
every piece tested correct in isolation, and the last of them was a call site
in `check_pipeline` that no unit test could reach. A signature assertion --
"`joint_metrics` is a parameter of `optimize`" -- is not evidence that the
parameter works; this is.

Marked ``endtoend`` with the rest of this directory: it builds a real
approximator and runs real trials.

Design: ``docs/plans/plan-joint-metric-path.md`` D6.
"""

from __future__ import annotations

import numpy as np
import optuna
import pytest

from bayesflow_hpo.validation.registry import JointMetricConfigurationError
from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

pytestmark = pytest.mark.endtoend


def _reference_from_data(inputs):
    """A reference point derived from the conditioning data.

    Deliberately a function of ``sim_batch`` and not of ``draws``: a provider
    that samples the posterior it is auditing produces a reference that looks
    data-dependent and is not, which is the blind spot the two-key split
    exists to prevent.
    """
    x = np.asarray(inputs.sim_batch["x"], dtype=float)
    per_sim = x.reshape(x.shape[0], -1).mean(axis=1, keepdims=True)
    return np.repeat(per_sim, inputs.draws.shape[2], axis=1)


def test_tarp_error_runs_as_an_objective_through_optimize(run_study):
    """The headline metric, used the way the docs say to use it.

    `tarp_error` is registered, resolvable and `kind="objective"`, but cannot
    run at a registry default -- reference points have to come from the
    caller. If any link in `optimize()` -> `check_pipeline` ->
    `ObjectiveConfig` -> `run_validation_pipeline` drops `joint_metrics`, the
    placeholder resolves instead and the study dies before its first trial.
    """
    study = run_study(
        objective_metrics=["nrmse", "tarp_error"],
        joint_metrics={
            "tarp_error": make_tarp_joint_metric(
                reference_points=_reference_from_data
            )
        },
    )

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    assert completed, "no trial completed"
    for trial in completed:
        assert "validation_error" not in trial.user_attrs, (
            f"trial {trial.number} fell through to the failure path: "
            f"{trial.user_attrs.get('validation_error')}"
        )
        assert not trial.user_attrs.get("failed_joint_metrics"), (
            f"trial {trial.number} recorded a failed joint metric: "
            f"{trial.user_attrs.get('failed_joint_metrics')}"
        )
        assert all(np.isfinite(v) for v in trial.values), trial.values


def test_the_settings_pin_is_stamped_by_a_real_study(run_study):
    """D7's attribute has to survive Optuna's own JSON round trip."""
    from bayesflow_hpo.objectives import JOINT_METRIC_SETTINGS_ATTR

    study = run_study(
        objective_metrics=["nrmse", "tarp_error"],
        joint_metrics={
            "tarp_error": make_tarp_joint_metric(
                reference_points=_reference_from_data, resolution=13
            )
        },
    )

    stored = study.user_attrs[JOINT_METRIC_SETTINGS_ATTR]
    assert stored["tarp_error"]["resolution"] == 13
    assert stored["tarp_error"]["reference_mode"] == "provided"


def test_tarp_error_without_a_reference_stops_before_training(run_study):
    """A configuration error must surface at pre-flight, not per trial.

    Left to run time it would be caught by the joint guard, penalized, and
    repeated for every trial in the study -- a full training run each time,
    all scoring the metric's worst case, behind a warning log.
    """
    from bayesflow_hpo.pipeline import PipelineError

    with pytest.raises(
        (JointMetricConfigurationError, PipelineError),
        match="reference points",
    ):
        run_study(objective_metrics=["nrmse", "tarp_error"])


def test_the_random_reference_diagnostic_needs_no_configuration(run_study):
    """`tarp_error_random` runs at its registered default, as a constraint."""
    study = run_study(
        objective_metrics=["nrmse"],
        metric_constraints_soft=[("tarp_error_random", 0.9, "below")],
    )
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    assert completed
    for trial in completed:
        assert "validation_error" not in trial.user_attrs, (
            trial.user_attrs.get("validation_error")
        )


def test_an_override_naming_a_non_joint_metric_is_refused(run_study):
    """Otherwise it runs a metric nobody asked for, under a borrowed name."""
    with pytest.raises(Exception, match="not registered joint metrics"):
        run_study(
            objective_metrics=["nrmse"],
            joint_metrics={"nrmse": lambda inputs: {"nrmse": 0.0}},
        )
