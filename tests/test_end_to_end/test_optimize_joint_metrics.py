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


def test_tarp_error_runs_as_an_objective_through_optimize(run_study) -> None:
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


def test_the_settings_pin_is_stamped_by_a_real_study(run_study) -> None:
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


def test_tarp_error_without_a_reference_stops_the_study(run_study) -> None:
    """A configuration error must stop the study, not penalize every trial.

    It surfaces at the FIRST TRIAL rather than at pre-flight, and that is
    deliberate: pre-flight generates its own five-simulation batch, so
    running a caller's joint metric there fails for reasons about the batch
    size rather than the configuration -- `n_folds=10` cannot fill five
    folds, and a reference array shaped for the production batch does not
    match five rows. A pre-flight that always rejects a valid configuration
    is worse than one that does not examine it.

    What must NOT happen is the failure being absorbed per trial: caught by
    the joint guard, penalized, and repeated for the whole study -- a full
    training run each time, every one scoring the metric's worst case,
    behind a warning log.
    """
    with pytest.raises(
        JointMetricConfigurationError, match="reference points"
    ):
        run_study(objective_metrics=["nrmse", "tarp_error"])


def test_a_ten_fold_lc2st_override_survives_preflight(run_study) -> None:
    """Pre-flight's tiny batch must not reject a valid production config.

    Five simulations per condition cannot fill ten folds, so forwarding the
    caller's real callable into pre-flight rejected a configuration that is
    perfectly valid against the production batch.
    """
    pytest.importorskip("sklearn")

    from bayesflow_hpo.validation.c2st import make_lc2st_joint_metric

    study = run_study(
        objective_metrics=["nrmse"],
        joint_metrics={"lc2st": make_lc2st_joint_metric(n_folds=10)},
        sims_per_condition=40,
    )
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    assert completed
    for trial in completed:
        assert "validation_error" not in trial.user_attrs, (
            trial.user_attrs.get("validation_error")
        )


def test_a_per_condition_reference_sequence_survives_preflight(run_study) -> None:
    """Arrays sized for the production batch must not be run against five rows.

    Truncating them would pair references with different, newly generated
    observations -- a silently wrong check rather than a failed one.
    """
    n_sims = 40
    references = [
        np.zeros((n_sims, 1)) for _ in range(1)
    ]
    study = run_study(
        objective_metrics=["nrmse", "tarp_error"],
        joint_metrics={
            "tarp_error": make_tarp_joint_metric(reference_points=references)
        },
        sims_per_condition=n_sims,
    )
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    assert completed
    for trial in completed:
        assert "validation_error" not in trial.user_attrs, (
            trial.user_attrs.get("validation_error")
        )
        assert not trial.user_attrs.get("failed_joint_metrics"), (
            trial.user_attrs.get("failed_joint_metrics")
        )


def test_the_random_reference_diagnostic_needs_no_configuration(run_study) -> None:
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


def test_an_override_naming_a_non_joint_metric_is_refused(run_study) -> None:
    """Otherwise it runs a metric nobody asked for, under a borrowed name."""
    with pytest.raises(Exception, match="not registered joint metrics"):
        run_study(
            objective_metrics=["nrmse"],
            joint_metrics={"nrmse": lambda inputs: {"nrmse": 0.0}},
        )


def test_a_custom_hook_must_still_produce_its_joint_objective_key(
    run_study,
) -> None:
    """The production-batch restriction excuses the BUILT-IN validator only.

    A custom `validate_fn` computes whatever it likes on whatever batch it
    is given, so nothing about pre-flight's tiny batch excuses it from
    producing the objective keys it was configured for. Narrowing the
    requirement for it would be worse than not checking: with a joint
    metric as the only objective, the required set would be empty and
    pre-flight would verify nothing, while every trial took a penalty.
    """
    from bayesflow_hpo.pipeline import PipelineError

    with pytest.raises(PipelineError, match="tarp_error"):
        run_study(
            objective_metrics=["nrmse", "tarp_error"],
            validate_fn=lambda a, d, n: {"nrmse": 0.2},
        )



def test_a_hook_declaring_settings_is_pinned_across_a_resume(
    run_study, tmp_path
) -> None:
    """#117: a `validate_fn` hook's declared settings reach the study."""
    from bayesflow_hpo.objectives import JOINT_METRIC_SETTINGS_ATTR

    def _hook(reference_id: str):
        def validate_fn(a, d, n):
            return {"nrmse": 0.2}

        validate_fn.joint_metric_settings = {
            "tarp_error_item": {"reference_id": reference_id}
        }
        return validate_fn

    storage = f"sqlite:///{(tmp_path / 'hook.db').as_posix()}"
    study = run_study(
        objective_metrics=["nrmse"],
        validate_fn=_hook("ref-a"),
        storage=storage,
        study_name="hook_pin",
    )
    assert study.user_attrs[JOINT_METRIC_SETTINGS_ATTR] == {
        "tarp_error_item": {"reference_id": "ref-a"}
    }

    # `n_trials` counts the stored trials too, so ask for more than exist.
    with pytest.raises(JointMetricConfigurationError, match="ref-b"):
        run_study(
            objective_metrics=["nrmse"],
            n_trials=4,
            resume=True,
            validate_fn=_hook("ref-b"),
            storage=storage,
            study_name="hook_pin",
        )
