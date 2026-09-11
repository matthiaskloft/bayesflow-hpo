"""Integration coverage for the metric-name paths #72 broke.

These are the load-bearing tests of the suite.  The smoke test in
``test_optimize_smoke.py`` requests only metrics already in
``DEFAULT_METRICS`` (``validation/registry.py:734``) and therefore guards
neither #72 defect; every test here deliberately requests something that is
*not* default, which is the condition under which both defects appeared.

The helper functions these exercise already have unit coverage in
``tests/test_optimization/test_metric_name_inventory.py``.  What was missing
is evidence that a real ``optimize()`` run reaches them -- both defects were
integration failures between individually well-tested components.

Sources for the contracts asserted here, all recorded in
``docs/references.md``. The ``log_gamma`` direction is BayesFlow's:
``calibration_log_gamma`` reports ``log(gamma / null_quantile)``, the gamma
discrepancy of Modrak et al. (2025), *Bayesian Analysis* 20(2), 461-488,
Equation 7, with ``log_gamma < 0`` rejecting rank uniformity -- so larger is
better and its minimize-form is negation. The ranking and Pareto claims are
Optuna's (Akiba et al., 2019): every objective whose direction is ``minimize``
is minimized, and a trial is non-dominated when no other trial is at least as
good on every objective and strictly better on one -- which is why the
selection tests hold the cost coordinate equal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bayesflow_hpo import objectives as objectives_module
from bayesflow_hpo.objectives import register_metric_direction
from bayesflow_hpo.pipeline import PipelineError

from .conftest import (
    ATTR_ROUNDING_TOL,
    assert_all_minimize,
    assert_trials_succeeded,
)

pytestmark = pytest.mark.endtoend


@pytest.fixture
def restore_metric_directions():
    """Snapshot and restore the global direction registries **in place**.

    ``METRIC_DIRECTIONS`` and ``HIGHER_IS_BETTER`` are module-level mutables
    that other modules import directly, so rebinding the module attribute
    would leave those importers holding the old object.  Mutate and restore
    the same objects instead.
    """
    directions = objectives_module.METRIC_DIRECTIONS
    higher = objectives_module.HIGHER_IS_BETTER
    saved_directions = dict(directions)
    saved_higher = set(higher)
    try:
        yield
    finally:
        directions.clear()
        directions.update(saved_directions)
        higher.clear()
        higher.update(saved_higher)


def test_registered_non_default_metric_is_optimizable(run_study):
    """``log_gamma`` -- registered but not default -- survives the round trip.

    This is the #72 headline defect: ``default_validate_fn`` computed
    ``DEFAULT_METRICS`` and ignored ``objective_metrics``, so optimizing
    ``log_gamma`` raised ``PipelineError`` in pre-flight before training
    started.

    Must fail if: ``objective_metrics`` stops being threaded through to
    ``default_validate_fn``; the higher-is-better conversion is reversed.
    """
    study = run_study(n_trials=1, objective_metrics=["log_gamma"])

    assert_trials_succeeded(study, expected=1)
    assert_all_minimize(study, expected=2)  # log_gamma, cost

    for trial in study.trials:
        raw = trial.user_attrs["log_gamma"]
        objective = trial.values[0]

        # Finiteness is checked *first* and is not decoration. If final
        # validation stopped computing log_gamma, sanitization would insert
        # the raw penalty -inf (objectives.py:291, stored at
        # objective.py:1335), conversion would yield +inf, and the
        # `objective == -raw` identity below would still hold. Without this
        # guard the test passes with the defect present.
        assert math.isfinite(raw), f"trial {trial.number} raw log_gamma is {raw}"
        assert math.isfinite(objective), (
            f"trial {trial.number} objective is {objective}"
        )

        # log_gamma is higher-is-better and unbounded, so its minimize-form is
        # negation. Compared with tolerance because user attrs are rounded to
        # six decimals (objective.py:1347) while the objective is not (:1357).
        assert objective == pytest.approx(-raw, abs=ATTR_ROUNDING_TOL), (
            f"trial {trial.number}: objective {objective} is not the negation "
            f"of stored log_gamma {raw}"
        )


def test_multi_output_constraint_key_is_computed(run_study):
    """A constraint on an output key pulls its *producer* into the pipeline.

    ``left_coverage_90`` is emitted by the ``coverage_left`` metric and is not
    itself a registered metric name. Filtering the pipeline on registered
    names alone dropped the producer, so nothing computed the key -- and
    neither constraint path complains, because the hard path skips a missing
    key and the soft path reads it as zero violation.

    The threshold is permissive so the trial is *not* rejected: this test is
    about the key being computed. Enforcement is
    ``test_unattainable_hard_constraint_rejects_deterministically``.

    Must fail if: ``producer_for_key()`` is dropped from
    ``_pipeline_metrics()``.
    """
    study = run_study(
        n_trials=1,
        metric_constraints_hard=[("left_coverage_90", -1.0, "below")],
    )

    assert_trials_succeeded(study, expected=1)
    for trial in study.trials:
        assert "left_coverage_90" in trial.user_attrs, (
            f"trial {trial.number} never computed left_coverage_90; "
            f"has {sorted(trial.user_attrs)}"
        )
        # Asserting the value would be asserting the model; assert it is a
        # real number that the pipeline produced.
        assert math.isfinite(trial.user_attrs["left_coverage_90"])


def test_unattainable_hard_constraint_rejects_deterministically(run_study):
    """An unreachable threshold rejects regardless of model quality.

    Coverage is a mean of booleans (``validation/registry.py:690``), so it
    cannot exceed 1.0 and a ``"below"`` threshold of 1.1 is violated by every
    trial -- the rejection does not depend on two epochs of training landing
    anywhere in particular.

    ``max_total_trials`` is pinned because hard-rejected trials do not count
    toward the trained total (``optimization/study.py:780``), so the study
    would otherwise keep running until the default ``3 * n_trials`` cap.

    Must fail if: the ``_check_hard_constraints()`` call at
    ``objective.py:1351`` is removed -- which the test above would not notice.
    """
    study = run_study(
        n_trials=1,
        max_total_trials=1,
        metric_constraints_hard=[("left_coverage_90", 1.1, "below")],
    )

    assert len(study.trials) == 1
    trial = study.trials[0]
    assert trial.user_attrs.get("rejected_reason") == "metric_constraint", (
        f"expected a metric_constraint rejection, got "
        f"{trial.user_attrs.get('rejected_reason')!r}"
    )
    # The rejected trial carries the penalty tuple, not a measured objective.
    assert trial.values is not None
    assert all(not np.isfinite(v) or v >= 1.0 for v in trial.values[:-1]), (
        f"expected penalty objectives, got {trial.values}"
    )


def test_soft_constraint_violation_reaches_the_sampler(run_study):
    """Soft constraints are computed and handed to Optuna as violations.

    Separate API and sampler wiring from the hard path, and easy to disable by
    accident: ``optimize()`` silently skips soft constraints when handed a
    sampler *instance* (``api.py:478``), so this deliberately does not pass
    one.

    Uses the same unattainable threshold, which makes the violation
    deterministic: ``max(0, 1.1 - left_coverage_90)`` is strictly positive for
    any attainable coverage.

    Must fail if: soft constraints stop being wired into the sampler preset,
    or their metric stops being computed by the pipeline.
    """
    study = run_study(
        n_trials=1,
        metric_constraints_soft=[("left_coverage_90", 1.1, "below")],
    )

    trial = study.trials[0]
    constraints = trial.system_attrs.get("constraints")
    assert constraints is not None, (
        "no constraints recorded on the trial; soft constraints were not "
        "wired into the sampler"
    )
    # index 0 is the budget-rejection flag, index 1 the soft violation.
    assert len(constraints) == 2, f"expected 2 constraint entries, got {constraints}"
    assert constraints[1] > 0.0, (
        f"expected a positive violation for an unattainable threshold, "
        f"got {constraints[1]}"
    )


def test_direction_registered_under_an_alias_is_honoured(
    run_study, restore_metric_directions
):
    """A direction registered under an alias applies to the canonical metric.

    ``register_metric_direction`` canonicalizes before storing, because a
    direction stored under an alias was never found: registering
    ``"cal_error"`` left ``_direction_for("calibration_error")`` returning the
    built-in lower-is-better conversion, with nothing raised.

    The alias is passed to ``optimize()`` as well, which is the direction the
    bug actually ran: registering under an alias while optimizing the
    canonical name does not exercise the canonicalization boundary.

    The expected objective is asserted against the **independent** formula
    ``1 - raw`` -- the documented default conversion for a higher-is-better
    metric (``objectives.py:436``) -- rather than against whatever the
    implementation happens to compute.

    Must fail if: ``register_metric_direction`` stops canonicalizing, or
    ``optimize()`` stops canonicalizing the objective name it looks up.
    """
    register_metric_direction("cal_error", higher_is_better=True, worst_raw=0.0)

    study = run_study(n_trials=1, objective_metrics=["cal_error"])

    assert_trials_succeeded(study, expected=1)
    trial = study.trials[0]

    raw = trial.user_attrs["calibration_error"]
    objective = trial.values[0]
    assert math.isfinite(raw) and math.isfinite(objective)
    assert objective == pytest.approx(1.0 - raw, abs=ATTR_ROUNDING_TOL), (
        f"objective {objective} is not the higher-is-better conversion "
        f"1 - {raw} = {1.0 - raw}; the aliased direction was not applied"
    )


def test_direction_registry_is_restored(restore_metric_directions):
    """The restore fixture actually restores, so ordering cannot leak.

    Guards the fixture itself: a flipped ``calibration_error`` direction
    leaking into later tests would corrupt every study that follows in the
    same process.
    """
    before = objectives_module.METRIC_DIRECTIONS["calibration_error"]
    assert "calibration_error" not in objectives_module.HIGHER_IS_BETTER

    register_metric_direction("cal_error", higher_is_better=True, worst_raw=0.0)
    assert "calibration_error" in objectives_module.HIGHER_IS_BETTER
    assert objectives_module.METRIC_DIRECTIONS["calibration_error"] is not before


def test_optuna_direction_is_not_leaked_between_tests():
    """Runs after the fixture-scoped flip above and sees the original state."""
    assert "calibration_error" not in objectives_module.HIGHER_IS_BETTER
    assert not objectives_module.METRIC_DIRECTIONS[
        "calibration_error"
    ].higher_is_better


def test_optuna_study_metric_names_record_the_schema(run_study):
    """``metric_names`` records what the stored objective columns mean.

    A study's objective tuple is uninterpretable without it -- the 0.2.0
    schema work exists for this reason.

    Must fail if: ``set_metric_names()`` stops being called, or the recorded
    names stop matching the requested objectives.
    """
    study = run_study(n_trials=1, objective_metrics=["log_gamma"])

    assert study.metric_names is not None, "study recorded no metric names"
    assert study.metric_names[0] == "log_gamma"
    assert len(study.metric_names) == len(study.directions)


def test_unregistered_objective_metric_is_refused(run_study, training_spy):
    """A name no metric produces fails in pre-flight, not silently.

    ``check_pipeline()`` trains one step and then validates
    (``pipeline.py:362``, ``:369``), so the failure costs exactly one training
    call and no optimization trial ever starts. Asserting the call count is
    what distinguishes "pre-flight caught it" from "the run failed somewhere
    later for some other reason" -- a bare ``pytest.raises(Exception)`` would
    accept either.

    Must fail if: pre-flight stops checking that the requested objectives are
    actually produced, or the rejection moves after the trial loop begins.
    """
    with pytest.raises(PipelineError, match="not_a_real_metric"):
        run_study(
            n_trials=1,
            objective_metrics=["not_a_real_metric"],
            train_fn=training_spy,
        )

    assert training_spy.n_calls == 1, (
        f"expected exactly the pre-flight training call, got "
        f"{training_spy.n_calls}; an optimization trial trained despite the "
        f"pre-flight rejection"
    )


def test_optuna_trial_user_attrs_survive_a_multi_objective_study(run_study):
    """Raw metrics remain readable alongside converted objectives.

    Post-hoc analysis reads the raw values; the objective tuple carries the
    minimize-forms. Both must be present and distinguishable.
    """
    study = run_study(n_trials=1, objective_metrics=["log_gamma", "nrmse"])

    assert_trials_succeeded(study, expected=1)
    assert_all_minimize(study, expected=3)

    for trial in study.trials:
        # nrmse is already lower-is-better, so objective == raw.
        assert trial.values[1] == pytest.approx(
            trial.user_attrs["nrmse"], abs=ATTR_ROUNDING_TOL
        )
        # log_gamma is higher-is-better, so objective == -raw. The two
        # conversions differing in the same study is the point.
        assert trial.values[0] == pytest.approx(
            -trial.user_attrs["log_gamma"], abs=ATTR_ROUNDING_TOL
        )


def test_optuna_reports_the_expected_number_of_trained_trials(run_study):
    """``n_trials`` counts *trained* trials, and nothing here is rejected."""
    study = run_study(n_trials=2)
    assert_trials_succeeded(study, expected=2)
    assert all("rejected_reason" not in t.user_attrs for t in study.trials)
