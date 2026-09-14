"""Pinning the configuration a study's joint metrics ran at.

A joint metric's score moves with its settings -- TARP's with ``resolution``,
``metric``, ``standardize`` and the reference draw -- so two trials scored
under different settings are not comparable. This is the same defect class
as the objective schema guard, and deliberately NOT the same mechanism:
``bayesflow_hpo_objective_schema`` is a positional list of column names, so
appending settings to it would change its length and break resumption for
every study already stamped.

Design: ``docs/plans/plan-joint-metric-path.md`` D7.
"""

from __future__ import annotations

import optuna
import pytest

from bayesflow_hpo.objectives import (
    JOINT_METRIC_SETTINGS_ATTR,
    check_or_stamp_joint_metric_settings,
)
from bayesflow_hpo.validation.registry import joint_metric_settings

TARP = {"tarp_error": {"resolution": 20, "metric": "euclidean", "seed": 42}}


def _study():
    return optuna.create_study(directions=["minimize"])


# ---------------------------------------------------------------------------
# Stamp, then compare
# ---------------------------------------------------------------------------


def test_a_fresh_study_is_stamped():
    study = _study()
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=0)
    assert study.user_attrs[JOINT_METRIC_SETTINGS_ATTR] == TARP


def test_identical_settings_pass():
    study = _study()
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=0)
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=3)


def test_changed_settings_are_refused_and_named():
    """The message has to say WHAT changed, or it cannot be acted on."""
    study = _study()
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=0)

    changed = {"tarp_error": {**TARP["tarp_error"], "resolution": 100}}
    with pytest.raises(ValueError, match="different joint metric settings"):
        check_or_stamp_joint_metric_settings(
            study, changed, n_completed_trials=3
        )

    with pytest.raises(ValueError, match="resolution"):
        check_or_stamp_joint_metric_settings(
            study, changed, n_completed_trials=3
        )


def test_a_changed_reference_mode_is_refused():
    """The pin's most load-bearing field: the two modes are different metrics."""
    study = _study()
    check_or_stamp_joint_metric_settings(
        study, {"tarp_error": {"reference_mode": "provided"}},
        n_completed_trials=0,
    )
    with pytest.raises(ValueError, match="reference_mode"):
        check_or_stamp_joint_metric_settings(
            study, {"tarp_error": {"reference_mode": "random"}},
            n_completed_trials=1,
        )


# ---------------------------------------------------------------------------
# The three configurations D7 leaves to the implementation
# ---------------------------------------------------------------------------


def test_a_study_populated_before_this_existed_is_refused_not_stamped():
    """Stamping would assert something about trials that is unknown.

    Those trials ran at settings nobody recorded. Marking the study as
    having run at THIS run's settings would make the pin a false claim, and
    a later resume would then compare against it as if verified.
    """
    study = _study()
    with pytest.raises(ValueError, match="records no joint metric settings"):
        check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=5)
    assert JOINT_METRIC_SETTINGS_ATTR not in study.user_attrs


def test_the_refusal_names_the_escape_hatch():
    """A study whose settings the user DOES know must be recoverable."""
    study = _study()
    with pytest.raises(ValueError, match=JOINT_METRIC_SETTINGS_ATTR):
        check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=5)

    study.set_user_attr(JOINT_METRIC_SETTINGS_ATTR, TARP)
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=5)


def test_a_study_using_no_joint_metrics_never_acquires_the_attribute():
    """The guard must not touch the overwhelmingly common case."""
    study = _study()
    check_or_stamp_joint_metric_settings(study, {}, n_completed_trials=9)
    assert JOINT_METRIC_SETTINGS_ATTR not in study.user_attrs


def test_a_metric_added_mid_study_is_recorded_alongside():
    """Its trials are comparable among themselves; earlier ones lack the key."""
    study = _study()
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=0)
    check_or_stamp_joint_metric_settings(
        study,
        {**TARP, "lc2st": {"n_folds": 5}},
        n_completed_trials=4,
    )
    stored = study.user_attrs[JOINT_METRIC_SETTINGS_ATTR]
    assert set(stored) == {"tarp_error", "lc2st"}


def test_an_unrecognized_stored_value_is_treated_as_absent():
    """`user_attrs` is caller-writable and round-trips through JSON.

    Refusing on a value this code cannot interpret would block a study over
    something it cannot even describe -- the same choice the objective
    schema guard makes.
    """
    study = _study()
    study.set_user_attr(JOINT_METRIC_SETTINGS_ATTR, "not a mapping")
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=0)
    assert study.user_attrs[JOINT_METRIC_SETTINGS_ATTR] == TARP


def test_the_attribute_survives_a_json_round_trip():
    """It is stored by Optuna, so it has to be JSON-serializable."""
    import json

    from bayesflow_hpo.validation.c2st import make_lc2st_joint_metric
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    declared = joint_metric_settings(
        {
            "tarp_error_random": make_tarp_joint_metric(),
            "lc2st": make_lc2st_joint_metric(max_conditions=4),
        }
    )
    assert json.loads(json.dumps(declared)) == declared


# ---------------------------------------------------------------------------
# What the metrics actually declare
# ---------------------------------------------------------------------------


def test_the_settings_come_from_the_callable_that_ran():
    """Not from a parameter the caller repeats to optimize().

    A caller-supplied record can disagree with what was computed; a
    callable's own declaration cannot.
    """
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    fn = make_tarp_joint_metric(resolution=77, metric="manhattan", seed=3)
    declared = joint_metric_settings({"tarp_error_random": fn})
    assert declared["tarp_error_random"] == {
        "resolution": 77,
        "metric": "manhattan",
        "standardize": True,
        "seed": 3,
        "reference_mode": "random",
    }


def test_a_metric_declaring_nothing_is_omitted_not_recorded_as_empty():
    """Absent means "makes no claim", which is not "configured with nothing"."""
    assert joint_metric_settings({"plain": lambda inputs: {}}) == {}


def test_the_reference_mode_is_declared_from_how_it_was_built():
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    provided = make_tarp_joint_metric(reference_points=lambda i: None)
    random = make_tarp_joint_metric()
    assert provided.joint_metric_settings["reference_mode"] == "provided"
    assert random.joint_metric_settings["reference_mode"] == "random"
