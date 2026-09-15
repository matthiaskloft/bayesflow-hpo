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


def test_a_fresh_study_is_stamped() -> None:
    study = _study()
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=0)
    assert study.user_attrs[JOINT_METRIC_SETTINGS_ATTR] == TARP


def test_identical_settings_pass() -> None:
    study = _study()
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=0)
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=3)


def test_changed_settings_are_refused_and_named() -> None:
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


def test_a_changed_reference_mode_is_refused() -> None:
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


def test_a_study_populated_before_this_existed_is_refused_not_stamped() -> None:
    """Stamping would assert something about trials that is unknown.

    Those trials ran at settings nobody recorded. Marking the study as
    having run at THIS run's settings would make the pin a false claim, and
    a later resume would then compare against it as if verified.
    """
    study = _study()
    with pytest.raises(ValueError, match="records no joint metric settings"):
        check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=5)
    assert JOINT_METRIC_SETTINGS_ATTR not in study.user_attrs


def test_the_refusal_names_the_escape_hatch() -> None:
    """A study whose settings the user DOES know must be recoverable."""
    study = _study()
    with pytest.raises(ValueError, match=JOINT_METRIC_SETTINGS_ATTR):
        check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=5)

    study.set_user_attr(JOINT_METRIC_SETTINGS_ATTR, TARP)
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=5)


def test_a_study_using_no_joint_metrics_never_acquires_the_attribute() -> None:
    """The guard must not touch the overwhelmingly common case."""
    study = _study()
    check_or_stamp_joint_metric_settings(study, {}, n_completed_trials=9)
    assert JOINT_METRIC_SETTINGS_ATTR not in study.user_attrs


def test_a_metric_added_mid_study_is_recorded_alongside() -> None:
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


def test_an_unrecognized_stored_value_is_treated_as_absent() -> None:
    """`user_attrs` is caller-writable and round-trips through JSON.

    Refusing on a value this code cannot interpret would block a study over
    something it cannot even describe -- the same choice the objective
    schema guard makes.
    """
    study = _study()
    study.set_user_attr(JOINT_METRIC_SETTINGS_ATTR, "not a mapping")
    check_or_stamp_joint_metric_settings(study, TARP, n_completed_trials=0)
    assert study.user_attrs[JOINT_METRIC_SETTINGS_ATTR] == TARP


def test_the_attribute_survives_a_json_round_trip() -> None:
    """It is stored by Optuna, so it has to be JSON-serializable."""
    import json

    pytest.importorskip("sklearn", reason="the L-C2ST factory guards on it")

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


def test_the_settings_come_from_the_callable_that_ran() -> None:
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
        # Recorded even when unset, so a study that starts labelling its
        # reference reads as changed -- which it is, in the only sense the
        # pin can check.
        "reference_id": None,
    }


def test_a_metric_declaring_nothing_is_omitted_not_recorded_as_empty() -> None:
    """Absent means "makes no claim", which is not "configured with nothing"."""
    assert joint_metric_settings({"plain": lambda inputs: {}}) == {}


def test_the_reference_mode_is_declared_from_how_it_was_built() -> None:
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    provided = make_tarp_joint_metric(reference_points=lambda i: None)
    random = make_tarp_joint_metric()
    assert provided.joint_metric_settings["reference_mode"] == "provided"
    assert random.joint_metric_settings["reference_mode"] == "random"


def test_the_registered_lc2st_declares_the_factory_defaults() -> None:
    """Otherwise the pin covers a configured L-C2ST but not the registry's.

    A study stamped by `make_lc2st_validate_fn(n_folds=10)` and then resumed
    with plain `objective_metrics=["lc2st"]` would declare nothing, so
    `check_or_stamp_joint_metric_settings` would return early and the two
    scales would mix unchecked.

    The two are spelled separately -- the registered name cannot call the
    factory, which guards on scikit-learn -- so they can drift. This is what
    notices, and it needs the factory, hence the skip: the registered name
    itself stays importable without scikit-learn, which
    `test_importing_the_package_does_not_require_sklearn` covers.
    """
    pytest.importorskip("sklearn", reason="the L-C2ST factory guards on it")

    from bayesflow_hpo.validation.c2st import (
        _default_lc2st_metric,
        make_lc2st_joint_metric,
    )

    assert (
        _default_lc2st_metric.joint_metric_settings
        == make_lc2st_joint_metric().joint_metric_settings
    )


def test_every_runnable_registered_joint_metric_declares_settings() -> None:
    """A metric that declares nothing is silently exempt from the pin.

    "Runnable" excludes the placeholders that `resolve_joint_metrics`
    refuses outright -- `tarp_error` is registered so the routing surface
    knows the name, but cannot run until a caller configures it, and the
    configured callable is what declares. A placeholder can never reach the
    pin, so it has nothing to declare; anything else that declares nothing
    would be exempt by accident.
    """
    from bayesflow_hpo.validation.registry import _JOINT, get_metric

    runnable = {
        name: get_metric(name)
        for name in _JOINT
        if not getattr(get_metric(name), "_bf_hpo_requires_configuration", None)
    }
    assert runnable, "no runnable joint metrics found; the filter is wrong"

    undeclared = sorted(
        name
        for name, fn in runnable.items()
        if not getattr(fn, "joint_metric_settings", None)
    )
    assert undeclared == [], (
        f"registered joint metrics {undeclared} declare no settings, so a "
        "study using them records nothing and compares nothing"
    )


def test_a_labelled_reference_is_pinned() -> None:
    """What `reference_mode="provided"` alone cannot say: WHICH provider.

    A callable is not serializable, so two studies both reporting
    `tarp_error` may have used entirely different providers and the
    settings check passes. Nothing can derive the identity automatically --
    a provider's output depends on the data it is given -- so the caller
    supplies it, and supplying it turns the pin from "a provider was used"
    into "this provider was used".
    """
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    a = make_tarp_joint_metric(
        reference_points=lambda i: None, reference_id="item-difficulty-v1"
    )
    b = make_tarp_joint_metric(
        reference_points=lambda i: None, reference_id="item-difficulty-v2"
    )
    assert a.joint_metric_settings["reference_id"] == "item-difficulty-v1"
    assert (
        a.joint_metric_settings["reference_id"]
        != b.joint_metric_settings["reference_id"]
    ), "two labelled references must be distinguishable by the pin"

    study = _study()
    check_or_stamp_joint_metric_settings(
        study, {"tarp_error": a.joint_metric_settings}, n_completed_trials=0
    )
    with pytest.raises(ValueError, match="reference_id"):
        check_or_stamp_joint_metric_settings(
            study,
            {"tarp_error": b.joint_metric_settings},
            n_completed_trials=1,
        )


# ---------------------------------------------------------------------------
# Which trials count as "completed"
# ---------------------------------------------------------------------------


def test_budget_rejected_trials_do_not_block_the_first_stamp() -> None:
    """A rejected proposal is COMPLETE but measured nothing.

    A trial rejected for `max_memory_mb` or `max_param_count` returns
    `_penalty()` and Optuna records it COMPLETE. Counting it made a fresh
    study whose FIRST proposal was oversized -- routine early in a search --
    reach the next, feasible trial with a positive count and no stored
    settings, so the guard aborted the whole study over trials that never
    ran a metric.
    """
    from bayesflow_hpo.optimization.objective import _n_measured_trials

    study = _study()
    for reason in ("memory_budget", "param_budget"):
        t = study.ask()
        t.set_user_attr("rejected_reason", reason)
        study.tell(t, 1.0)

    assert _n_measured_trials(study) == 0
    check_or_stamp_joint_metric_settings(
        study, TARP, n_completed_trials=_n_measured_trials(study)
    )
    assert study.user_attrs[JOINT_METRIC_SETTINGS_ATTR] == TARP


def test_failed_and_fallback_trials_do_not_count_either() -> None:
    """They reach COMPLETE without producing measured joint values."""
    from bayesflow_hpo.optimization.objective import _n_measured_trials

    study = _study()
    for marker in ("training_error", "validation_error"):
        t = study.ask()
        t.set_user_attr(marker, "boom")
        study.tell(t, 1.0)
    assert _n_measured_trials(study) == 0


def test_a_measured_trial_does_count() -> None:
    """The legacy-study guard has to keep working, or this is a hole."""
    from bayesflow_hpo.optimization.objective import _n_measured_trials

    study = _study()
    study.tell(study.ask(), 1.0)
    assert _n_measured_trials(study) == 1

    with pytest.raises(ValueError, match="records no joint metric settings"):
        check_or_stamp_joint_metric_settings(
            study, TARP, n_completed_trials=_n_measured_trials(study)
        )

