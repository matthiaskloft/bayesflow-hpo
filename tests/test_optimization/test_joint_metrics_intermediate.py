"""Joint metrics under ``PeriodicValidationCallback``.

The measured cost says joint metrics should not run at every pruning
interval (L-C2ST: ~54 s per condition, so ~18 minutes per interval on a
20-condition grid). But excluding them is not free of consequences, and the
consequence is the subtle part: ``_run_lightweight_validation`` requires
EVERY entry of ``objective_metrics`` and returns ``None`` when any is
missing, and ``on_epoch_end`` then returns before both pruning and
``_update_early_stopping``. A naive exclusion therefore turns off marginal
pruning and validation early stopping too.

Design: ``docs/plans/plan-joint-metric-path.md`` D9.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import optuna
import pytest

from bayesflow_hpo.optimization.validation_callback import (
    PeriodicValidationCallback,
)
from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.registry import (
    _JOINT,
    _REGISTRY,
    register_joint_metric,
)


@pytest.fixture
def joint_metric():
    registered: list[str] = []

    def _register(name: str, fn, **kwargs) -> str:
        register_joint_metric(name, fn, **kwargs)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        _REGISTRY.pop(name, None)
        _JOINT.discard(name)


def _trial(n_objectives: int = 2) -> optuna.Trial:
    study = optuna.create_study(directions=["minimize"] * n_objectives)
    return study.ask()


def _dataset() -> ValidationDataset:
    rng = np.random.default_rng(0)
    return ValidationDataset(
        simulations=[{"theta": rng.normal(size=8), "x": rng.normal(size=(8, 2))}],
        condition_labels=[{}],
        param_keys=["theta"],
        data_keys=["x"],
        seed=0,
    )


def _callback(joint_name, **kwargs):
    """Build a callback whose validate_fn returns MARGINAL metrics only.

    That is the situation the exclusion creates: the joint key is genuinely
    absent from the intermediate summary, and the callback has to treat that
    as expected rather than as a failed validation.
    """
    approximator = MagicMock()
    approximator.get_weights.return_value = []

    def validate_fn(approx, data, n_samples):
        return {"nrmse": 0.5}

    params = dict(
        trial=_trial(),
        approximator=approximator,
        validation_data=_dataset(),
        interval=1,
        warmup=0,
        validate_fn=validate_fn,
        objective_metrics=["nrmse", joint_name],
    )
    params.update(kwargs)
    return PeriodicValidationCallback(**params)


# ---------------------------------------------------------------------------
# The regression this exists to prevent
# ---------------------------------------------------------------------------


def test_a_mixed_study_still_prunes_and_still_stops_early(joint_metric):
    """Excluding the joint metric must not disable the marginal machinery.

    Without an explicit intermediate set, the absent joint key makes
    `_run_lightweight_validation` return None, `on_epoch_end` bails, and
    NEITHER pruning nor early stopping ever runs -- silently, behind a
    warning log.
    """
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    cb = _callback(
        "joint_slow",
        early_stopping_patience=2,
        early_stopping_monitor="nrmse",
    )

    assert [str(m) for m in cb.intermediate_metrics] == ["nrmse"]

    cb.on_epoch_end(0)

    # The validation counted: a step was taken, no failure recorded, and
    # early stopping saw a value.
    assert cb._step == 1
    assert cb._consecutive_failures == 0
    assert cb._early_stopping_values, (
        "early stopping never received a value, so the trial can no longer "
        "stop on validation"
    )
    assert np.isfinite(cb.best_validation_score)


def test_the_joint_metric_is_not_computed_at_an_interval(joint_metric):
    """The point of the exclusion: it must not be paid for per interval."""
    calls: list[int] = []

    def expensive(inputs):
        calls.append(inputs.cond_id)
        return {"joint_slow": 0.1}

    joint_metric("joint_slow", expensive)
    cb = _callback("joint_slow", validate_fn=None)
    cb.on_epoch_end(0)

    assert calls == [], "the joint metric ran during intermediate validation"


def test_opting_in_computes_it(joint_metric):
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    cb = _callback(
        "joint_slow",
        validate_fn=lambda a, d, n: {"nrmse": 0.5, "joint_slow": 0.1},
        include_joint_metrics=True,
    )
    assert [str(m) for m in cb.intermediate_metrics] == [
        "nrmse",
        "joint_slow",
    ]
    cb.on_epoch_end(0)
    assert cb._step == 1


# ---------------------------------------------------------------------------
# The three configurations that must be specified, not discovered
# ---------------------------------------------------------------------------


def test_a_joint_only_study_is_rejected_up_front(joint_metric):
    """Not degraded into a study that silently cannot stop early."""
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    with pytest.raises(ValueError, match="could neither prune nor stop early"):
        _callback("joint_slow", objective_metrics=["joint_slow"])


def test_a_joint_only_study_is_allowed_when_opted_into(joint_metric):
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    cb = _callback(
        "joint_slow",
        objective_metrics=["joint_slow"],
        include_joint_metrics=True,
    )
    assert [str(m) for m in cb.intermediate_metrics] == ["joint_slow"]


def test_monitoring_an_excluded_joint_metric_is_rejected(joint_metric):
    """Nothing would ever be monitored, so early stopping could never fire."""
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    with pytest.raises(ValueError, match="nothing would ever be monitored"):
        _callback(
            "joint_slow",
            early_stopping_patience=2,
            early_stopping_monitor="joint_slow",
        )


def test_a_primary_metric_that_is_excluded_is_rejected(joint_metric):
    """No pruning decision could ever be made on a metric never computed."""
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    with pytest.raises(ValueError, match="no pruning decision"):
        _callback(
            "joint_slow",
            pruning_strategy=("primary", "joint_slow"),
        )


def test_objective_mean_says_that_its_members_changed(joint_metric, caplog):
    """Not an error, but it silently changes what is being monitored.

    The mid-training mean is taken over the remaining metrics only, so it is
    not on the same scale as the final objective mean.
    """
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    with caplog.at_level("INFO"):
        _callback(
            "joint_slow",
            early_stopping_patience=2,
            early_stopping_monitor="objective_mean",
        )
    assert "objective_mean" in caplog.text
    assert "joint_slow" in caplog.text


def test_objective_mean_averages_only_the_intermediate_metrics(joint_metric):
    """The mean must not index a key the intermediate summary lacks."""
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    cb = _callback(
        "joint_slow",
        early_stopping_patience=2,
        early_stopping_monitor="objective_mean",
    )
    cb.on_epoch_end(0)
    # nrmse is lower-is-better and passes through unchanged.
    assert cb._early_stopping_values == [pytest.approx(0.5)]


def test_a_marginal_only_study_is_unchanged():
    """The common case must behave exactly as before."""
    cb = _callback("nrmse", objective_metrics=["nrmse", "calibration_error"])
    assert [str(m) for m in cb.intermediate_metrics] == [
        "nrmse",
        "calibration_error",
    ]
    assert cb.intermediate_metrics == cb.objective_metrics
