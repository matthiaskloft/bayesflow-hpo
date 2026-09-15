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


class _RealApproximator:
    """Returns per-parameter draws, so the real pipeline actually runs.

    The whole point: `_run_lightweight_validation` wraps its body in a broad
    `except Exception`, so a mock that makes the pipeline throw turns any
    assertion about what the pipeline did into a vacuous one.
    """

    def __init__(self, param_keys, n_sims):
        self.param_keys = param_keys
        self.n_sims = n_sims

    def get_weights(self):
        return []

    def sample(self, *, conditions, num_samples):
        rng = np.random.default_rng(0)
        return {
            k: rng.normal(size=(self.n_sims, num_samples, 1))
            for k in self.param_keys
        }


def _pipeline_callback(joint_name, **kwargs):
    """A callback with NO validate_fn, so the pipeline branch is exercised."""
    params = dict(
        trial=_trial(),
        approximator=_RealApproximator(["theta"], 8),
        validation_data=_dataset(),
        interval=1,
        warmup=0,
        n_posterior_samples=16,
        validate_fn=None,
        objective_metrics=["nrmse", joint_name],
    )
    params.update(kwargs)
    return PeriodicValidationCallback(**params)



# ---------------------------------------------------------------------------
# The regression this exists to prevent
# ---------------------------------------------------------------------------


def test_a_mixed_study_still_prunes_and_still_stops_early(joint_metric) -> None:
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


def test_the_joint_metric_is_not_computed_at_an_interval(joint_metric) -> None:
    """The point of the exclusion: it must not be paid for per interval.

    Driven through the REAL pipeline with a working approximator. An
    earlier version of this test used a MagicMock, which made the pipeline
    throw, `_run_lightweight_validation`'s broad `except Exception` swallow
    it, and `calls == []` hold for the wrong reason -- it passed with the
    exclusion removed entirely.
    """
    calls: list[int] = []

    def expensive(inputs):
        calls.append(inputs.cond_id)
        return {"joint_slow": 0.1}

    joint_metric("joint_slow", expensive)
    cb = _pipeline_callback("joint_slow")
    cb.on_epoch_end(0)

    assert cb._step == 1, (
        "intermediate validation did not complete, so this test would pass "
        "for the wrong reason"
    )
    assert calls == [], "the joint metric ran during intermediate validation"


def test_the_pipeline_branch_still_prunes_and_stops_early(joint_metric) -> None:
    """The default branch: no `validate_fn`, so the pipeline runs directly.

    `_run_lightweight_validation` has TWO missing-key checks, one per
    branch. The `validate_fn` branch was fixed to require only the
    intermediate set; the pipeline branch was not, so it demanded the
    excluded joint key, found it absent by design, returned None on every
    interval, and `on_epoch_end` bailed before pruning AND
    `_update_early_stopping` -- the exact regression the explicit set exists
    to prevent, on the branch most studies take.
    """
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    cb = _pipeline_callback(
        "joint_slow",
        early_stopping_patience=2,
        early_stopping_monitor="nrmse",
    )

    cb.on_epoch_end(0)

    assert cb._step == 1
    assert cb._consecutive_failures == 0
    assert cb._early_stopping_values, (
        "early stopping never received a value on the pipeline branch"
    )


def test_opting_in_computes_it(joint_metric) -> None:
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


def test_a_joint_only_study_is_rejected_up_front(joint_metric) -> None:
    """Not degraded into a study that silently cannot stop early."""
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    with pytest.raises(ValueError, match="could neither prune nor stop early"):
        _callback("joint_slow", objective_metrics=["joint_slow"])


def test_a_joint_only_study_is_allowed_when_opted_into(joint_metric) -> None:
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    cb = _callback(
        "joint_slow",
        objective_metrics=["joint_slow"],
        include_joint_metrics=True,
    )
    assert [str(m) for m in cb.intermediate_metrics] == ["joint_slow"]


def test_monitoring_an_excluded_joint_metric_is_rejected(joint_metric) -> None:
    """Nothing would ever be monitored, so early stopping could never fire."""
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    with pytest.raises(ValueError, match="nothing would ever be monitored"):
        _callback(
            "joint_slow",
            early_stopping_patience=2,
            early_stopping_monitor="joint_slow",
        )


def test_a_primary_metric_that_is_excluded_is_rejected(joint_metric) -> None:
    """No pruning decision could ever be made on a metric never computed."""
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    with pytest.raises(ValueError, match="no pruning decision"):
        _callback(
            "joint_slow",
            pruning_strategy=("primary", "joint_slow"),
        )


def test_objective_mean_says_that_its_members_changed(joint_metric, caplog) -> None:
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


def test_objective_mean_averages_only_the_intermediate_metrics(joint_metric) -> None:
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


def test_a_marginal_only_study_is_unchanged() -> None:
    """The common case must behave exactly as before."""
    cb = _callback("nrmse", objective_metrics=["nrmse", "calibration_error"])
    assert [str(m) for m in cb.intermediate_metrics] == [
        "nrmse",
        "calibration_error",
    ]
    assert cb.intermediate_metrics == cb.objective_metrics


def test_a_bare_primary_strategy_defaulting_onto_an_excluded_metric_is_caught(
    joint_metric,
) -> None:
    """`pruning_strategy="primary"` leaves the metric to be defaulted.

    The tuple form names the metric up front; the bare string defaults it to
    `objective_metrics[0]`. If that default lands on an excluded joint
    metric and the guard has already run, nothing refuses it and the failure
    surfaces later as an uncaught KeyError from `_evaluate_pruning`,
    mid-training.
    """
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    with pytest.raises(ValueError, match="no pruning decision"):
        _callback(
            "joint_slow",
            objective_metrics=["joint_slow", "nrmse"],
            pruning_strategy="primary",
        )


def test_the_tuple_primary_form_is_still_caught(joint_metric) -> None:
    joint_metric("joint_slow", lambda inputs: {"joint_slow": 0.1})
    with pytest.raises(ValueError, match="no pruning decision"):
        _callback("joint_slow", pruning_strategy=("primary", "joint_slow"))


# ---------------------------------------------------------------------------
# The exclusion must cover overrides, not just the registry route
# ---------------------------------------------------------------------------


def test_an_overridden_joint_metric_is_excluded_too(joint_metric) -> None:
    """`joint_metrics=` bypasses the `metrics=` filter entirely.

    `run_validation_pipeline` merges the override dict UNCONDITIONALLY,
    independent of `metrics` -- that is how `make_lc2st_validate_fn` adds a
    metric its own list omits. So filtering `metrics` excludes nothing that
    arrives by override, which is every metric that needs one at all:
    `tarp_error`, or a configured L-C2ST. The callback logged the metric as
    excluded, computed it anyway at ~56 s per condition, and then discarded
    the value, because the extraction keys on `intermediate_metrics`.
    """
    calls: list[int] = []

    def expensive(inputs):
        calls.append(inputs.cond_id)
        return {"joint_slow": 0.1}

    joint_metric("joint_slow", expensive)
    cb = _pipeline_callback(
        "joint_slow", joint_metrics={"joint_slow": expensive}
    )
    cb.on_epoch_end(0)

    assert cb._step == 1, "validation did not complete; the test is vacuous"
    assert calls == [], (
        "an overridden joint metric ran during intermediate validation "
        "despite being excluded from the intermediate set"
    )


def test_opting_in_runs_the_overridden_metric(joint_metric) -> None:
    """The escape hatch has to work through the override route as well."""
    calls: list[int] = []

    def expensive(inputs):
        calls.append(inputs.cond_id)
        return {"joint_slow": 0.1}

    joint_metric("joint_slow", expensive)
    cb = _pipeline_callback(
        "joint_slow",
        joint_metrics={"joint_slow": expensive},
        include_joint_metrics=True,
    )
    cb.on_epoch_end(0)

    assert calls, "opting in did not run the overridden metric"
    assert cb._step == 1


def test_include_joint_metrics_is_reachable_from_optimize() -> None:
    """Three of this callback's error messages tell the caller to set it."""
    import inspect

    from bayesflow_hpo import optimize

    assert "include_joint_metrics" in inspect.signature(optimize).parameters

