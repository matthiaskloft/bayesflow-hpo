"""Joint metrics as objectives, through `GenericObjective` rather than around it.

Every other test of this feature calls the pieces directly. That is what let
two P1 defects survive a review round: the pieces were each correct and the
path through them was not.

- `check_or_stamp_joint_metric_settings` raised inside the objective's
  `except Exception`, which converts anything it catches into a
  training-loss fallback. The guard therefore did not stop the study; it
  replaced a real-but-incomparable number with a fabricated one and let the
  run continue, which is worse than not guarding.
- `tarp_error` was registered, resolvable, `kind="objective"` -- and
  unusable, because the caller's configured metric was never consulted:
  `resolve_joint_metrics` refused the placeholder before the override
  merged, and `optimize()` had no parameter to supply one with.

Design: ``docs/plans/plan-joint-metric-path.md`` D6, D7.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pytest

from bayesflow_hpo.objectives import (
    JOINT_METRIC_SETTINGS_ATTR,
    check_or_stamp_joint_metric_settings,
)
from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.pipeline import run_validation_pipeline
from bayesflow_hpo.validation.registry import (
    JointMetricConfigurationError,
    resolve_joint_metrics,
)
from bayesflow_hpo.validation.tarp import make_tarp_joint_metric


class _Approximator:
    def __init__(self, param_keys: list[str], n_sims: int) -> None:
        self.param_keys = param_keys
        self.n_sims = n_sims

    def get_weights(self) -> list[Any]:
        return []

    def sample(
        self, *, conditions: Any, num_samples: int
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        return {
            k: rng.normal(size=(self.n_sims, num_samples, 1))
            for k in self.param_keys
        }


def _dataset(n_conditions: int = 2, n_sims: int = 30) -> ValidationDataset:
    rng = np.random.default_rng(1)
    sims = [
        {"a": rng.normal(size=n_sims), "b": rng.normal(size=n_sims),
         "x": rng.normal(size=(n_sims, 2))}
        for _ in range(n_conditions)
    ]
    return ValidationDataset(
        simulations=sims,
        condition_labels=[{"c": i} for i in range(n_conditions)],
        param_keys=["a", "b"],
        data_keys=["x"],
        seed=0,
    )


# ---------------------------------------------------------------------------
# `tarp_error` is actually usable as an objective
# ---------------------------------------------------------------------------


def test_a_configured_tarp_error_overrides_its_placeholder() -> None:
    """The remedy `tarp_error`'s own error message prescribes must work.

    The objective path unions the objective names into `metrics=`, so
    `tarp_error` is always in BOTH lists. Resolving before merging the
    override meant the placeholder raised first and the caller's metric --
    the entire point -- was never consulted.
    """
    data = _dataset()
    configured = make_tarp_joint_metric(
        reference_points=lambda inputs: np.asarray(
            inputs.sim_batch["x"]
        )[:, :2]
    )

    result = run_validation_pipeline(
        approximator=_Approximator(["a", "b"], 30),
        validation_data=data,
        n_posterior_samples=32,
        metrics=["nrmse", "tarp_error"],
        joint_metrics={"tarp_error": configured},
    )

    assert "tarp_error" in result.summary, result.failed_joint_metrics
    assert np.isfinite(result.summary["tarp_error"])
    assert result.failed_joint_metrics == {}


def test_without_an_override_the_placeholder_still_refuses() -> None:
    """The refusal must survive the override mechanism being added."""
    with pytest.raises(JointMetricConfigurationError, match="reference points"):
        run_validation_pipeline(
            approximator=_Approximator(["a", "b"], 30),
            validation_data=_dataset(),
            n_posterior_samples=32,
            metrics=["nrmse", "tarp_error"],
        )


def test_the_override_is_not_resolved_from_the_registry() -> None:
    """Directly: an overridden name must not be looked up at all."""
    assert resolve_joint_metrics(["tarp_error"], overridden=["tarp_error"]) == {}
    with pytest.raises(JointMetricConfigurationError):
        resolve_joint_metrics(["tarp_error"])


def test_optimize_exposes_a_route_for_configured_joint_metrics() -> None:
    """Without a parameter on the public API the feature is unreachable."""
    import inspect

    from bayesflow_hpo import optimize

    assert "joint_metrics" in inspect.signature(optimize).parameters


# ---------------------------------------------------------------------------
# A configuration error must not become a per-trial penalty
# ---------------------------------------------------------------------------


def test_a_settings_mismatch_raises_a_type_the_objective_re_raises() -> None:
    """`GenericObjective` converts a caught exception into a fallback score.

    So this refusal has to be a type it re-raises explicitly. Were it a
    plain ValueError, the study would record a fabricated value for this
    trial and every trial after it -- the condition is a property of the
    STUDY, so it never clears -- behind a single warning line.
    """
    study = optuna.create_study(directions=["minimize"])
    study.set_user_attr(
        JOINT_METRIC_SETTINGS_ATTR, {"tarp_error": {"resolution": 20}}
    )
    with pytest.raises(JointMetricConfigurationError):
        check_or_stamp_joint_metric_settings(
            study, {"tarp_error": {"resolution": 100}}, n_completed_trials=1
        )


def test_the_objective_re_raises_it_instead_of_penalizing() -> None:
    """Asserted against the source, since the handler order is the fix."""
    import inspect

    from bayesflow_hpo.optimization import objective as objective_module

    source = inspect.getsource(objective_module.GenericObjective)
    reraise = source.index("except JointMetricConfigurationError:")
    catchall = source.index("except Exception as exc:", reraise)
    assert reraise < catchall, (
        "the configuration handler must precede the catch-all, or the "
        "catch-all converts the refusal into a training-loss fallback"
    )


def test_a_missing_optional_dependency_refuses_at_resolve_time() -> None:
    """Otherwise the study trains to completion while optimizing a constant.

    A missing scikit-learn made `lc2st` raise once per condition, which the
    joint guard caught and turned into the registered worst case of 0.25 --
    identical on every trial. The study runs its full budget optimizing a
    constant, with only a warning log to say so.

    It must also surface as a CONFIGURATION error rather than the bare
    ImportError: the objective re-raises only that type ahead of its
    catch-all, so an ImportError would be converted into a training-loss
    fallback and the study would carry on regardless -- which is the same
    defect one layer out.
    """
    from unittest.mock import patch

    with patch(
        "bayesflow_hpo.validation.c2st._require_sklearn",
        side_effect=ImportError("C2ST metrics require scikit-learn"),
    ):
        with pytest.raises(
            JointMetricConfigurationError, match="scikit-learn"
        ) as excinfo:
            resolve_joint_metrics(["lc2st"])
    # The original is kept, so the traceback still names the real cause.
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_a_present_dependency_resolves_normally() -> None:
    pytest.importorskip("sklearn")
    assert set(resolve_joint_metrics(["lc2st"])) == {"lc2st"}


# ---------------------------------------------------------------------------
# A failed joint metric must not read as a satisfied constraint
# ---------------------------------------------------------------------------


def test_a_failed_constraint_metric_rejects_a_hard_constrained_trial() -> None:
    """A constraint names an output key, and a missing key read as "fine".

    When the constrained metric is a joint one that FAILED, the key is
    missing because nothing measured it -- not because the constraint was
    satisfied. The hard path skipped it with a warning, so the trial
    completed on its good NRMSE and was classified feasible on the strength
    of a measurement that never happened.
    """
    from bayesflow_hpo.optimization.objective import (
        GenericObjective,
        ObjectiveConfig,
    )

    study = optuna.create_study(directions=["minimize", "minimize"])
    trial = study.ask()
    trial.set_user_attr("failed_metric_keys", ["tarp_error_random"])

    config = ObjectiveConfig(
        simulator=None,
        adapter=None,
        search_space=None,
        validation_data=_dataset(),
        objective_metrics=["nrmse"],
        metric_constraints_hard=[("tarp_error_random", 0.1, "above")],
    )
    objective = GenericObjective(config)

    penalty = objective._check_hard_constraints({"nrmse": 0.2}, trial)
    assert penalty is not None, (
        "an unmeasured hard constraint let the trial through as feasible"
    )
    assert trial.user_attrs.get("rejected_reason") == "metric_constraint"


def test_a_measured_constraint_metric_still_passes() -> None:
    """The guard must not reject a constraint that simply was not requested."""
    from bayesflow_hpo.optimization.objective import (
        GenericObjective,
        ObjectiveConfig,
    )

    study = optuna.create_study(directions=["minimize", "minimize"])
    trial = study.ask()

    config = ObjectiveConfig(
        simulator=None,
        adapter=None,
        search_space=None,
        validation_data=_dataset(),
        objective_metrics=["nrmse"],
        metric_constraints_hard=[("tarp_error_random", 0.9, "above")],
    )
    objective = GenericObjective(config)

    # Present and within bounds.
    assert objective._check_hard_constraints(
        {"nrmse": 0.2, "tarp_error_random": 0.1}, trial
    ) is None
    # Absent and NOT marked failed: skipped, as before.
    assert objective._check_hard_constraints({"nrmse": 0.2}, trial) is None


def test_an_unmeasured_soft_constraint_reports_a_positive_violation() -> None:
    """Zero violation means satisfied, which is what was reported."""
    from bayesflow_hpo.optimization.study import _make_constraints_func

    constraints = _make_constraints_func(
        budget_aware=False,
        soft_thresholds=[("tarp_error_random", 0.1, "above")],
    )

    study = optuna.create_study(directions=["minimize"])
    study.tell(study.ask(), 1.0)
    failed, measured = study.trials[0], study.trials[0]

    failed = optuna.trial.create_trial(
        params={},
        distributions={},
        value=1.0,
        user_attrs={"failed_metric_keys": ["tarp_error_random"]},
    )
    assert constraints(failed)[0] > 0.0, (
        "an unmeasured soft constraint reported zero violation, which "
        "Optuna reads as feasible"
    )

    measured = optuna.trial.create_trial(
        params={}, distributions={}, value=1.0,
        user_attrs={"tarp_error_random": 0.05},
    )
    assert constraints(measured)[0] == 0.0

    unrequested = optuna.trial.create_trial(
        params={}, distributions={}, value=1.0, user_attrs={},
    )
    assert constraints(unrequested)[0] == 0.0


def test_the_training_path_re_raises_configuration_errors() -> None:
    """`PeriodicValidationCallback` runs DURING training.

    Its re-raise is undone by the catch-all around training, which records
    a `training_error` and returns `_penalty()` -- so a misconfiguration
    consumes the study's whole trial cap instead of stopping on the first.
    Now reachable in practice, because joint metrics no longer run in
    pre-flight.
    """
    import inspect

    from bayesflow_hpo.optimization import objective as objective_module

    source = inspect.getsource(objective_module.GenericObjective)
    training = source.index("failed during training")
    reraise = source.rindex("except JointMetricConfigurationError:", 0, training)
    catchall = source.rindex("except Exception as exc:", 0, training)
    assert reraise < catchall, (
        "the configuration handler must precede the training catch-all, or "
        "the catch-all turns the refusal into a per-trial penalty"
    )


# ---------------------------------------------------------------------------
# Configuration is rejected before it costs a training run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "kwargs", "match"),
    [
        ("tarp", {"metric": "l2"}, "euclidean"),
        ("tarp", {"resolution": 0}, "resolution must be at least 1"),
        ("lc2st", {"max_conditions": 0}, "max_conditions must be at least 1"),
        ("lc2st", {"n_folds": 1}, "n_folds must be at least 2"),
        ("lc2st", {"n_null_trials": -1}, "non-negative"),
    ],
)
def test_invalid_factory_settings_are_rejected_at_construction(
    factory: str, kwargs: dict[str, Any], match: str
) -> None:
    """None of these depend on the data, so none need data to be checked.

    Left to the per-condition numerical guard, a typo becomes an exception
    the guard converts into the metric's registered worst case -- so every
    trial trains to completion and scores an identical penalty, and the
    study optimizes a constant behind a warning log. The low-level
    functions do reject these, but only once called.
    """
    if factory == "lc2st":
        pytest.importorskip("sklearn")
        from bayesflow_hpo.validation.c2st import (
            make_lc2st_joint_metric as make,
        )
    else:
        make = make_tarp_joint_metric

    with pytest.raises(ValueError, match=match):
        make(**kwargs)


def test_valid_factory_settings_still_construct() -> None:
    """The guard must not reject the boundary values themselves."""
    assert make_tarp_joint_metric(metric="manhattan", resolution=1) is not None


# ---------------------------------------------------------------------------
# The resume guard must fire before pruning can exit the trial
# ---------------------------------------------------------------------------


def test_planned_settings_match_what_validation_will_declare() -> None:
    """The pre-training check is only useful if it predicts the real thing.

    It is computed from the resolved callables and the config; the
    post-validation check is computed from the metrics that actually ran. If
    the two disagreed, the early check would either refuse valid resumes or
    stamp settings the run then contradicts.
    """
    from bayesflow_hpo.optimization.objective import (
        ObjectiveConfig,
        _planned_joint_settings,
    )
    from bayesflow_hpo.validation.pipeline import VALIDATION_RUN_SETTINGS

    data = _dataset(n_conditions=2, n_sims=30)
    configured = make_tarp_joint_metric(
        reference_points=lambda inputs: np.asarray(
            inputs.sim_batch["x"]
        )[:, :2],
        resolution=13,
    )
    config = ObjectiveConfig(
        simulator=None,
        adapter=None,
        search_space=None,
        validation_data=data,
        objective_metrics=["nrmse", "tarp_error"],
        joint_metrics={"tarp_error": configured},
        n_posterior_samples=32,
    )

    planned = _planned_joint_settings(config)

    actual = run_validation_pipeline(
        approximator=_Approximator(["a", "b"], 30),
        validation_data=data,
        n_posterior_samples=32,
        metrics=["nrmse", "tarp_error"],
        joint_metrics={"tarp_error": configured},
    ).joint_metric_settings

    assert planned == actual, (
        "the pre-training prediction disagrees with what validation "
        "declares, so the early check would refuse valid resumes or stamp "
        "settings the run contradicts"
    )
    assert planned["tarp_error"]["resolution"] == 13
    assert planned[VALIDATION_RUN_SETTINGS] == {
        "n_posterior_samples": 32,
        "n_conditions": 2,
    }


def test_a_study_with_no_joint_metrics_plans_nothing() -> None:
    """The common case must not acquire the attribute via the early check."""
    from bayesflow_hpo.optimization.objective import (
        ObjectiveConfig,
        _planned_joint_settings,
    )

    config = ObjectiveConfig(
        simulator=None,
        adapter=None,
        search_space=None,
        validation_data=_dataset(),
        objective_metrics=["nrmse"],
    )
    assert _planned_joint_settings(config) == {}


def test_a_custom_validate_fn_plans_nothing() -> None:
    """The hook owns its validation step and declares no settings."""
    from bayesflow_hpo.optimization.objective import (
        ObjectiveConfig,
        _planned_joint_settings,
    )

    config = ObjectiveConfig(
        simulator=None,
        adapter=None,
        search_space=None,
        validation_data=_dataset(),
        objective_metrics=["nrmse", "tarp_error"],
        joint_metrics={"tarp_error": make_tarp_joint_metric(
            reference_points=lambda inputs: None
        )},
        validate_fn=lambda a, d, n: {"nrmse": 0.1},
    )
    assert _planned_joint_settings(config) == {}


def test_the_settings_check_precedes_training_in_the_source() -> None:
    """Order is the fix, so order is what is asserted.

    A resumed study with `include_joint_metrics=True` lets the callback
    compute the new-scale statistic, report it, and prune -- exiting the
    trial before a post-training check ever runs. The run could then spend
    its whole budget pruning without issuing the incompatibility error.
    """
    import inspect

    from bayesflow_hpo.optimization import objective as objective_module

    source = inspect.getsource(objective_module.GenericObjective)
    check = source.index("check_or_stamp_joint_metric_settings(")
    training = source.index("Step 7: TRAIN")
    assert check < training, (
        "the settings check must precede training, or intermediate pruning "
        "can exit the trial before the incompatibility is ever noticed"
    )

