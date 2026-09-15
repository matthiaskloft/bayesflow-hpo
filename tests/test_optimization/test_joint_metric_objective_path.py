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
