"""The joint, data-dependent metric path.

Covers the three failure modes the design identifies, each of which produces
a metric that *looks* configured and silently never reports:

1. the 3-D shape invariant, which two separate squeezes break;
2. a joint key vanishing from the multi-parameter summary;
3. a per-condition failure being averaged away instead of penalized.

Design: ``docs/plans/plan-joint-metric-path.md``, decisions D2, D3, D4, D8.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn
from bayesflow_hpo.validation.pipeline import run_validation_pipeline
from bayesflow_hpo.validation.registry import (
    JointMetricInputs,
    is_joint_metric,
    register_joint_metric,
    register_metric,
    resolve_joint_metrics,
    resolve_metrics,
    unregister_metric,
)
from bayesflow_hpo.validation.result import ValidationResult


@pytest.fixture
def joint_metric() -> Iterator[Callable[..., str]]:
    """Register joint metrics for one test and remove them afterwards.

    Teardown goes through `unregister_metric`, not `_REGISTRY.pop`:
    registration writes to six tables, and clearing one leaves
    `producer_for_key` resolving a declared output to a metric `get_metric`
    can no longer find, plus a stale `_KINDS` entry that the
    objective-encoding inventory counts as a candidate.
    """
    registered: list[str] = []

    def _register(name: str, fn: Any, **kwargs: Any) -> str:
        register_joint_metric(name, fn, **kwargs)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        unregister_metric(name)


class _FakeApproximator:
    """Returns per-parameter draws the way BayesFlow's sampler does.

    Deliberately NOT a stand-in for ``infer_fn``: the tests below drive the
    real ``make_bayesflow_infer_fn`` closure, because the squeeze that breaks
    the shape invariant lives inside it. A mock at the ``infer_fn`` level
    returning 3-D draws would pass every assertion here while the real
    pipeline failed on every condition.
    """

    def __init__(self, param_keys: list[str], n_sims: int) -> None:
        self.param_keys = param_keys
        self.n_sims = n_sims

    def sample(
        self, *, conditions: Any, num_samples: int
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        return {
            key: rng.normal(size=(self.n_sims, num_samples, 1))
            for key in self.param_keys
        }


def _dataset(
    param_keys: list[str], n_conditions: int, n_sims: int
) -> ValidationDataset:
    rng = np.random.default_rng(1)
    sims = [
        {
            **{pk: rng.normal(size=(n_sims,)) for pk in param_keys},
            "x": rng.normal(size=(n_sims, 3)),
        }
        for _ in range(n_conditions)
    ]
    return ValidationDataset(
        simulations=sims,
        condition_labels=[{"c": i} for i in range(n_conditions)],
        param_keys=list(param_keys),
        data_keys=["x"],
        seed=0,
    )


def _run(
    param_keys: list[str],
    metric_names: list[str],
    n_conditions: int = 3,
    n_sims: int = 8,
    n_samples: int = 16,
) -> ValidationResult:
    data = _dataset(param_keys, n_conditions, n_sims)
    approximator = _FakeApproximator(param_keys, n_sims)
    return run_validation_pipeline(
        approximator=approximator,
        validation_data=data,
        n_posterior_samples=n_samples,
        metrics=metric_names,
    )


# ---------------------------------------------------------------------------
# D2 / D3 -- the shape invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param_keys", [["theta"], ["a", "b"]])
def test_joint_metric_always_receives_three_dimensional_draws(
    joint_metric, param_keys
):
    """The contract promises 3-D unconditionally, including for one parameter.

    ``make_bayesflow_infer_fn`` squeezes the trailing axis for a
    single-parameter study (``validation/inference.py:57-61``) BEFORE the
    pipeline sees the array, and the pipeline squeezes again in its
    single-parameter branch. Either one leaves a joint metric with a 2-D
    array, and a metric that asserts rank then fails on every condition.
    """
    seen: list[tuple[int, ...]] = []

    def record(inputs: JointMetricInputs) -> dict[str, float]:
        assert inputs.draws.ndim == 3, (
            f"joint metric received {inputs.draws.ndim}-D draws"
        )
        seen.append(inputs.draws.shape)
        return {"joint_shape_probe": float(inputs.draws.shape[-1])}

    joint_metric("joint_shape_probe", record)
    result = _run(param_keys, ["nrmse", "joint_shape_probe"])

    assert seen, "joint metric was never dispatched"
    assert all(shape == (8, 16, len(param_keys)) for shape in seen), seen
    assert result.summary["joint_shape_probe"] == float(len(param_keys))


def test_the_closure_really_does_squeeze_for_one_parameter():
    """Anchors the test above to the behaviour it guards against.

    If ``make_bayesflow_infer_fn`` ever stops squeezing, the parametrized
    test would still pass while no longer testing anything.
    """
    infer_fn = make_bayesflow_infer_fn(
        approximator=_FakeApproximator(["theta"], 8),
        param_keys=["theta"],
        data_keys=["x"],
        available_keys={"theta", "x"},
    )
    draws = infer_fn({"theta": np.zeros(8), "x": np.zeros((8, 3))}, 16)
    assert draws.ndim == 2, (
        "the closure no longer squeezes; _joint_draws' reason for existing "
        "has changed and this suite's premise needs rechecking"
    )


def test_marginal_metrics_still_see_two_dimensional_draws():
    """Normalizing for the joint path must not change the marginal one."""
    marginal_ndim: list[int] = []

    def record_marginal(
        draws: np.ndarray, true_values: np.ndarray
    ) -> dict[str, float]:
        marginal_ndim.append(draws.ndim)
        return {"marginal_probe": 0.0}

    register_metric("marginal_probe", record_marginal, overwrite=True)
    try:
        _run(["theta"], ["marginal_probe"])
        assert marginal_ndim and set(marginal_ndim) == {2}
    finally:
        unregister_metric("marginal_probe")


# ---------------------------------------------------------------------------
# D3(b) -- the joint key must survive the multi-parameter summary
# ---------------------------------------------------------------------------


def test_joint_key_survives_the_multi_parameter_summary(joint_metric):
    """The overall summary is built FROM the per-parameter summaries.

    That loop takes its key set from the first parameter's summary, so a
    joint key -- which has no per-parameter value -- would be dropped, the
    objective would find nothing, and every trial would take the same
    penalty: a study silently optimizing a constant.
    """
    joint_metric(
        "joint_const", lambda inputs: {"joint_const": 0.5 + inputs.cond_id}
    )
    result = _run(["a", "b"], ["nrmse", "joint_const"])

    assert result.per_parameter is not None
    assert "joint_const" in result.summary
    # Mean over conditions 0, 1, 2 -> 0.5, 1.5, 2.5.
    assert result.summary["joint_const"] == pytest.approx(1.5)
    # And it stays OUT of the per-parameter tables, which have no joint value
    # to report and would otherwise carry a duplicate someone later averages.
    for param_result in result.per_parameter.values():
        assert "joint_const" not in param_result.summary
    assert "joint_const" not in result.condition_metrics.columns


def test_joint_key_reaches_the_single_parameter_summary(joint_metric):
    joint_metric("joint_one", lambda inputs: {"joint_one": 2.0})
    result = _run(["theta"], ["nrmse", "joint_one"])
    assert result.summary["joint_one"] == pytest.approx(2.0)
    assert "nrmse" in result.summary


def test_the_contract_carries_the_data_and_the_column_order(joint_metric):
    """Every field is load-bearing; L-C2ST and TARP need all of them."""
    captured: list[JointMetricInputs] = []

    def capture(inputs: JointMetricInputs) -> dict[str, float]:
        captured.append(inputs)
        return {"joint_capture": 0.0}

    joint_metric("joint_capture", capture)
    _run(["a", "b"], ["joint_capture"], n_conditions=2)

    first = captured[0]
    assert first.param_keys == ("a", "b")
    assert first.data_keys == ("x",)
    assert set(first.sim_batch) == {"a", "b", "x"}
    assert first.true_values.shape == (8, 2)
    # Column order of true_values must match param_keys, or a joint metric
    # compares a draw of `a` against the truth of `b`.
    np.testing.assert_allclose(
        first.true_values[:, 0], np.asarray(first.sim_batch["a"]).reshape(-1)
    )
    np.testing.assert_allclose(
        first.true_values[:, 1], np.asarray(first.sim_batch["b"]).reshape(-1)
    )
    assert [c.cond_id for c in captured] == [0, 1]
    assert first.approximator is not None


# ---------------------------------------------------------------------------
# D8 -- failure invalidates the metric for the whole trial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failing", [(0,), (2,), (0, 1, 2)], ids=["first", "last", "all"]
)
def test_a_failure_on_any_condition_invalidates_the_whole_trial(
    joint_metric, failing
):
    """All three failure positions must produce the SAME outcome.

    Omitting only the failed condition would make the score depend on which
    condition failed: ``aggregate_condition_rows`` takes its key set from the
    first row, so a failure on condition 0 discards every later success,
    while a failure on condition 2 reports a flattering mean over the
    successes and never reaches the registered worst case at all. A model
    would then benefit from failing on the conditions it finds hardest.
    """

    def sometimes_raises(inputs: JointMetricInputs) -> dict[str, float]:
        if inputs.cond_id in failing:
            raise RuntimeError("constant dimension")
        return {"joint_flaky": 0.01}

    joint_metric("joint_flaky", sometimes_raises)
    result = _run(["a", "b"], ["nrmse", "joint_flaky"])

    assert "joint_flaky" not in result.summary, (
        "a partially computed joint metric reached the objective as a "
        "finite value, so its registered worst case was never applied"
    )
    assert "joint_flaky" in result.failed_joint_metrics
    assert "constant dimension" in result.failed_joint_metrics["joint_flaky"]
    # The marginal metrics are unaffected: the guard's whole purpose is to
    # stop one joint metric from costing the trial its other results.
    assert "nrmse" in result.summary
    assert np.isfinite(result.summary["nrmse"])


def test_one_failing_joint_metric_does_not_invalidate_another(joint_metric):
    joint_metric("joint_ok", lambda inputs: {"joint_ok": 1.0})

    def boom(inputs: JointMetricInputs) -> dict[str, float]:
        raise ValueError("no")

    joint_metric("joint_bad", boom)
    result = _run(["theta"], ["joint_ok", "joint_bad"])

    assert result.summary["joint_ok"] == pytest.approx(1.0)
    assert "joint_bad" not in result.summary
    assert set(result.failed_joint_metrics) == {"joint_bad"}


def test_a_failed_joint_metric_is_not_recomputed(joint_metric):
    """Once invalidated, paying for it again buys a value that is discarded."""
    calls: list[int] = []

    def boom(inputs: JointMetricInputs) -> dict[str, float]:
        calls.append(inputs.cond_id)
        raise ValueError("no")

    joint_metric("joint_expensive", boom)
    _run(["theta"], ["joint_expensive"], n_conditions=4)
    assert calls == [0]


def test_a_multi_output_joint_metric_drops_all_its_keys_on_failure(
    joint_metric,
):
    """The guard records metric NAMES; the summary is keyed by OUTPUTS."""

    def two_keys(inputs: JointMetricInputs) -> dict[str, float]:
        if inputs.cond_id == 1:
            raise ValueError("no")
        return {"joint_left": 1.0, "joint_right": 2.0}

    joint_metric(
        "joint_pair",
        two_keys,
        kind="diagnostic",
        outputs=("joint_left", "joint_right"),
    )
    result = _run(["theta"], ["joint_pair"])

    assert "joint_left" not in result.summary
    assert "joint_right" not in result.summary


# ---------------------------------------------------------------------------
# D4 -- the routing surface
# ---------------------------------------------------------------------------


def test_joint_names_route_to_the_joint_resolver_only(joint_metric):
    joint_metric("joint_routed", lambda inputs: {"joint_routed": 0.0})

    names = ["nrmse", "joint_routed"]
    assert set(resolve_metrics(names)) == {"nrmse"}
    assert set(resolve_joint_metrics(names)) == {"joint_routed"}
    assert is_joint_metric("joint_routed")
    assert not is_joint_metric("nrmse")


def test_an_unknown_name_still_raises_in_both_resolvers():
    """A typo must not be quietly reclassified as the other kind."""
    with pytest.raises(KeyError):
        resolve_metrics(["definitely_not_a_metric"])
    with pytest.raises(KeyError):
        resolve_joint_metrics(["definitely_not_a_metric"])


def test_a_joint_name_is_visible_to_the_shared_lookup_tables(joint_metric):
    """The reason the marker lives on one registry rather than beside it.

    ``producer_for_key`` drives ``_metric_names_for_pipeline``, which DROPS
    names it returns None for -- so a joint objective held in a separate
    registry would request nothing, compute nothing, and take the penalty on
    every trial, with no error raised anywhere.
    """
    from bayesflow_hpo.validation.registry import (
        list_metrics,
        producer_for_key,
        validate_objective_metric_kinds,
    )

    joint_metric(
        "joint_diag",
        lambda inputs: {"joint_diag": 0.0},
        kind="diagnostic",
    )
    assert producer_for_key("joint_diag") == "joint_diag"
    assert "joint_diag" in list_metrics()
    with pytest.raises(ValueError, match="Diagnostic metric"):
        validate_objective_metric_kinds(["joint_diag"])


def test_overwriting_a_joint_name_with_a_marginal_one_clears_the_marker(
    joint_metric,
):
    """A stale marker routes a 2-argument callable to the joint dispatch."""
    joint_metric("joint_then_marginal", lambda inputs: {})
    assert is_joint_metric("joint_then_marginal")
    try:
        register_metric(
            "joint_then_marginal",
            lambda draws, true_values: {"joint_then_marginal": 0.0},
            overwrite=True,
        )
        assert not is_joint_metric("joint_then_marginal")
    finally:
        unregister_metric("joint_then_marginal")


# ---------------------------------------------------------------------------
# The built-in `lc2st` joint metric
#
# These live here rather than beside the other L-C2ST tests because
# `test_c2st.py` opens with `pytest.importorskip("sklearn")`. They must run
# WITHOUT scikit-learn -- one of them asserts precisely that importing the
# package does not need it, which is unfalsifiable in a module that skips
# itself when it is missing.
# ---------------------------------------------------------------------------


def test_lc2st_is_registered_as_a_joint_metric():
    """So that ``objective_metrics=["lc2st"]`` resolves at all.

    The name has to be visible to ``producer_for_key``, which drives
    ``_metric_names_for_pipeline``; a name it returns None for is DROPPED
    from the pipeline's metric list, so the objective would request
    nothing, compute nothing, and take the penalty on every trial.
    """
    from bayesflow_hpo.validation.registry import producer_for_key

    assert is_joint_metric("lc2st")
    assert producer_for_key("lc2st") == "lc2st"


def test_importing_the_package_does_not_require_sklearn():
    """The registered default must build its classifier lazily.

    Calling `make_lc2st_joint_metric()` at module scope would raise
    ImportError during ``import bayesflow_hpo`` wherever scikit-learn is
    absent, making an optional dependency mandatory. The guard has to fire
    when the metric RUNS, not when it is registered.
    """
    from unittest.mock import patch

    from bayesflow_hpo.validation.c2st import _default_lc2st_metric
    from bayesflow_hpo.validation.registry import get_metric

    assert get_metric("lc2st") is _default_lc2st_metric
    with patch(
        "bayesflow_hpo.validation.c2st._require_sklearn",
        side_effect=ImportError("no sklearn"),
    ):
        with pytest.raises(ImportError):
            _default_lc2st_metric(None)


# ---------------------------------------------------------------------------
# D9 -- the condition sub-sample
# ---------------------------------------------------------------------------


def test_the_contract_reports_the_grid_size(joint_metric):
    """Nothing else in JointMetricInputs implies it."""
    seen: list[tuple[int, int]] = []

    def capture(inputs: JointMetricInputs) -> dict[str, float]:
        seen.append((inputs.cond_id, inputs.n_conditions))
        return {"joint_size": 0.0}

    joint_metric("joint_size", capture)
    _run(["theta"], ["joint_size"], n_conditions=5)
    assert seen == [(i, 5) for i in range(5)]


def test_a_subsampled_metric_is_averaged_over_the_conditions_it_ran_on(
    joint_metric,
):
    """Skipping a condition returns no key, which contributes no row.

    A sentinel value would be averaged in and would drag the score toward
    whatever the sentinel is; an empty dict simply does not participate.
    """

    def every_other(inputs: JointMetricInputs) -> dict[str, float]:
        if inputs.cond_id % 2:
            return {}
        return {"joint_sparse": float(inputs.cond_id)}

    joint_metric("joint_sparse", every_other)
    result = _run(["theta"], ["joint_sparse"], n_conditions=5)
    # Ran on 0, 2, 4 -> mean 2.0. Had the skipped conditions contributed a
    # zero, the mean would be 1.2.
    assert result.summary["joint_sparse"] == pytest.approx(2.0)


def test_the_subsample_is_spread_over_the_grid_not_taken_from_its_front():
    """A validation grid is ordered, so a prefix is one corner of it."""
    from bayesflow_hpo.validation.c2st import _subsampled_conditions

    assert _subsampled_conditions(20, 4) == {0, 6, 13, 19}
    assert _subsampled_conditions(5, 10) == {0, 1, 2, 3, 4}
    # One condition takes the MIDDLE. `np.linspace(0, n-1, 1)` is [0], which
    # is the ordered-grid prefix this function exists to avoid -- the single
    # worst choice available, arrived at by the formula rather than chosen.
    assert _subsampled_conditions(5, 1) == {2}
    assert _subsampled_conditions(20, 1) == {10}


def test_the_subsample_is_identical_across_trials():
    """A metric scored on different conditions per trial is not comparable."""
    from bayesflow_hpo.validation.c2st import _subsampled_conditions

    assert all(
        _subsampled_conditions(17, 5) == _subsampled_conditions(17, 5)
        for _ in range(5)
    )


def test_a_metric_emitting_an_undeclared_key_still_drops_it_on_failure(
    joint_metric,
):
    """A metric's emitted keys need not match what it declared.

    Whole-trial invalidation drops `output_keys_for(name)`, which is the
    metric's own name plus whatever `outputs=` declared. A key it emits
    beyond that -- an undeclared extra, or an override callable that emits
    something else entirely -- would survive, and a flattering partial mean
    would reach the objective, defeating the invalidation. So the pipeline
    also records what each metric was SEEN to emit.
    """

    def flaky(inputs: JointMetricInputs) -> dict[str, float]:
        if inputs.cond_id == 1:
            raise RuntimeError("boom")
        return {"an_undeclared_key": 0.01}

    joint_metric("joint_undeclared", flaky, kind="diagnostic")
    result = _run(["theta"], ["nrmse", "joint_undeclared"])

    assert "joint_undeclared" in result.failed_joint_metrics
    assert "an_undeclared_key" not in result.summary, (
        "a partially computed joint metric reached the summary, so its "
        "registered worst case was never applied"
    )
    assert "nrmse" in result.summary


def test_an_override_must_name_a_registered_joint_metric():
    """A key that names nothing runs a metric that appeared from nowhere.

    `canonical_metric_name` passes unknown names through by design, so an
    unchecked merge would accept a typo and report its value in the summary
    under a name no configuration mentions.
    """
    from bayesflow_hpo.validation.registry import (
        JointMetricConfigurationError,
    )

    with pytest.raises(
        JointMetricConfigurationError, match="not registered joint metrics"
    ):
        run_validation_pipeline(
            approximator=_FakeApproximator(["theta"], 8),
            validation_data=_dataset(["theta"], 2, 8),
            n_posterior_samples=16,
            metrics=["nrmse"],
            joint_metrics={"never_registered": lambda inputs: {"x": 0.0}},
        )


def test_an_override_must_not_name_a_marginal_metric():
    """Otherwise `nrmse` computes marginally AND dispatches jointly."""
    from bayesflow_hpo.validation.registry import (
        JointMetricConfigurationError,
    )

    with pytest.raises(JointMetricConfigurationError, match="nrmse"):
        run_validation_pipeline(
            approximator=_FakeApproximator(["theta"], 8),
            validation_data=_dataset(["theta"], 2, 8),
            n_posterior_samples=16,
            metrics=["nrmse"],
            joint_metrics={"nrmse": lambda inputs: {"nrmse": 0.0}},
        )


def test_an_aliased_override_suppresses_its_registry_entry():
    """Both sides of the skip check must be canonicalized, not just one.

    The list name was canonicalized and the override key was not, so an
    aliased override left the registry entry resolved under the canonical
    name AND dispatched the override under the alias -- the metric ran
    twice. Only `ObjectiveConfig.__post_init__` canonicalizes keys before
    this, so `optimize()` was covered while `check_pipeline`,
    `validate_once` and a direct pipeline call were not.
    """
    from bayesflow_hpo.validation.registry import (
        register_joint_metric,
        resolve_joint_metrics,
    )

    register_joint_metric(
        "aliased_joint",
        lambda inputs: {"aliased_joint": 0.0},
        aliases=["aj"],
        kind="diagnostic",
        overwrite=True,
    )
    try:
        assert resolve_joint_metrics(["aliased_joint"], overridden={"aj"}) == {}
        assert resolve_joint_metrics(["aj"], overridden={"aliased_joint"}) == {}
        # Without an override it still resolves, under either spelling.
        assert set(resolve_joint_metrics(["aj"])) == {"aj"}
    finally:
        # Removes the alias too, which is why teardown is one call.
        unregister_metric("aliased_joint")

