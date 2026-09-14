"""The joint, data-dependent metric path.

Covers the three failure modes the design identifies, each of which produces
a metric that *looks* configured and silently never reports:

1. the 3-D shape invariant, which two separate squeezes break;
2. a joint key vanishing from the multi-parameter summary;
3. a per-condition failure being averaged away instead of penalized.

Design: ``docs/plans/plan-joint-metric-path.md``, decisions D2, D3, D4, D8.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn
from bayesflow_hpo.validation.pipeline import run_validation_pipeline
from bayesflow_hpo.validation.registry import (
    _JOINT,
    _REGISTRY,
    JointMetricInputs,
    is_joint_metric,
    register_joint_metric,
    register_metric,
    resolve_joint_metrics,
    resolve_metrics,
)


@pytest.fixture
def joint_metric():
    """Register joint metrics for one test and remove them afterwards."""
    registered: list[str] = []

    def _register(name: str, fn, **kwargs) -> str:
        register_joint_metric(name, fn, **kwargs)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        _REGISTRY.pop(name, None)
        _JOINT.discard(name)


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

    def sample(self, *, conditions, num_samples):
        rng = np.random.default_rng(0)
        return {
            key: rng.normal(size=(self.n_sims, num_samples, 1))
            for key in self.param_keys
        }


def _dataset(param_keys: list[str], n_conditions: int, n_sims: int):
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


def _run(param_keys, metric_names, n_conditions=3, n_sims=8, n_samples=16):
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

    def record_marginal(draws, true_values):
        marginal_ndim.append(draws.ndim)
        return {"marginal_probe": 0.0}

    register_metric("marginal_probe", record_marginal, overwrite=True)
    try:
        _run(["theta"], ["marginal_probe"])
        assert marginal_ndim and set(marginal_ndim) == {2}
    finally:
        _REGISTRY.pop("marginal_probe", None)


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
        _REGISTRY.pop("joint_then_marginal", None)
