"""Joint metrics keep their per-condition values (issue #111).

Marginal rows reach ``ValidationResult.condition_metrics``; joint ones used
to be reduced and discarded, so a condition where a joint metric is blind
was indistinguishable from one where it works. See
``ValidationResult.joint_condition_metrics``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np
import pytest

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.pipeline import run_validation_pipeline
from bayesflow_hpo.validation.registry import (
    JointMetricFn,
    JointMetricInputs,
    register_joint_metric,
    unregister_metric,
)
from bayesflow_hpo.validation.result import ValidationResult


class _FakeApproximator:
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


def _dataset(param_keys: list[str], n_conditions: int) -> ValidationDataset:
    rng = np.random.default_rng(1)
    sims = [
        {
            **{pk: rng.normal(size=(6,)) for pk in param_keys},
            "x": rng.normal(size=(6, 3)),
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
    *,
    joint_metrics: dict[str, Any] | None = None,
    n_conditions: int = 3,
    **kwargs: Any,
) -> ValidationResult:
    return run_validation_pipeline(
        approximator=_FakeApproximator(param_keys, 6),
        validation_data=_dataset(param_keys, n_conditions),
        n_posterior_samples=8,
        metrics=["mae"],
        joint_metrics=joint_metrics,
        **kwargs,
    )


class RegisterJoint(Protocol):
    """What the ``registered`` fixture hands a test.

    A plain ``Callable[..., str]`` would not carry the keyword arguments
    forwarded to ``register_joint_metric`` (``outputs=``, and so on), so a
    test passing one would not type-check.
    """

    def __call__(
        self, name: str, fn: JointMetricFn, **kwargs: Any
    ) -> str: ...


@pytest.fixture
def registered() -> Iterator[RegisterJoint]:
    names: list[str] = []

    def _register(name: str, fn: JointMetricFn, **kwargs: Any) -> str:
        register_joint_metric(name, fn, overwrite=True, **kwargs)
        names.append(name)
        return name

    yield _register

    for name in names:
        unregister_metric(name)


def _probe(inputs: JointMetricInputs) -> dict[str, float]:
    """A value that differs per condition, so a mean would hide it."""
    return {"joint_probe": float(inputs.cond_id) / 10.0}


@pytest.mark.parametrize("param_keys", [["theta"], ["a", "b"]])
def test_joint_per_condition_values_survive(
    registered: RegisterJoint, param_keys: list[str],
) -> None:
    registered("joint_probe", _probe)
    result = _run(param_keys, joint_metrics={"joint_probe": _probe})

    frame = result.joint_condition_metrics
    assert list(frame["id_cond"]) == [0, 1, 2]
    assert list(frame["joint_probe"]) == pytest.approx([0.0, 0.1, 0.2])
    # The summary still carries the reduction, unchanged.
    assert result.summary["joint_probe"] == pytest.approx(0.1)


def test_reduction_is_recoverable_from_the_rows(registered: RegisterJoint) -> None:
    """``worst`` over the exposed rows is the value the summary reports."""
    registered("joint_probe", _probe)
    result = _run(
        ["a", "b"],
        joint_metrics={"joint_probe": _probe},
        aggregate={"joint_probe": "worst"},
    )
    rows = result.joint_condition_metrics["joint_probe"]
    assert result.summary["joint_probe"] == pytest.approx(float(rows.max()))
    assert result.summary["joint_probe"] == pytest.approx(0.2)


def test_frame_is_empty_without_joint_metrics() -> None:
    result = _run(["a", "b"])
    assert result.joint_condition_metrics.empty
    assert result.joint_condition_table().empty


def test_frame_is_empty_when_every_joint_metric_is_invalidated(
    registered: RegisterJoint,
) -> None:
    """All-invalidated is absence, not a frame of bare condition ids.

    ``_run_joint_metrics`` returns an empty row per condition once the only
    metric is in ``failed_joint``, so the rows list is non-empty while
    carrying no value. A frame of bare ``id_cond`` would report ``.empty``
    as False and send the natural guard into a ``KeyError``.
    """

    def always_fails(inputs: JointMetricInputs) -> dict[str, float]:
        raise RuntimeError("boom")

    registered("joint_doomed", always_fails)
    result = _run(["a", "b"], joint_metrics={"joint_doomed": always_fails})

    assert result.joint_condition_metrics.empty
    assert list(result.joint_condition_metrics.columns) == []
    assert "joint_doomed" in result.failed_joint_metrics
    assert "joint_doomed" not in result.summary


def test_invalidated_metric_leaves_no_column(registered: RegisterJoint) -> None:
    """A metric dropped from the summary is dropped from the rows too.

    Keeping the conditions it survived would let a caller reduce them by
    hand and recover exactly the flattering partial value that whole-trial
    invalidation exists to prevent.
    """

    def flaky(inputs: JointMetricInputs) -> dict[str, float]:
        if inputs.cond_id == 2:
            raise RuntimeError("bad condition")
        return {"joint_flaky": 0.5}

    registered("joint_probe", _probe)
    registered("joint_flaky", flaky)
    result = _run(
        ["a", "b"],
        joint_metrics={"joint_probe": _probe, "joint_flaky": flaky},
    )

    frame = result.joint_condition_metrics
    assert "joint_flaky" not in frame.columns
    assert "joint_flaky" not in result.summary
    assert "joint_flaky" in result.failed_joint_metrics
    # The metric beside it is untouched.
    assert list(frame["joint_probe"]) == pytest.approx([0.0, 0.1, 0.2])


def test_declared_outputs_are_dropped_when_nothing_was_emitted(
    registered: RegisterJoint,
) -> None:
    """The declared half of the drop-set union, isolated.

    Every other metric here emits a key equal to its own name, so
    ``emitted_keys`` alone would pass. A metric that fails on the FIRST
    condition emits nothing, so only ``output_keys_for`` knows its columns
    -- which is what ``_dropped_joint_keys`` consults both sources for.
    """

    def fails_first(inputs: JointMetricInputs) -> dict[str, float]:
        if inputs.cond_id == 0:
            raise RuntimeError("boom")
        return {"declared_a": 1.0, "declared_b": 2.0}

    registered("joint_probe", _probe)
    registered(
        "joint_declared", fails_first, outputs=("declared_a", "declared_b")
    )
    result = _run(
        ["a", "b"],
        joint_metrics={"joint_probe": _probe, "joint_declared": fails_first},
    )

    frame = result.joint_condition_metrics
    assert "declared_a" not in frame.columns
    assert "declared_b" not in frame.columns
    # Frame and summary agree on exactly which metrics survived.
    assert set(frame.columns) - {"id_cond"} == set(result.summary) - {
        "mae", "n_sims",
    }


def test_id_cond_is_the_pipelines_not_the_metrics(registered: RegisterJoint) -> None:
    """A metric emitting ``id_cond`` must not overwrite the condition index.

    The index is what makes every other column attributable to a condition,
    so a collision drops the metric's key rather than the pipeline's value.
    """

    def collide(inputs: JointMetricInputs) -> dict[str, float]:
        return {"id_cond": 99.0, "joint_probe": float(inputs.cond_id) / 10.0}

    registered("joint_collide", collide, outputs=("joint_probe",))
    result = _run(["a", "b"], joint_metrics={"joint_collide": collide})

    frame = result.joint_condition_metrics
    assert list(frame["id_cond"]) == [0, 1, 2]
    assert list(frame["joint_probe"]) == pytest.approx([0.0, 0.1, 0.2])


def test_worst_recovers_the_min_for_a_higher_is_better_joint_metric(
    registered: RegisterJoint,
) -> None:
    """``worst`` is direction-aware jointly, not a blanket maximum.

    The companion test above recovers ``worst`` as ``rows.max()``, which
    holds only for a lower-is-better metric. Registering a direction has to
    flip that, or a higher-is-better metric reports its BEST condition.
    """
    from bayesflow_hpo.objectives import (
        HIGHER_IS_BETTER,
        METRIC_DIRECTIONS,
        register_metric_direction,
    )

    name = "joint_higher"

    def higher(inputs: JointMetricInputs) -> dict[str, float]:
        return {name: float(inputs.cond_id) / 10.0}

    registered(name, higher)
    # Through the public API, not by writing METRIC_DIRECTIONS:
    # `_direction_for` honours a removal from the legacy HIGHER_IS_BETTER
    # set, so a direct write leaves `worst_reducer` on its np.max fallback
    # and the metric would report its BEST condition.
    register_metric_direction(name, higher_is_better=True, worst_raw=0.0)
    try:
        result = _run(
            ["a", "b"],
            joint_metrics={name: higher},
            aggregate={name: "worst"},
        )
        rows = result.joint_condition_metrics[name]
        assert result.summary[name] == pytest.approx(float(rows.min()))
        assert result.summary[name] == pytest.approx(0.0)
    finally:
        METRIC_DIRECTIONS.pop(name, None)
        HIGHER_IS_BETTER.discard(name)


def test_joint_condition_table_filters_by_metric(registered: RegisterJoint) -> None:
    def two(inputs: JointMetricInputs) -> dict[str, float]:
        return {"joint_probe": 1.0, "other_key": 2.0}

    registered("joint_two", two, outputs=("joint_probe", "other_key"))
    result = _run(["a", "b"], joint_metrics={"joint_two": two})

    cols = list(result.joint_condition_table(metric="joint_probe").columns)
    assert cols == ["id_cond", "joint_probe"]
