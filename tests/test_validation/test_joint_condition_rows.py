"""Joint metrics keep their per-condition values (issue #111).

Marginal rows reach ``ValidationResult.condition_metrics``; joint ones used
to be reduced and discarded, so a condition where a joint metric is blind
was indistinguishable from one where it works. See
``ValidationResult.joint_condition_metrics``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.pipeline import run_validation_pipeline
from bayesflow_hpo.validation.registry import (
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


@pytest.fixture
def registered() -> Iterator[Any]:
    names: list[str] = []

    def _register(name: str, fn: Any, **kwargs: Any) -> str:
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
def test_joint_per_condition_values_survive(registered, param_keys) -> None:
    registered("joint_probe", _probe)
    result = _run(param_keys, joint_metrics={"joint_probe": _probe})

    frame = result.joint_condition_metrics
    assert list(frame["id_cond"]) == [0, 1, 2]
    assert list(frame["joint_probe"]) == pytest.approx([0.0, 0.1, 0.2])
    # The summary still carries the reduction, unchanged.
    assert result.summary["joint_probe"] == pytest.approx(0.1)


def test_reduction_is_recoverable_from_the_rows(registered) -> None:
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
    registered,
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


def test_invalidated_metric_leaves_no_column(registered) -> None:
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


def test_joint_condition_table_filters_by_metric(registered) -> None:
    def two(inputs: JointMetricInputs) -> dict[str, float]:
        return {"joint_probe": 1.0, "other_key": 2.0}

    registered("joint_two", two, outputs=("joint_probe", "other_key"))
    result = _run(["a", "b"], joint_metrics={"joint_two": two})

    cols = list(result.joint_condition_table(metric="joint_probe").columns)
    assert cols == ["id_cond", "joint_probe"]
