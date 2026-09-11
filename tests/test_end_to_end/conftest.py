"""Fixtures for the end-to-end ``optimize()`` suite.

This directory is the one place in ``tests/`` that builds **real** BayesFlow
approximators and runs the **real** validation pipeline.  Everything else
mocks the pipeline out (``tests/test_api.py``) or hand-feeds synthetic metric
values (``tests/test_optimization/test_direction_end_to_end.py``), which is
why the integration seam between ``optimize()``, ``check_pipeline()``, the
builders and ``run_validation_pipeline()`` had no coverage at all.

See ``docs/plans/plan-end-to-end-optimize-tests.md`` for the reasoning behind
each fixture; the short version of the traps it avoids:

- Search-space dimensions **must** be ``Dimension`` objects.  Dimension
  discovery filters on ``isinstance(..., _DIMENSION_TYPES)``
  (``search_spaces/base.py:276``), so a bare ``int`` is silently dropped and
  the builder loses the key.
- Every study needs its **own** ``CheckpointPool``.  The default one is rooted
  at the relative path ``checkpoints`` (``checkpoint_pool.py:41``) and trial
  numbering restarts per study, so studies otherwise overwrite each other's
  files -- and save failures are swallowed (``:76``), so the suite would stay
  green while colliding.
- Seeding must **not** be done by passing a sampler instance: ``optimize()``
  silently skips soft constraints when given one (``api.py:478``).
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

# Keep the plotting backend headless: this directory does not inherit the
# `Agg` selection that `tests/test_visualization.py:14` performs at import.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import bayesflow as bf  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pytest  # noqa: E402

from bayesflow_hpo.api import optimize  # noqa: E402
from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool  # noqa: E402
from bayesflow_hpo.optimization.objective import default_train_fn  # noqa: E402
from bayesflow_hpo.search_spaces.base import (  # noqa: E402
    FloatDimension,
    IntDimension,
)
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace  # noqa: E402
from bayesflow_hpo.search_spaces.inference.flow_matching import (  # noqa: E402
    FlowMatchingSpace,
)
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace  # noqa: E402
from bayesflow_hpo.search_spaces.training import TrainingSpace  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

#: User attributes that mark a trial as having taken a failure or fallback
#: path.  A trial carrying any of these reached ``TrialState.COMPLETE``
#: *without* completing the pipeline: training and validation exceptions
#: return penalty tuples rather than propagating (``objective.py:1277``,
#: ``:1362``), so "all trials COMPLETE" on its own asserts almost nothing.
FAILURE_ATTRS = (
    "rejected_reason",
    "compile_error",
    "build_error",
    "param_probe_error",
    "training_error",
    "validation_error",
    "validation_fallback",
)


# ---------------------------------------------------------------------------
# Model under test: small enough to train in ~1s, real enough to exercise the
# builders, the BayesFlow fit loop and the validation pipeline.
# ---------------------------------------------------------------------------


def _prior_fn() -> dict[str, np.ndarray]:
    return {"theta": np.random.normal(0.0, 1.0, size=(1,)).astype("float32")}


def _likelihood_fn(theta: np.ndarray) -> dict[str, np.ndarray]:
    theta_value = float(np.squeeze(theta))
    return {"x": np.random.normal(theta_value, 1.0, size=(4, 1)).astype("float32")}


@pytest.fixture
def tiny_simulator() -> Any:
    """Gaussian location model: scalar ``theta``, four observations."""
    return bf.simulators.make_simulator([_prior_fn, _likelihood_fn])


@pytest.fixture
def tiny_adapter() -> Any:
    """Adapter mapping ``theta``/``x`` onto BayesFlow's canonical keys."""
    return (
        bf.Adapter()
        .as_set(["x"])
        .rename("theta", "inference_variables")
        .concatenate(["x"], into="summary_variables", axis=-1)
    )


def make_tiny_search_space() -> CompositeSearchSpace:
    """Fully pinned search space -- every trial builds the same ~5.3K net.

    Every dimension is spelled out as a ``Dimension`` with ``constant=``,
    including both dropout dimensions (``flow_matching.py:64``,
    ``deep_set.py:61``).  Left unpinned, dropout is sampled and trials stop
    being identical, which the selection tests rely on.

    Pinning also keeps the parameter count far below any budget, so no trial
    is budget-rejected and ``n_trials`` means what it says.
    """
    return CompositeSearchSpace(
        inference_space=FlowMatchingSpace(
            subnet_width=IntDimension("fm_subnet_width", constant=16),
            subnet_depth=IntDimension("fm_subnet_depth", constant=1),
            dropout=FloatDimension("fm_dropout", constant=0.0),
        ),
        summary_space=DeepSetSpace(
            summary_dim=IntDimension("ds_summary_dim", constant=4),
            depth=IntDimension("ds_depth", constant=1),
            width=IntDimension("ds_width", constant=16),
            dropout=FloatDimension("ds_dropout", constant=0.0),
        ),
        training_space=TrainingSpace(
            batch_size=IntDimension("batch_size", constant=32),
            initial_lr=FloatDimension("initial_lr", constant=1e-3),
        ),
    )


@pytest.fixture
def tiny_search_space() -> CompositeSearchSpace:
    return make_tiny_search_space()


# ---------------------------------------------------------------------------
# Training spy
# ---------------------------------------------------------------------------


class TrainingSpy:
    """Wraps the real ``default_train_fn`` and records that it did work.

    A no-op ``train_fn`` leaves an initialized model that still validates and
    still satisfies every structural assertion, so "the study completed" does
    not establish that training ran.  This records the optimizer's iteration
    count after each call, which is zero for a model that never took a step.

    Note the first recorded call is always ``check_pipeline()``'s pre-flight
    training step (``pipeline.py:323``), not trial 0.
    """

    def __init__(self) -> None:
        self.iterations: list[int] = []

    def __call__(
        self,
        approximator: Any,
        simulator: Any,
        hparams: dict[str, Any],
        callbacks: list[Any],
    ) -> None:
        default_train_fn(approximator, simulator, hparams, callbacks)
        optimizer = getattr(approximator, "optimizer", None)
        steps = getattr(optimizer, "iterations", None)
        self.iterations.append(int(steps) if steps is not None else 0)

    @property
    def n_calls(self) -> int:
        return len(self.iterations)


@pytest.fixture
def training_spy() -> TrainingSpy:
    return TrainingSpy()


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------


@pytest.fixture
def run_study(tiny_simulator, tiny_adapter, tmp_path):
    """Run a real ``optimize()`` with fast, isolated defaults.

    Each call gets its own ``CheckpointPool`` under ``tmp_path``; passing
    ``checkpoint_pool`` explicitly overrides that.  Defaults are the smallest
    configuration that still exercises the whole pipeline (~15s per study).
    """
    counter = {"n": 0}

    def _run(**overrides: Any) -> optuna.Study:
        counter["n"] += 1
        np.random.seed(20260911)

        kwargs: dict[str, Any] = dict(
            simulator=tiny_simulator,
            adapter=tiny_adapter,
            search_space=make_tiny_search_space(),
            n_trials=2,
            epochs=2,
            num_batches=4,
            sims_per_condition=32,
            n_posterior_samples=50,
            storage=None,
            show_progress_bar=False,
            checkpoint_pool=CheckpointPool(
                pool_dir=tmp_path / f"checkpoints_{counter['n']}"
            ),
        )
        kwargs.update(overrides)
        return optimize(**kwargs)

    return _run


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_trials_succeeded(study: optuna.Study, expected: int) -> None:
    """Assert every trial completed the pipeline for real.

    ``TrialState.COMPLETE`` alone is not evidence: training and validation
    exceptions are caught and converted to penalty tuples
    (``objective.py:1277``, ``:1362``).  This checks the absence of every
    failure marker and the finiteness of every objective.
    """
    assert len(study.trials) == expected, (
        f"expected {expected} trials, got {len(study.trials)}"
    )
    for trial in study.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE, (
            f"trial {trial.number} is {trial.state}"
        )
        taken = {k: trial.user_attrs[k] for k in FAILURE_ATTRS if k in trial.user_attrs}
        assert not taken, f"trial {trial.number} took a failure path: {taken}"
        assert trial.values is not None
        for i, value in enumerate(trial.values):
            assert np.isfinite(value), (
                f"trial {trial.number} objective {i} is not finite: {value}"
            )


def assert_all_minimize(study: optuna.Study, expected: int) -> None:
    """Assert the study has *expected* directions and all are ``MINIMIZE``.

    Counting directions alone would not notice a flipped one.
    """
    assert len(study.directions) == expected, (
        f"expected {expected} directions, got {len(study.directions)}"
    )
    assert all(d == optuna.study.StudyDirection.MINIMIZE for d in study.directions), (
        f"expected all MINIMIZE, got {study.directions}"
    )


#: Tolerance for comparing an objective value against a stored user attr.
#: User attrs are rounded to six decimals (``objective.py:1347``) while
#: objective extraction reads the unrounded summary (``:1357``), so exact
#: equality is wrong.
ATTR_ROUNDING_TOL = 1e-5
