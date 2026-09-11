"""Does a real search actually select the better trial?

These run the full lifecycle -- real build, real training, real Optuna study --
but hand the *metric values* to the objective through a custom ``validate_fn``.
That makes selection assertable without asserting anything about a model
trained for two epochs.

Two traps this file is built around, both of which sink the obvious
implementation:

**Pre-flight consumes a response.**  ``optimize()`` calls ``check_pipeline()``
before the study runs (``api.py:460``), and pre-flight trains its own
approximator and calls the *same* ``validate_fn`` (``pipeline.py:331``),
raising ``PipelineError`` on missing or non-finite required metrics
(``pipeline.py:365``, ``:372``).  A hook with one response per trial therefore
has one consumer too many.  The hook below makes the contract explicit and
fails loudly on an unexpected call rather than returning a default.  Mocking
pre-flight away is not an option -- it is part of the seam under test.

**The default cost metric defeats the assertion.**  With
``cost_metric="inference_time"`` (the default, ``api.py:130``) the cost of a
trial using a custom validator is that hook's own wall-clock time
(``objective.py:1297``).  A trial that omits a metric can return faster than
one that reports it, making both trials non-dominated -- so a Pareto-membership
assertion passes even with the metric direction inverted.  Every test here uses
``cost_metric="param_count"`` with a fully pinned architecture, so both trials
carry an identical cost coordinate and selection is decided by the metric
alone.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from .conftest import assert_trials_succeeded

pytestmark = pytest.mark.endtoend


class ScriptedValidator:
    """A ``validate_fn`` returning a fixed script of responses.

    Call 0 is always ``check_pipeline()``'s pre-flight validation and gets
    *preflight*, which must be valid and finite or the run is rejected before
    the study starts. Calls 1..n are trials, in order -- trials run
    sequentially, so call order is trial order.

    An unscripted call raises rather than returning a default, so a change in
    how often the pipeline validates surfaces as a test failure instead of
    silently shifting which response a trial receives.
    """

    def __init__(
        self,
        preflight: dict[str, float],
        responses: list[dict[str, float]],
    ) -> None:
        self.preflight = preflight
        self.responses = responses
        self.n_calls = 0

    def __call__(
        self,
        approximator: Any,
        validation_data: Any,
        n_posterior_samples: int,
    ) -> dict[str, float]:
        index = self.n_calls
        self.n_calls += 1
        if index == 0:
            return dict(self.preflight)
        trial_index = index - 1
        if trial_index >= len(self.responses):
            raise AssertionError(
                f"validate_fn called {self.n_calls} times; the script covers "
                f"1 pre-flight + {len(self.responses)} trials"
            )
        return dict(self.responses[trial_index])

    @property
    def trial_calls(self) -> int:
        return max(0, self.n_calls - 1)


def test_scripted_validator_covers_preflight_and_each_trial(run_study):
    """The hook contract itself: one pre-flight call plus one per trial.

    Asserted separately so that if the pipeline ever starts validating at a
    different cadence, the failure names that fact rather than showing up as
    a mysterious selection error.
    """
    validator = ScriptedValidator(
        preflight={"log_gamma": 0.0},
        responses=[{"log_gamma": 1.0}, {"log_gamma": 2.0}],
    )

    study = run_study(
        objective_metrics=["log_gamma"],
        cost_metric="param_count",
        validate_fn=validator,
    )

    assert_trials_succeeded(study, expected=2)
    assert validator.n_calls == 3, (
        f"expected 1 pre-flight + 2 trial calls, got {validator.n_calls}"
    )
    assert validator.trial_calls == 2


def test_omitted_metric_ranks_strictly_worse_than_a_bad_reported_value(run_study):
    """Failing to report must be strictly worse than reporting something bad.

    This is the inversion #72 fixed, asserted through the public entry point.
    A trial that omits the metric gets the sanitization penalty; a trial that
    reports a genuinely bad value gets a real objective. If the penalty is not
    worse than every finite value, a search learns to fail rather than to
    improve.

    Strict inequality matters: "does not outrank" would pass for a broken
    implementation that gave every trial the same penalty.

    Must fail if: the missing-metric penalty stops being worse than reported
    values; the ``log_gamma`` conversion is reversed.
    """
    validator = ScriptedValidator(
        preflight={"log_gamma": 0.0},
        responses=[
            {"log_gamma": -5.0},  # bad, but finite and reported
            {},  # omits the metric entirely
        ],
    )

    study = run_study(
        objective_metrics=["log_gamma"],
        cost_metric="param_count",
        validate_fn=validator,
    )

    reported, omitted = study.trials[0], study.trials[1]
    assert reported.values is not None and omitted.values is not None

    assert math.isfinite(reported.values[0]), (
        f"the reporting trial should have a finite objective, "
        f"got {reported.values[0]}"
    )
    assert omitted.values[0] > reported.values[0], (
        f"omitting the metric ({omitted.values[0]}) must rank strictly worse "
        f"than reporting a bad value ({reported.values[0]})"
    )
    assert omitted not in study.best_trials, (
        "a trial that never reported the metric is on the Pareto front"
    )


def test_the_better_trial_is_exactly_the_pareto_front(run_study):
    """With cost held equal, the better metric wins outright.

    Both trials build the identical pinned architecture, so
    ``cost_metric="param_count"`` gives them the same cost coordinate and the
    front is decided by ``log_gamma`` alone. Asserting *exact* membership --
    rather than "the good trial is somewhere in ``best_trials``" -- is what
    makes this catch a direction inversion.

    Must fail if: the ``log_gamma`` direction is inverted anywhere between the
    metric and Optuna's ranking.
    """
    validator = ScriptedValidator(
        preflight={"log_gamma": 0.0},
        responses=[
            {"log_gamma": 2.0},  # better: log_gamma is higher-is-better
            {"log_gamma": -2.0},  # worse
        ],
    )

    study = run_study(
        objective_metrics=["log_gamma"],
        cost_metric="param_count",
        validate_fn=validator,
    )

    assert_trials_succeeded(study, expected=2)
    better, worse = study.trials[0], study.trials[1]

    # Cost is genuinely equal, so the front is not decided by a tiebreak.
    assert better.values[1] == pytest.approx(worse.values[1]), (
        f"cost coordinates differ ({better.values[1]} vs {worse.values[1]}); "
        "the architectures were meant to be identical"
    )

    front = {t.number for t in study.best_trials}
    assert front == {better.number}, (
        f"expected exactly trial {better.number} on the front, got {front}"
    )


def test_custom_validate_fn_reaches_the_study_without_the_default_pipeline(
    run_study,
):
    """The hook replaces the built-in validator, and its values are stored.

    The ``validate_fn`` hook is public API and had no end-to-end coverage:
    every previous test of it stubbed the surrounding pipeline.

    Must fail if: the hook's return value stops reaching the objective, or
    stops being recorded as user attrs for post-hoc analysis.
    """
    validator = ScriptedValidator(
        preflight={"log_gamma": 0.0},
        responses=[{"log_gamma": 1.25}],
    )

    study = run_study(
        n_trials=1,
        objective_metrics=["log_gamma"],
        cost_metric="param_count",
        validate_fn=validator,
    )

    trial = study.trials[0]
    assert trial.user_attrs["log_gamma"] == pytest.approx(1.25)
    assert trial.values[0] == pytest.approx(-1.25)

    # The built-in metrics are absent: the default pipeline did not run.
    assert "calibration_error" not in trial.user_attrs, (
        "the default validation pipeline ran despite a custom validate_fn"
    )


def test_non_finite_hook_response_is_refused_in_preflight(run_study):
    """A hook that cannot produce a finite metric fails before training.

    Pre-flight exists to make interface errors cheap. Returning NaN from the
    very first call should stop the run, not start 50 trials that all fall
    back to a penalty.

    Must fail if: pre-flight stops checking finiteness of required metrics
    (``pipeline.py:372``).
    """
    validator = ScriptedValidator(
        preflight={"log_gamma": float("nan")},
        responses=[{"log_gamma": 1.0}],
    )

    with pytest.raises(Exception) as excinfo:
        run_study(
            n_trials=1,
            objective_metrics=["log_gamma"],
            cost_metric="param_count",
            validate_fn=validator,
        )

    assert validator.n_calls == 1, (
        "the run continued past a failed pre-flight validation"
    )
    assert "log_gamma" in str(excinfo.value)
