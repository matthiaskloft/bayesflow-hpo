"""End-to-end: does the search actually select the better model?

Every direction defect found on this branch was the same underlying failure --
a value pointing the wrong way somewhere between the metric and Optuna's
ranking -- and each was caught one path at a time: the extractor, the
pre-conversion penalty injection, the failure fallback, the training-loss
proxy, an explicit ``directions`` override, and metric aliases. Six paths, with
no reason to believe the enumeration was complete.

These tests close the class rather than the paths. They run a real Optuna
study over a fake objective in which one configuration is unambiguously
better calibrated than another, and assert the study selects it. Any remaining
inversion anywhere in that chain fails them, whether or not anyone thought of
the path.

They deliberately do **not** assert intermediate values. A test that checks
``_metric_to_minimize`` returns ``-1.5`` passes even when a later stage flips
the sign back; only the selected trial answers the question the user actually
has.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import optuna
import pytest

from bayesflow_hpo.objectives import (
    FAILED_TRIAL_COST,
    extract_multi_objective_values,
    worst_objective_value,
)
from bayesflow_hpo.optimization.objective import (
    _training_loss_fallback,
    _validate_metric_keys,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Raw metric values a trial might report. `log_gamma` is the interesting one:
# BayesFlow defines it so that log_gamma < 0 rejects rank uniformity, i.e.
# LARGER is better -- the opposite of every other metric here.
_WELL_CALIBRATED = {"log_gamma": 1.8, "nrmse": 0.20}
_MISCALIBRATED = {"log_gamma": -25.5, "nrmse": 0.20}
_WORSE_RECOVERY = {"log_gamma": 1.8, "nrmse": 0.60}
# Deliberately MODEST. A well-calibrated model does not have to score 1.8, and
# the buggy penalties are small numbers: a flat raw penalty of 1.0 converts to
# an objective of -1.0, which beats this trial's -0.5 while losing to -1.8. So
# a test written only against a strongly-calibrated trial passes while the bug
# is present -- verified by mutation, and the reason this case exists.
_MODESTLY_CALIBRATED = {"log_gamma": 0.5, "nrmse": 0.20}


def _run_study(
    trial_metrics: list[dict[str, float] | None],
    objective_metrics: list[str],
    *,
    directions: list[str] | None = None,
) -> optuna.Study:
    """Score a fixed list of trials through the real objective plumbing.

    ``None`` stands for a trial whose validation failed, so the failure paths
    are exercised by the same machinery rather than asserted separately.
    """
    n_obj = len(objective_metrics) + 1
    study = optuna.create_study(
        directions=directions or ["minimize"] * n_obj,
        sampler=optuna.samplers.NSGAIISampler(seed=7),
    )

    def objective(trial: optuna.Trial) -> tuple[float, ...]:
        idx = trial.suggest_int("idx", 0, len(trial_metrics) - 1)
        raw = trial_metrics[idx]
        if raw is None:
            return _training_loss_fallback(
                0.1, objective_metrics, "pareto", 50_000, 1_000_000,
                "param_count",
                tuple(worst_objective_value(m) for m in objective_metrics)
                + (FAILED_TRIAL_COST,),
            )
        cleaned = _validate_metric_keys(dict(raw), objective_metrics)
        return extract_multi_objective_values(
            {"summary": cleaned}, 0.5, objective_metrics,
            objective_mode="pareto",
        )

    # Enumerate every configuration rather than sampling, so the assertion is
    # about ranking and not about search luck.
    for i in range(len(trial_metrics)):
        study.enqueue_trial({"idx": i})
    study.optimize(objective, n_trials=len(trial_metrics))
    return study


def _selected(study: optuna.Study, key: int = 0) -> set[int]:
    """Indices of the configurations on the Pareto front."""
    return {t.params["idx"] for t in study.best_trials}


def test_the_search_prefers_the_well_calibrated_model() -> None:
    """The headline question, and the one no unit test was answering.

    With `log_gamma` unconverted, Optuna minimizes it and picks -25.5 over
    1.8 -- selecting the most miscalibrated model in the study while every
    number in the output looks ordinary.
    """
    study = _run_study(
        [_WELL_CALIBRATED, _MISCALIBRATED], ["log_gamma", "nrmse"]
    )
    assert _selected(study) == {0}, (
        "the miscalibrated configuration was selected or tied; a direction is "
        "inverted somewhere between the metric and Optuna"
    )


def test_a_failed_trial_does_not_win_on_the_calibration_objective() -> None:
    """Scoped to the direction claim, because the broader one is false here.

    A failed trial still receives the clamped training loss on metrics that
    accept the proxy, so with a low loss it can beat a real trial on `nrmse`
    and stay non-dominated. That is the pre-existing training-loss-proxy
    design, not a direction defect, and this branch does not change it -- so
    asserting "a failed trial never reaches the Pareto front" would encode a
    guarantee the codebase does not make.

    What must hold is that the failed trial does not look *well calibrated*:
    `log_gamma` does not accept the proxy, so it takes the worst objective.
    """
    study = _run_study(
        [_WELL_CALIBRATED, None], ["log_gamma", "nrmse"]
    )
    by_idx = {t.params["idx"]: t.values for t in study.trials if t.values}
    good_log_gamma = by_idx[0][0]
    failed_log_gamma = by_idx[1][0]
    assert failed_log_gamma > good_log_gamma
    assert failed_log_gamma == worst_objective_value("log_gamma")


@pytest.mark.parametrize(
    "reported", [_WELL_CALIBRATED, _MODESTLY_CALIBRATED],
    ids=["strong", "modest"],
)
def test_a_trial_that_omitted_the_metric_does_not_win(
    reported: dict[str, float],
) -> None:
    """Missing is not good.

    The penalty is injected before conversion, so it must be a raw-space
    value; a minimize-space one is converted a second time. The `modest` case
    is the one with teeth: a flat raw penalty of 1.0 becomes -1.0, which beats
    a reported log_gamma of 0.5 (-0.5) though it loses to 1.8 (-1.8). Testing
    only the strong case would pass with the bug present.
    """
    study = _run_study([reported, {"nrmse": 0.20}], ["log_gamma", "nrmse"])
    by_idx = {t.params["idx"]: t.values for t in study.trials if t.values}
    assert by_idx[1][0] > by_idx[0][0], (
        "a trial that never reported log_gamma scored better on it than one "
        "that did"
    )
    assert 1 not in _selected(study), (
        "a trial that never reported log_gamma reached the Pareto front"
    )


def test_recovery_still_discriminates_when_calibration_ties() -> None:
    """The direction fix must not flatten the other objective."""
    study = _run_study(
        [_WELL_CALIBRATED, _WORSE_RECOVERY], ["log_gamma", "nrmse"]
    )
    assert _selected(study) == {0}


def test_an_alias_objective_scores_the_reported_value() -> None:
    """Aliases used to make every trial tie at the penalty.

    `cal_error` was absent from the pipeline's canonical metric list, so the
    lookup missed and inserted 1.0 for every trial -- no error, just a study
    in which nothing could be distinguished.
    """
    from bayesflow_hpo.validation.registry import canonical_metric_name

    # What ObjectiveConfig now does at its boundary; asserted directly there
    # in test_objective.py, applied here to check the study that results.
    requested = ["cal_error", "nrmse"]
    objective_metrics = [canonical_metric_name(m) for m in requested]
    assert objective_metrics == ["calibration_error", "nrmse"]

    study = _run_study(
        [{"calibration_error": 0.02, "nrmse": 0.2},
         {"calibration_error": 0.40, "nrmse": 0.2}],
        objective_metrics,
    )
    assert _selected(study) == {0}

    # Without canonicalization the lookup misses and every trial ties at the
    # penalty, so nothing is distinguishable -- the failure this guards.
    tied = _run_study(
        [{"calibration_error": 0.02, "nrmse": 0.2},
         {"calibration_error": 0.40, "nrmse": 0.2}],
        requested,
    )
    assert len(_selected(tied)) == 2, (
        "expected the un-canonicalized form to be indistinguishable; if this "
        "now discriminates, the alias reaches the summary by another route "
        "and this test no longer demonstrates the hazard"
    )


def test_maximize_override_is_refused_rather_than_silently_inverting() -> None:
    """It was a valid workaround before the values became minimize-space."""
    from unittest.mock import MagicMock

    from bayesflow_hpo.api import _derive_directions

    obj = MagicMock()
    obj.n_objectives = 3
    with pytest.raises(ValueError, match="must be all 'minimize'"):
        _derive_directions(
            objective=obj,
            directions=["maximize", "minimize", "minimize"],
            objective_metrics=["log_gamma", "nrmse"],
            objective_mode="pareto",
            cost_metric="inference_time",
        )


def test_infinite_penalties_do_not_break_the_sampler() -> None:
    """`worst_objective` is inf for unbounded metrics, and Optuna must cope."""
    study = _run_study(
        [_WELL_CALIBRATED, {"nrmse": 0.2}], ["log_gamma", "nrmse"]
    )
    values = [t.values[0] for t in study.trials if t.values]
    assert any(math.isinf(v) for v in values)
    assert study.best_trials, "the study produced no Pareto front at all"
