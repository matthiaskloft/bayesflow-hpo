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

Sources for the claims asserted here. The `log_gamma` direction is
BayesFlow's: `calibration_log_gamma` documents the statistic as
`log(gamma/gamma_null)` with `log_gamma < 0` rejecting rank uniformity at the
5% level, so larger is better -- see Modrak et al. (2025), *Bayesian Analysis*
20(2), 461-488, recorded in `docs/references.md`. The ranking claims are
Optuna's: it minimizes every objective whose direction is `minimize`, and
`create_study(load_if_exists=True)` returns the stored study rather than
adopting the directions passed to it (Optuna 4.9.0 docstring), which is why
`TestResumeGuards` exists at all.

They deliberately do **not** assert intermediate values. A test that checks
``_metric_to_minimize`` returns ``-1.5`` passes even when a later stage flips
the sign back; only the selected trial answers the question the user actually
has.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import optuna
import pytest

from bayesflow_hpo.api import _guard_resumed_study
from bayesflow_hpo.objectives import (
    FAILED_TRIAL_COST,
    OBJECTIVE_ENCODING_VERSION,
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


class TestResumeGuards:
    """Resuming is where the encoding change bites, and it bites silently."""

    def _study(
        self, tmp_path: Path, directions: list[str], *, with_trial: bool
    ) -> optuna.Study:
        import optuna as _optuna

        url = "sqlite:///" + str(tmp_path / "s.db").replace("\\", "/")
        study = _optuna.create_study(
            study_name="s", storage=url, directions=directions
        )
        if with_trial:
            study.add_trial(
                _optuna.trial.create_trial(
                    params={}, distributions={}, values=[1.5, 0.2, 0.5]
                )
            )
        return _optuna.create_study(
            study_name="s", storage=url,
            directions=["minimize"] * 3, load_if_exists=True,
        )

    def test_optuna_keeps_the_stored_directions_on_load(self, tmp_path: Path) -> None:
        """The premise of the guard, asserted rather than assumed.

        `create_study(load_if_exists=True)` does not adopt the directions
        passed here; it returns the stored study unchanged. That is why a
        pre-change `maximize` study never reaches `_derive_directions`.
        """
        study = self._study(
            tmp_path, ["maximize", "minimize", "minimize"], with_trial=False
        )
        assert study.directions[0] == optuna.study.StudyDirection.MAXIMIZE

    def test_a_maximize_study_is_refused(self, tmp_path: Path) -> None:
        study = self._study(
            tmp_path, ["maximize", "minimize", "minimize"], with_trial=False
        )
        with pytest.raises(ValueError, match="minimize-is-better"):
            _guard_resumed_study(study, ["log_gamma", "nrmse"])

    def test_pre_encoding_trials_are_refused_for_a_flipped_metric(
        self, tmp_path: Path
    ) -> None:
        """Old raw log_gamma and new negated log_gamma cannot share a front."""
        study = self._study(
            tmp_path, ["minimize"] * 3, with_trial=True
        )
        with pytest.raises(ValueError, match="different scales"):
            _guard_resumed_study(study, ["log_gamma", "nrmse"])

    def test_pre_encoding_trials_are_fine_when_no_metric_flipped(
        self, tmp_path: Path
    ) -> None:
        """`calibration_error` and `nrmse` store the same numbers as before,
        so an old study stays comparable and must not be refused."""
        study = self._study(tmp_path, ["minimize"] * 3, with_trial=True)
        _guard_resumed_study(study, ["calibration_error", "nrmse"])

    def test_a_stamped_study_resumes(self, tmp_path: Path) -> None:
        study = self._study(tmp_path, ["minimize"] * 3, with_trial=True)
        study.set_user_attr(
            "bayesflow_hpo_objective_encoding", OBJECTIVE_ENCODING_VERSION
        )
        _guard_resumed_study(study, ["log_gamma", "nrmse"])

    def test_a_fresh_study_is_stamped(self, tmp_path: Path) -> None:
        study = self._study(tmp_path, ["minimize"] * 3, with_trial=False)
        _guard_resumed_study(study, ["log_gamma", "nrmse"])
        assert (
            study.user_attrs["bayesflow_hpo_objective_encoding"]
            == OBJECTIVE_ENCODING_VERSION
        )

    def test_an_in_flight_trial_blocks_the_stamp(self, tmp_path: Path) -> None:
        """Shared storage: another worker may be mid-trial on the old version.

        Counting only COMPLETE trials let a new worker declare the study
        re-encoded while an old worker still held a RUNNING `log_gamma` trial.
        That trial's raw value then landed in a study stamped as fully
        negated -- encodings mixed behind a guard that had already passed.
        """
        import optuna as _optuna

        url = "sqlite:///" + str(tmp_path / "s.db").replace("\\", "/")
        study = _optuna.create_study(
            study_name="s", storage=url, directions=["minimize"] * 3
        )
        study.ask()  # RUNNING, no values stored yet
        reloaded = _optuna.create_study(
            study_name="s", storage=url,
            directions=["minimize"] * 3, load_if_exists=True,
        )
        with pytest.raises(ValueError, match="different scales"):
            _guard_resumed_study(reloaded, ["log_gamma", "nrmse"])
        assert "bayesflow_hpo_objective_encoding" not in reloaded.user_attrs

    def test_a_warm_started_study_inherits_the_source_encoding(
        self, tmp_path: Path
    ) -> None:
        """Copied trials carry their encoding, so provenance must copy too.

        `create_study(warm_start_from=...)` copies COMPLETE trials into a
        fresh study. Without inheriting the stamp the target looks exactly
        like a legacy study -- completed trials, no encoding attribute -- so
        the guard refused a warm start from an already-valid v2 source.
        """
        from bayesflow_hpo.optimization.study import create_study

        source = self._study(tmp_path, ["minimize"] * 3, with_trial=True)
        source.set_user_attr(
            "bayesflow_hpo_objective_encoding", OBJECTIVE_ENCODING_VERSION
        )
        target = create_study(
            study_name="target",
            directions=["minimize"] * 3,
            storage=None,
            warm_start_from=source,
            warm_start_top_k=1,
        )
        _guard_resumed_study(target, ["log_gamma", "nrmse"])

    def test_a_warm_start_from_a_legacy_source_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        """Inheriting provenance must not degrade into inheriting nothing.

        An unstamped source holds pre-encoding values, so copying its trials
        copies the incompatibility along with them.
        """
        from bayesflow_hpo.optimization.study import create_study

        source = self._study(tmp_path, ["minimize"] * 3, with_trial=True)
        target = create_study(
            study_name="target_legacy",
            directions=["minimize"] * 3,
            storage=None,
            warm_start_from=source,
            warm_start_top_k=1,
        )
        with pytest.raises(ValueError, match="different scales"):
            _guard_resumed_study(target, ["log_gamma", "nrmse"])


def test_a_legacy_set_addition_still_flips_a_registered_lower_metric() -> None:
    """Registering the error metrics must not remove the old override.

    Before `calibration_error` / `nrmse` / `rmse` were registered explicitly,
    `HIGHER_IS_BETTER.add("rmse")` selected the historical `1 - value`
    conversion. Registering them made `_direction_for` return the table entry
    unconditionally, silently taking that away.
    """
    from bayesflow_hpo.objectives import HIGHER_IS_BETTER, _metric_to_minimize

    assert _metric_to_minimize("rmse", 0.8) == pytest.approx(0.8)
    HIGHER_IS_BETTER.add("rmse")
    try:
        assert _metric_to_minimize("rmse", 0.8) == pytest.approx(0.2)
    finally:
        HIGHER_IS_BETTER.discard("rmse")
    assert _metric_to_minimize("rmse", 0.8) == pytest.approx(0.8)


def test_contraction_studies_still_resume(tmp_path: Path) -> None:
    """A false refusal is a real cost, not a safe default.

    `contraction` is higher-is-better AND a usable objective, but its
    conversion (`1 - value`) is identical to before this change, so its stored
    values never moved. Inferring encoding-sensitivity from `higher_is_better`
    would refuse a study that is perfectly comparable and force the user to
    throw away completed trials for nothing. Only `log_gamma` actually
    changed.
    """
    url = "sqlite:///" + str(tmp_path / "c.db").replace("\\", "/")
    study = optuna.create_study(
        study_name="c", storage=url, directions=["minimize"] * 3
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={}, distributions={}, values=[0.2, 0.3, 0.5]
        )
    )
    resumed = optuna.create_study(
        study_name="c", storage=url,
        directions=["minimize"] * 3, load_if_exists=True,
    )
    _guard_resumed_study(resumed, ["contraction", "nrmse"])

    # And passing it must NOT have stamped the study. The stamp asserts every
    # trial was written by this encoding; these were not. Stamping here would
    # let a later `log_gamma` resume sail past the guard -- which is exactly
    # what this assertion caught the first time it was written.
    assert "bayesflow_hpo_objective_encoding" not in resumed.user_attrs
    with pytest.raises(ValueError, match="different scales"):
        _guard_resumed_study(resumed, ["log_gamma", "nrmse"])


def test_a_legacy_set_metric_ranks_missing_strictly_worst() -> None:
    """Set membership gives a direction, never a scale.

    With a finite `worst_raw` of 0.0 a missing metric scored 1.0 while a
    reported -0.5 scored 1.5, so failing to report outranked reporting a bad
    value. This is the same inversion fixed for `correlation` in round 2,
    surviving in the fallback path that serves unregistered metrics -- where
    the range is unknowable, so no finite constant is defensible.
    """
    from bayesflow_hpo.objectives import (
        HIGHER_IS_BETTER,
        _metric_to_minimize,
        worst_objective_value,
    )

    HIGHER_IS_BETTER.add("custom_corr")
    try:
        missing = worst_objective_value("custom_corr")
        for reported in (0.9, 0.5, 0.0, -0.5, -50.0):
            assert _metric_to_minimize("custom_corr", reported) < missing, (
                f"reported {reported} must beat a missing value"
            )
    finally:
        HIGHER_IS_BETTER.discard("custom_corr")


class TestObjectiveSchema:
    """The encoding says how values were written, not what they mean."""

    def _study(self, tmp_path: Path, name: str = "s") -> optuna.Study:
        import optuna as _optuna

        url = "sqlite:///" + str(tmp_path / "s.db").replace("\\", "/")
        return _optuna.create_study(
            study_name=name, storage=url,
            directions=["minimize"] * 3, load_if_exists=True,
        )

    def test_a_fresh_study_records_its_schema(self, tmp_path: Path) -> None:
        study = self._study(tmp_path)
        names = ["log_gamma", "nrmse", "inference_time"]
        _guard_resumed_study(study, ["log_gamma", "nrmse"], names)
        assert study.user_attrs["bayesflow_hpo_objective_schema"] == names

    def test_a_different_metric_set_is_refused(self, tmp_path: Path) -> None:
        """Same width, different meaning -- the case an encoding check misses."""
        study = self._study(tmp_path)
        _guard_resumed_study(
            study, ["log_gamma", "nrmse"],
            ["log_gamma", "nrmse", "inference_time"],
        )
        with pytest.raises(ValueError, match="same column"):
            _guard_resumed_study(
                study, ["calibration_error", "nrmse"],
                ["calibration_error", "nrmse", "inference_time"],
            )

    def test_reordering_the_same_metrics_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Optuna addresses objectives by position, so order is meaning."""
        study = self._study(tmp_path)
        _guard_resumed_study(
            study, ["log_gamma", "nrmse"],
            ["log_gamma", "nrmse", "inference_time"],
        )
        with pytest.raises(ValueError, match="same column"):
            _guard_resumed_study(
                study, ["nrmse", "log_gamma"],
                ["nrmse", "log_gamma", "inference_time"],
            )

    def test_an_unchanged_schema_resumes(self, tmp_path: Path) -> None:
        study = self._study(tmp_path)
        names = ["log_gamma", "nrmse", "inference_time"]
        _guard_resumed_study(study, ["log_gamma", "nrmse"], names)
        _guard_resumed_study(study, ["log_gamma", "nrmse"], list(names))

    def test_callers_without_metric_names_skip_the_check(
        self, tmp_path: Path
    ) -> None:
        """The parameter is optional, and omitting it must not refuse."""
        study = self._study(tmp_path)
        _guard_resumed_study(
            study, ["log_gamma", "nrmse"],
            ["log_gamma", "nrmse", "inference_time"],
        )
        _guard_resumed_study(study, ["log_gamma", "nrmse"])


class TestUnverifiableProvenance:
    """Absent provenance is one case; unrecognized provenance is another."""

    def _study(self, tmp_path: Path, *, with_trial: bool) -> optuna.Study:
        import optuna as _optuna

        url = "sqlite:///" + str(tmp_path / "p.db").replace("\\", "/")
        study = _optuna.create_study(
            study_name="p", storage=url, directions=["minimize"] * 3
        )
        if with_trial:
            study.add_trial(
                _optuna.trial.create_trial(
                    params={}, distributions={}, values=[1.5, 0.2, 0.5]
                )
            )
        return _optuna.create_study(
            study_name="p", storage=url,
            directions=["minimize"] * 3, load_if_exists=True,
        )

    @pytest.mark.parametrize(
        "encoding", [1, 999, "v2", None], ids=["older", "future", "junk", "absent"]
    )
    def test_any_non_current_encoding_is_refused(
        self, tmp_path: Path, encoding
    ) -> None:
        """`!= current`, not `is None`.

        Checking only for an absent attribute waved through an older version,
        a future one written by a newer build, and a malformed caller-written
        value -- three kinds of "provenance I cannot verify" treated as one
        kind of "provenance I have".
        """
        study = self._study(tmp_path, with_trial=True)
        if encoding is not None:
            study.set_user_attr("bayesflow_hpo_objective_encoding", encoding)
        with pytest.raises(ValueError, match="different scales"):
            _guard_resumed_study(study, ["log_gamma", "nrmse"])

    def test_a_stamped_study_with_trials_but_no_schema_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Backfilling a populated study asserts what it cannot check.

        The values are correctly encoded, but which metric wrote each column
        is unknown. Adopting the current run's names would mark it verified,
        and if the columns came from a different same-width metric set every
        later trial mixes meanings behind a schema that now looks checked.
        """
        study = self._study(tmp_path, with_trial=True)
        study.set_user_attr(
            "bayesflow_hpo_objective_encoding", OBJECTIVE_ENCODING_VERSION
        )
        with pytest.raises(ValueError, match="records no objective schema"):
            _guard_resumed_study(
                study, ["log_gamma", "nrmse"],
                ["log_gamma", "nrmse", "inference_time"],
            )
        assert "bayesflow_hpo_objective_schema" not in study.user_attrs

    def test_an_empty_stamped_study_is_backfilled(self, tmp_path: Path) -> None:
        """With nothing stored, there is nothing to contradict."""
        study = self._study(tmp_path, with_trial=False)
        study.set_user_attr(
            "bayesflow_hpo_objective_encoding", OBJECTIVE_ENCODING_VERSION
        )
        names = ["log_gamma", "nrmse", "inference_time"]
        _guard_resumed_study(study, ["log_gamma", "nrmse"], names)
        assert study.user_attrs["bayesflow_hpo_objective_schema"] == names

    def test_a_warm_start_with_a_conflicting_schema_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Validate the source BEFORE copying, not after.

        Inheriting the encoding alone left the target looking verified while
        its columns meant something else: the guard would find no schema,
        record the requested names, and treat copied values as though they
        had always represented those metrics.
        """
        from bayesflow_hpo.optimization.study import create_study

        source = self._study(tmp_path, with_trial=True)
        source.set_user_attr(
            "bayesflow_hpo_objective_encoding", OBJECTIVE_ENCODING_VERSION
        )
        source.set_user_attr(
            "bayesflow_hpo_objective_schema",
            ["calibration_error", "nrmse", "inference_time"],
        )
        with pytest.raises(ValueError, match="mean something else"):
            create_study(
                study_name="ws_conflict",
                directions=["minimize"] * 3,
                metric_names=["log_gamma", "nrmse", "inference_time"],
                storage=None,
                warm_start_from=source,
                warm_start_top_k=1,
            )
