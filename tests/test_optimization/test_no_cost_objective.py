"""Tests for ``cost_metric=None`` — a study that optimizes quality alone.

Dropping the cost direction is not the same as ignoring the cost column at
selection time.  As an Optuna direction, cost shapes the search: the sampler
models it and spends budget on the cheap-model frontier, and a cheap trial is
non-dominated on that axis however mediocre its quality, so it enters the
Pareto front that selection and warm-start read.  (The non-dominance rule is
Deb et al., 2002, recorded in ``docs/references.md``.)  That budget cannot be
recovered after the fact, which is why the setting exists at all.

Intermediate pruning is *not* part of that story, despite what an earlier
draft of this file and of issue #81 claimed: the strategies in
``optimization/pruning_strategies.py`` compare ``val_{metric}_step_{N}`` user
attrs written from ``objective_metrics`` alone, so cost has never entered a
pruning comparison.  ``test_dominance_pruning_never_reads_cost`` pins that.

What must survive the change: ``param_count`` and ``inference_time_s`` are
still measured and stored as trial user attributes, so post-hoc cost ranking
remains possible, and ``max_param_count`` still constrains what gets built.

The trap this file exists for is :func:`mean_objective_score`.  It drops the
last element as a cost score, and with ``cost_metric=None`` and two quality
metrics that element is ``nrmse`` — a real objective silently missing from the
checkpoint-pool and warm-start rankings, with no error and no warning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bayesflow_hpo.api import _derive_directions
from bayesflow_hpo.objectives import (
    FAILED_TRIAL_CAL_ERROR,
    FAILED_TRIAL_COST,
    extract_multi_objective_values,
    mean_objective_score,
    worst_objective_value,
)
from bayesflow_hpo.optimization.objective import (
    GenericObjective,
    ObjectiveConfig,
    _training_loss_fallback,
)
from bayesflow_hpo.optimization.study import _mean_ranking_key
from bayesflow_hpo.validation.data import ValidationDataset

_DUMMY_VALIDATION_DATA = ValidationDataset(
    simulations=[],
    condition_labels=[],
    param_keys=["p"],
    data_keys=["x"],
    seed=0,
)


class _FakeSearchSpace:
    dimensions: dict[str, Any] = {}

    def sample(self, trial: Any) -> dict[str, Any]:
        return {}


def _objective(**overrides: Any) -> GenericObjective:
    kwargs = dict(
        simulator=object(),
        adapter=object(),
        search_space=_FakeSearchSpace(),
        epochs=1,
        num_batches=1,
        validation_data=_DUMMY_VALIDATION_DATA,
        objective_metrics=["calibration_error", "nrmse"],
        cost_metric=None,
    )
    kwargs.update(overrides)
    return GenericObjective(ObjectiveConfig(**kwargs))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_none_is_an_accepted_cost_metric() -> None:
    assert _objective().config.cost_metric is None


def test_an_unknown_cost_metric_is_still_rejected() -> None:
    """``None`` widening the accepted set must not turn off the check."""
    with pytest.raises(ValueError, match="Unknown cost_metric"):
        _objective(cost_metric="wall_clock")


# ---------------------------------------------------------------------------
# Objective arity and penalties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "metrics", "expected"),
    [
        ("pareto", ["calibration_error", "nrmse"], 2),
        ("pareto", ["calibration_error"], 1),
        ("mean", ["calibration_error", "nrmse"], 1),
        ("mean", ["calibration_error"], 1),
    ],
)
def test_n_objectives_drops_the_cost_column(mode, metrics, expected) -> None:
    objective = _objective(objective_mode=mode, objective_metrics=metrics)
    assert objective.n_objectives == expected


@pytest.mark.parametrize(
    ("mode", "metrics"),
    [
        ("pareto", ["calibration_error", "nrmse"]),
        ("pareto", ["log_gamma"]),
        ("mean", ["calibration_error", "nrmse"]),
        ("mean", ["log_gamma"]),
    ],
)
def test_penalty_shape_matches_n_objectives(mode, metrics) -> None:
    """A penalty tuple of the wrong arity makes Optuna reject the trial."""
    objective = _objective(objective_mode=mode, objective_metrics=metrics)
    penalty = objective._penalty()
    assert len(penalty) == objective.n_objectives
    assert FAILED_TRIAL_COST not in penalty


def test_pareto_penalty_is_still_per_metric() -> None:
    """Removing the cost tail must not flatten the per-metric worst cases."""
    objective = _objective(objective_metrics=["log_gamma", "nrmse"])
    assert objective._penalty() == (
        worst_objective_value("log_gamma"),
        worst_objective_value("nrmse"),
    )


def test_cost_metric_still_produces_the_cost_column() -> None:
    """The default path is unchanged."""
    objective = _objective(cost_metric="inference_time")
    assert objective.n_objectives == 3
    assert objective._penalty()[-1] == FAILED_TRIAL_COST


def test_cost_metric_name_is_none_when_there_is_no_cost_objective() -> None:
    assert _objective()._cost_metric_name is None
    assert _objective(cost_metric="param_count")._cost_metric_name == (
        "param_count_norm"
    )


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("pareto", (0.1, 0.2)), ("mean", (0.15000000000000002,))],
)
def test_extract_omits_the_cost_entry_for_a_none_cost_score(mode, expected) -> None:
    values = extract_multi_objective_values(
        {"summary": {"calibration_error": 0.1, "nrmse": 0.2}},
        cost_score=None,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode=mode,
    )
    assert len(values) == len(expected)
    assert values == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["pareto", "mean"])
def test_training_loss_fallback_omits_the_cost_entry(mode) -> None:
    values = _training_loss_fallback(
        best_training_loss=0.4,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode=mode,
        param_count=1_000,
        max_param_count=10_000,
        cost_metric=None,
        penalty=(FAILED_TRIAL_CAL_ERROR,) * (2 if mode == "pareto" else 1),
    )
    assert len(values) == (2 if mode == "pareto" else 1)
    assert FAILED_TRIAL_COST not in values


# ---------------------------------------------------------------------------
# The silent one: mean_objective_score
# ---------------------------------------------------------------------------


def test_mean_objective_score_keeps_every_metric_without_a_cost_column() -> None:
    """The regression this feature would otherwise introduce.

    With ``cost_metric=None`` and two quality metrics the tuple is
    ``(calibration_error, nrmse)``.  The default ``has_cost=True`` averages
    ``values[:-1]`` and returns ``calibration_error`` alone — a wrong
    checkpoint ranking with nothing in the output looking wrong.
    """
    values = (0.1, 0.9)
    assert mean_objective_score(values, has_cost=False) == pytest.approx(0.5)
    # The default, for contrast: nrmse is gone.
    assert mean_objective_score(values) == pytest.approx(0.1)


def test_mean_objective_score_default_is_unchanged() -> None:
    assert mean_objective_score((0.1, 0.3, 0.2)) == pytest.approx(0.2)
    assert mean_objective_score((0.2, 0.4)) == pytest.approx(0.2)
    assert mean_objective_score((0.42,)) == pytest.approx(0.42)


def test_mean_objective_score_single_value_without_cost() -> None:
    """Mean mode with ``cost_metric=None`` returns exactly one value."""
    assert mean_objective_score((0.42,), has_cost=False) == pytest.approx(0.42)


def test_mean_ranking_key_honours_has_cost() -> None:
    """Warm-start ranking reads the same tuples."""

    class _Trial:
        values = [0.1, 0.9]

    assert _mean_ranking_key(_Trial(), has_cost=False) == pytest.approx(0.5)
    assert _mean_ranking_key(_Trial()) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Directions and metric names
# ---------------------------------------------------------------------------


def test_derive_directions_pareto_without_cost() -> None:
    objective = _objective()
    directions, metric_names = _derive_directions(
        objective=objective,
        directions=None,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="pareto",
        cost_metric=None,
    )
    assert directions == ["minimize", "minimize"]
    assert metric_names == ["calibration_error", "nrmse"]


def test_derive_directions_mean_without_cost() -> None:
    objective = _objective(objective_mode="mean")
    directions, metric_names = _derive_directions(
        objective=objective,
        directions=None,
        objective_metrics=["calibration_error", "nrmse"],
        objective_mode="mean",
        cost_metric=None,
    )
    assert directions == ["minimize"]
    assert metric_names == ["mean(calibration_error+nrmse)"]


def test_explicit_directions_must_match_the_shorter_tuple() -> None:
    objective = _objective()
    with pytest.raises(ValueError, match="directions has 3 entries"):
        _derive_directions(
            objective=objective,
            directions=["minimize"] * 3,
            objective_metrics=["calibration_error", "nrmse"],
            objective_mode="pareto",
            cost_metric=None,
        )


# ---------------------------------------------------------------------------
# Resume across a change of the setting
# ---------------------------------------------------------------------------


def _schema_for(
    cost_metric: str | None,
    metrics: tuple[str, ...] = ("calibration_error", "nrmse"),
) -> tuple[list[str], list[str]]:
    """The directions and metric names ``optimize()`` would produce.

    Derived through ``_derive_directions`` rather than written out, so these
    tests bind the schema guard to what ``cost_metric`` actually generates.
    A hand-written literal would keep passing if the feature stopped changing
    the schema at all.
    """
    return _derive_directions(
        objective=_objective(
            objective_metrics=list(metrics), cost_metric=cost_metric
        ),
        directions=None,
        objective_metrics=list(metrics),
        objective_mode="pareto",
        cost_metric=cost_metric,
    )


def test_resuming_a_cost_bearing_study_without_cost_is_refused(tmp_path) -> None:
    """The arity change must raise, not mis-index column by column.

    A three-column study continued as a two-column one would compare
    ``nrmse`` against the stored ``inference_time`` — Optuna addresses
    objectives by position.
    """
    import optuna

    from bayesflow_hpo.optimization.study import create_study

    with_cost = _schema_for("inference_time")
    without_cost = _schema_for(None)
    assert len(with_cost[1]) == len(without_cost[1]) + 1

    storage = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    create_study(
        study_name="s",
        directions=with_cost[0],
        metric_names=with_cost[1],
        storage=storage,
    )
    study = optuna.load_study(study_name="s", storage=storage)
    study.set_user_attr("bayesflow_hpo_objective_schema", with_cost[1])

    with pytest.raises(ValueError, match="stores objectives"):
        create_study(
            study_name="s",
            directions=without_cost[0],
            metric_names=without_cost[1],
            storage=storage,
            has_cost=False,
        )


def test_warm_start_across_the_setting_is_refused() -> None:
    import optuna

    from bayesflow_hpo.optimization.study import create_study

    with_cost = _schema_for("inference_time")
    without_cost = _schema_for(None)

    source = optuna.create_study(directions=with_cost[0])
    source.add_trial(
        optuna.trial.create_trial(
            params={},
            distributions={},
            values=[0.1, 0.2, 0.3],
            state=optuna.trial.TrialState.COMPLETE,
        )
    )
    source.set_user_attr("bayesflow_hpo_objective_schema", with_cost[1])

    with pytest.raises(ValueError, match="Cannot warm-start"):
        create_study(
            study_name="target",
            directions=without_cost[0],
            metric_names=without_cost[1],
            storage=None,
            warm_start_from=source,
            has_cost=False,
        )


def test_warm_start_without_cost_ranks_on_every_metric() -> None:
    """``has_cost=False`` must reach the ranking key, not just the schema."""
    import optuna

    from bayesflow_hpo.optimization.study import warm_start_study

    source = optuna.create_study(directions=["minimize"] * 2)
    for values in ([0.1, 0.9], [0.2, 0.2]):
        source.add_trial(
            optuna.trial.create_trial(
                params={"a": values[0]},
                distributions={"a": optuna.distributions.FloatDistribution(0, 1)},
                values=values,
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

    target = optuna.create_study(directions=["minimize"] * 2)
    assert warm_start_study(target, source, top_k=1, has_cost=False) == 1
    # Means are 0.5 and 0.2, so the second trial wins. Under has_cost=True
    # only the first column counts and the first trial would win instead.
    assert np.allclose(target.trials[0].values, [0.2, 0.2])


# ---------------------------------------------------------------------------
# Wiring: the keyword must actually reach the helper from the real call path
# ---------------------------------------------------------------------------


def test_create_study_stamps_whether_the_last_column_is_cost() -> None:
    """Readers cannot recover this from the column names alone.

    ``cost_metric=None`` over two quality metrics and
    ``cost_metric="param_count"`` over one both produce two columns.
    """
    from bayesflow_hpo.optimization.study import create_study

    with_cost = create_study(
        study_name="a", storage=None,
        directions=["minimize"] * 3,
        metric_names=["calibration_error", "nrmse", "inference_time"],
    )
    without = create_study(
        study_name="b", storage=None,
        directions=["minimize"] * 2,
        metric_names=["calibration_error", "nrmse"],
        has_cost=False,
    )
    assert with_cost.user_attrs["bayesflow_hpo_has_cost_objective"] is True
    assert without.user_attrs["bayesflow_hpo_has_cost_objective"] is False


def test_dominance_pruning_never_reads_cost() -> None:
    """Pins the corrected claim: cost is not part of any pruning comparison.

    An earlier draft of this feature (and issue #81) asserted that
    ``pruning_strategy="dominance"`` keeps cheap trials alive because they are
    non-dominated on the cost axis.  It does not: the reference vectors come
    from ``val_{metric}_step_{N}`` attrs written from ``objective_metrics``
    alone.
    """
    import inspect

    from bayesflow_hpo.optimization import pruning_strategies

    source = inspect.getsource(pruning_strategies)
    for token in ("cost", "inference_time", "param_count"):
        assert token not in source, (
            f"{token!r} appears in pruning_strategies.py; the docstrings in "
            f"api.py, docs/optimization.md and CHANGELOG.md state that cost "
            f"never enters a pruning comparison and would need updating"
        )


def test_no_metrics_and_no_cost_is_rejected() -> None:
    """The one configuration that would leave zero objectives.

    Optuna refuses a study with no directions ("The number of objectives
    must be greater than 0", `optuna/study/study.py:1264` on 5.0.0), but
    only once `create_study` is reached — after the search space,
    validation data and pre-flight have been built, and with a message
    naming neither setting responsible.
    """
    with pytest.raises(ValueError, match="objective_metrics is empty"):
        _objective(objective_metrics=[], cost_metric=None)


def test_no_metrics_with_a_cost_metric_is_still_allowed() -> None:
    """Unchanged: a cost column alone is still one objective."""
    assert _objective(
        objective_metrics=[], cost_metric="param_count"
    ).n_objectives == 1
