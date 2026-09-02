"""Every configuration field naming a metric must be canonicalized.

This file exists because the same defect was found five separate times in
review: a metric name normalized at one site and read un-normalized at
another. Each instance was fixed individually, and each fix left the next
instance in place -- the pipeline metric list, ``early_stopping_monitor``, the
``pruning_strategy`` tuple, hard and soft constraint specs, and custom
``validate_fn`` output.

The failure is silent in every case. Nothing raises: the lookup misses, and a
penalty, a zero violation, or a ``KeyError``-to-penalty conversion takes the
place of the real value. So the defect cannot be caught by "does it run".

Rather than assert the five known cases, this enumerates the fields of
:class:`ObjectiveConfig` that hold a metric name and checks each one accepts an
alias. A field added later joins the inventory automatically and fails here
until it is canonicalized too.
"""

from __future__ import annotations

import dataclasses

import pytest

from bayesflow_hpo.optimization.objective import ObjectiveConfig
from bayesflow_hpo.validation.data import ValidationDataset


class _FakeSimulator:
    def sample(self, shape):
        return {}


class _FakeAdapter:
    def __call__(self, data):
        return data


class _FakeInferenceSpace:
    def sample(self, trial):
        return {}


class _FakeSearchSpace:
    def __init__(self):
        self.inference_space = _FakeInferenceSpace()
        self.summary_space = None

    def sample(self, trial):
        return {"initial_lr": 1e-3}


_DUMMY_VALIDATION_DATA = ValidationDataset(
    simulations=[],
    condition_labels=[],
    param_keys=["p"],
    data_keys=["x"],
    seed=0,
)

# Fields of ObjectiveConfig whose value is, or contains, a metric name.
# `cost_metric` is deliberately excluded: "inference_time" is measured by the
# objective itself and is not a registry metric.
_METRIC_NAME_FIELDS = {
    "objective_metrics",
    "early_stopping_monitor",
    "pruning_strategy",
    "metric_constraints_hard",
    "metric_constraints_soft",
}

# Every other field of ObjectiveConfig. Listing these is what makes the
# inventory exhaustive: a newly added field belongs to exactly one of the two
# sets, and `test_inventory_partitions_the_dataclass` fails until someone says
# which. `cost_metric` sits here deliberately -- "inference_time" is measured
# by the objective itself and is not a registry metric.
_NON_METRIC_FIELDS = {
    "adapter",
    "build_approximator_fn",
    "checkpoint_pool",
    "cost_metric",
    "early_stopping_patience",
    "early_stopping_window",
    "epochs",
    "intermediate_validation_interval",
    "intermediate_validation_warmup",
    "lr_warmup_epochs",
    "lr_warmup_fraction",
    "lr_warmup_steps",
    "max_memory_mb",
    "max_param_count",
    "n_intermediate_posterior_samples",
    "n_posterior_samples",
    "num_batches",
    "objective_mode",
    "pruning_n_startup_trials",
    "report_frequency",
    "search_space",
    "simulator",
    "train_fn",
    "training_mode",
    "validate_fn",
    "validation_data",
}

# One registered alias and the canonical name it must resolve to.
_ALIAS = "cal_error"
_CANONICAL = "calibration_error"


def _config(**overrides) -> ObjectiveConfig:
    """Build a config with the given overrides, everything else defaulted."""
    base = {
        "simulator": _FakeSimulator(),
        "adapter": _FakeAdapter(),
        "search_space": _FakeSearchSpace(),
        "validation_data": _DUMMY_VALIDATION_DATA,
        "objective_metrics": [_CANONICAL, "nrmse"],
    }
    base.update(overrides)
    return ObjectiveConfig(**base)


def test_inventory_partitions_the_dataclass():
    """Every field must be classified as metric-bearing or not.

    This is what makes the inventory self-maintaining, and the first version
    of this test did not deliver it: asserting `_METRIC_NAME_FIELDS - known`
    is empty catches a field that was *removed*, while a newly added
    metric-bearing field simply grows `known` and leaves the difference empty.
    The test therefore advertised a guarantee it did not provide -- the same
    shape of mistake as the defect it guards, a claim that reads as checked
    and is not.

    An exhaustive partition fails in both directions: a new field belongs to
    exactly one set, and until someone says which, this fails.
    """
    known = {f.name for f in dataclasses.fields(ObjectiveConfig)}
    classified = _METRIC_NAME_FIELDS | _NON_METRIC_FIELDS

    assert not (classified - known), (
        f"Fields listed here no longer exist on ObjectiveConfig: "
        f"{sorted(classified - known)}. Update the inventory."
    )
    assert not (known - classified), (
        f"Unclassified ObjectiveConfig fields: {sorted(known - classified)}. "
        "Add each to _METRIC_NAME_FIELDS (and canonicalize it in "
        "__post_init__) or to _NON_METRIC_FIELDS."
    )
    assert not (_METRIC_NAME_FIELDS & _NON_METRIC_FIELDS)


@pytest.mark.parametrize("field_name", sorted(_METRIC_NAME_FIELDS))
def test_field_canonicalizes_aliases(field_name):
    """Each metric-name field must accept an alias and store the canonical."""
    if field_name == "objective_metrics":
        cfg = _config(objective_metrics=[_ALIAS, "nrmse"])
        assert cfg.objective_metrics == [_CANONICAL, "nrmse"]

    elif field_name == "early_stopping_monitor":
        cfg = _config(early_stopping_monitor=_ALIAS)
        assert cfg.early_stopping_monitor == _CANONICAL

    elif field_name == "pruning_strategy":
        cfg = _config(pruning_strategy=("primary", _ALIAS))
        assert cfg.pruning_strategy == ("primary", _CANONICAL)

    elif field_name in ("metric_constraints_hard", "metric_constraints_soft"):
        cfg = _config(**{field_name: [(_ALIAS, 0.1, "below")]})
        assert getattr(cfg, field_name) == [(_CANONICAL, 0.1, "below")]

    else:  # pragma: no cover - guarded by the inventory test above
        pytest.fail(f"No canonicalization check written for {field_name!r}")


def test_unregistered_names_pass_through_unchanged():
    """Custom metric names must survive canonicalization untouched.

    A caller's own ``validate_fn`` may emit names the registry has never seen,
    and rewriting those would break the very hooks this is meant to protect.
    """
    cfg = _config(
        objective_metrics=["my_custom_metric"],
        early_stopping_monitor="my_custom_metric",
        pruning_strategy=("primary", "my_custom_metric"),
        metric_constraints_hard=[("coverage_90", 0.8, "above")],
    )
    assert cfg.objective_metrics == ["my_custom_metric"]
    assert cfg.early_stopping_monitor == "my_custom_metric"
    assert cfg.pruning_strategy == ("primary", "my_custom_metric")
    assert cfg.metric_constraints_hard == [("coverage_90", 0.8, "above")]


def test_pruning_metric_matches_the_key_the_pipeline_writes():
    """The concrete failure: an aliased pruning metric indexed a canonical dict.

    ``PeriodicValidationCallback`` builds its score dict from the pipeline,
    which emits canonical names, then indexes it with ``pruning_strategy[1]``.
    An alias there raised ``KeyError`` at every intermediate validation, which
    the trial handler converts into a training-failure penalty -- so the trial
    looked merely bad rather than broken.
    """
    cfg = _config(
        objective_metrics=[_ALIAS, "nrmse"],
        pruning_strategy=("primary", _ALIAS),
    )
    pruning_metric = cfg.pruning_strategy[1]
    assert pruning_metric in cfg.objective_metrics


# `metric_constraints_soft` is now a field of ObjectiveConfig -- it has to be,
# so its metrics get COMPUTED rather than silently reading as zero violation --
# which is what brought it into the inventory above. It is still canonicalized
# at the API boundary as well, because `optimize()` hands its own copy to
# `create_study` without going through the config.


def test_default_validator_computes_non_default_objectives(monkeypatch):
    """The built-in validator must compute the metrics the run optimizes.

    `log_gamma`, `sbc_ks` and `sbc_chi2` are registered but not in
    DEFAULT_METRICS. While the validator ignored `objective_metrics` it
    computed DEFAULT_METRICS only, so pre-flight saw the objective as a
    missing key and raised before training started -- the headline metric of
    this change could not be optimized through the public workflow at all.
    """
    from bayesflow_hpo.optimization import objective as objective_mod
    from bayesflow_hpo.validation import pipeline as pipeline_mod

    seen = {}

    class _Result:
        summary = {"log_gamma": 1.0}
        timing: dict = {}

    def _fake_pipeline(*, metrics=None, **kwargs):
        seen["metrics"] = metrics
        return _Result()

    monkeypatch.setattr(pipeline_mod, "run_validation_pipeline", _fake_pipeline)

    objective_mod.default_validate_fn(
        object(), _DUMMY_VALIDATION_DATA, 2, objective_metrics=["log_gamma"]
    )
    assert "log_gamma" in seen["metrics"], (
        "default_validate_fn must forward its objectives to the pipeline"
    )
    # DEFAULT_METRICS stay in: constraints may reference metrics that can
    # never be objectives, and restricting the list would disable them.
    assert "calibration_error" in seen["metrics"]


def test_constraint_metrics_are_computed_by_the_pipeline():
    """A constraint on a non-default metric must actually be measured.

    Neither constraint path complains when the metric is absent: the hard
    path skips a missing key, and the soft callback reads a missing user
    attribute as zero violation. So a constraint naming `sbc_ks` alongside
    the default objectives was configured, inactive, and silent.
    """
    from bayesflow_hpo.optimization.objective import (
        _constraint_metric_names,
        _pipeline_metrics,
    )

    cfg = _config(
        metric_constraints_hard=[("sbc_ks", 0.2, "above")],
        metric_constraints_soft=[("sbc_chi2", 5.0, "below")],
    )
    names = _constraint_metric_names(cfg)
    assert names == ["sbc_ks", "sbc_chi2"]

    computed = _pipeline_metrics(cfg.objective_metrics, names)
    assert "sbc_ks" in computed
    assert "sbc_chi2" in computed
    # Objectives and DEFAULT_METRICS are still there.
    assert "calibration_error" in computed


def test_soft_constraint_specs_are_canonicalized():
    """Soft constraints reach ObjectiveConfig now, so they join the inventory."""
    cfg = _config(metric_constraints_soft=[(_ALIAS, 0.1, "below")])
    assert cfg.metric_constraints_soft == [(_CANONICAL, 0.1, "below")]


class TestRoundSevenFindings:
    """Regressions for the round-7 review findings, each reproduced first."""

    def test_a_direction_registered_under_an_alias_is_found(self):
        """The seventh instance of the canonicalization defect.

        `optimize()` canonicalizes the objective name, so a direction stored
        under the alias was never consulted: the built-in lower-is-better
        conversion applied instead, silently, inverting the search for a
        metric the caller had just declared higher-is-better.
        """
        from bayesflow_hpo.objectives import (
            HIGHER_IS_BETTER,
            METRIC_DIRECTIONS,
            _direction_for,
            register_metric_direction,
        )

        saved = METRIC_DIRECTIONS.get(_CANONICAL), _CANONICAL in HIGHER_IS_BETTER
        try:
            register_metric_direction(_ALIAS, higher_is_better=True, worst_raw=0.0)
            assert _ALIAS not in METRIC_DIRECTIONS
            assert METRIC_DIRECTIONS[_CANONICAL].higher_is_better
            direction = _direction_for(_CANONICAL)
            assert direction is not None and direction.higher_is_better
        finally:
            if saved[0] is not None:
                METRIC_DIRECTIONS[_CANONICAL] = saved[0]
            else:
                METRIC_DIRECTIONS.pop(_CANONICAL, None)
            if saved[1]:
                HIGHER_IS_BETTER.add(_CANONICAL)
            else:
                HIGHER_IS_BETTER.discard(_CANONICAL)

    def test_a_constraint_on_an_output_key_computes_its_producer(self):
        """A constraint names an output key, not the metric behind it.

        `coverage_left` emits `left_coverage_90`; that key is not itself
        registered, so filtering the pipeline list on registered names dropped
        the producer. Nothing computed the key, and neither constraint path
        complains -- the hard path skips a missing key, the soft path reads it
        as zero violation.
        """
        from bayesflow_hpo.optimization.objective import _pipeline_metrics
        from bayesflow_hpo.validation.registry import producer_for_key

        assert producer_for_key("left_coverage_90") == "coverage_left"
        assert producer_for_key("mean_abs_z_score") == "z_score"
        assert producer_for_key("nrmse") == "nrmse"
        assert producer_for_key("not_a_metric_key") is None

        computed = _pipeline_metrics([_CANONICAL, "nrmse"], ["left_coverage_90"])
        assert "coverage_left" in computed

    def test_a_custom_objective_counts_as_encoding_sensitive(self):
        """`ENCODING_CHANGED_AT_V2` cannot list a caller's own metric.

        An unregistered name's penalty moved from a finite 1.0 to +inf, so a
        legacy custom objective is exactly as incomparable as `log_gamma` --
        but membership of a fixed set can never say so. The question is
        provable absence of change, not presence in the list.
        """
        from bayesflow_hpo.api import _encoding_sensitive

        assert _encoding_sensitive("log_gamma")
        assert _encoding_sensitive("my_custom_metric")
        assert not _encoding_sensitive("nrmse")
        assert not _encoding_sensitive(_CANONICAL)

    def test_mean_mode_schemas_ignore_member_order(self):
        """Mean mode stores one average, so the member order is not meaning."""
        from bayesflow_hpo.objectives import schema_matches

        assert schema_matches(
            ["mean(nrmse+log_gamma)", "inference_time"],
            ["mean(log_gamma+nrmse)", "inference_time"],
        )
        # Pareto columns keep their order: Optuna addresses them by position.
        assert not schema_matches(
            ["log_gamma", "nrmse", "t"], ["nrmse", "log_gamma", "t"]
        )
        assert not schema_matches(["mean(a+b)", "t"], ["mean(a+c)", "t"])


class TestRoundEightFindings:
    """Regressions for round 8 -- three of which my round-7 fixes introduced."""

    def test_a_registered_custom_metric_is_still_encoding_sensitive(self):
        """"Registered" is not "audited".

        The round-7 fix asked `list_metrics()` whether a metric was known, but
        that is the LIVE registry: a custom metric installed through
        `register_metric()` read as known and therefore unchanged, while its
        penalty moved from a finite 1.0 to +inf like any unregistered name's.
        Only the frozen audited list can exempt a metric.
        """
        from bayesflow_hpo.api import _encoding_sensitive
        from bayesflow_hpo.validation import registry

        name = "custom_for_encoding_test"
        registry.register_metric(
            name,
            lambda draws, true_values: {name: 0.0},
            description="Test-only metric.",
            overwrite=True,
        )
        try:
            assert _encoding_sensitive(name)
            assert _encoding_sensitive("log_gamma")
            assert not _encoding_sensitive("nrmse")
        finally:
            # The registry is process-global and exposes no removal API, so
            # the private tables are the only way to undo this. Leaving the
            # entry behind broke an unrelated test that asserts every
            # registered metric is a documented built-in.
            for table in (
                registry._REGISTRY,
                registry._DESCRIPTIONS,
                registry._KINDS,
            ):
                table.pop(name, None)

    def test_the_audited_sets_partition_the_builtins(self):
        """Neither set may drift from the registry without someone noticing."""
        from bayesflow_hpo.objectives import (
            ENCODING_CHANGED_AT_V2,
            ENCODING_UNCHANGED_AT_V2,
        )

        assert not (ENCODING_CHANGED_AT_V2 & ENCODING_UNCHANGED_AT_V2)

    def test_warm_start_source_schemas_tolerate_mean_order(self):
        """The mean-mode fix has to cover the warm-start site as well.

        `_guard_resumed_study` compared through `schema_matches`, but the
        warm-start source check still used list inequality, so a mean-mode
        source stamped by an earlier build was refused for member order alone.
        """
        from bayesflow_hpo.objectives import schema_matches

        assert schema_matches(
            ["mean(nrmse+log_gamma)", "inference_time"],
            ["mean(log_gamma+nrmse)", "inference_time"],
        )

    def test_a_non_string_schema_member_refuses_rather_than_raising(self):
        """`user_attrs` is caller-writable and round-trips through JSON.

        Both callers type-check the outer container but not its members, so a
        schema holding a number or a null reached the normalizer and raised
        `AttributeError` instead of producing the guard's refusal.
        """
        from bayesflow_hpo.objectives import schema_matches

        stored: list[str] = [1, "nrmse"]  # type: ignore[list-item]
        assert not schema_matches(stored, ["log_gamma", "nrmse"])
        current: list[str] = [None, "nrmse"]  # type: ignore[list-item]
        assert not schema_matches(["log_gamma", "nrmse"], current)

    def test_a_populated_unstamped_study_without_schema_is_refused(self):
        """The refusal must not depend on the encoding stamp.

        It was nested under `encoding == OBJECTIVE_ENCODING_VERSION`, so a
        legacy study with neither encoding nor schema slipped through whenever
        its metrics were encoding-insensitive -- and resuming that with a
        different same-width metric set mixes objectives by column with
        nothing to compare against.
        """
        import optuna

        from bayesflow_hpo.api import _guard_resumed_study

        study = optuna.create_study(directions=["minimize"] * 2)
        study.add_trial(
            optuna.trial.create_trial(
                params={},
                distributions={},
                values=[0.1, 0.2],
                state=optuna.trial.TrialState.COMPLETE,
            )
        )
        # `nrmse` is encoding-insensitive, so nothing else refuses this.
        with pytest.raises(ValueError, match="records no objective schema"):
            _guard_resumed_study(
                study, ["nrmse", "rmse"], metric_names=["nrmse", "rmse"]
            )
