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


def test_inventory_matches_the_dataclass():
    """Fail when a field is added that this file has not classified.

    This is the part that makes the inventory self-maintaining. Without it the
    file only ever tests the fields someone remembered to list, which is the
    same weakness that let the defect recur.
    """
    known = {f.name for f in dataclasses.fields(ObjectiveConfig)}
    unknown = _METRIC_NAME_FIELDS - known
    assert not unknown, (
        f"Fields listed here no longer exist on ObjectiveConfig: {unknown}. "
        "Update the inventory."
    )


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
