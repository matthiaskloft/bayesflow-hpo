"""Pruned-trial weight retention through a real ``optimize()`` (#106).

The unit tests exercise ``CheckpointPool.save_pruned`` and
``GenericObjective._retain_pruned`` directly.  What they cannot show is
that the objective's ``except optuna.TrialPruned`` blocks actually reach
that call, and reach it while the approximator is still alive -- which is
the whole point of the issue, and the easiest thing to regress by
reordering ``cleanup_trial()``.
"""

from __future__ import annotations

import json
from typing import Any

import optuna
import pytest

from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool
from bayesflow_hpo.optimization.objective import default_train_fn


class _PruneAfterTraining:
    """Train, then prune -- exactly where a real pruner raises.

    The first call is ``check_pipeline()``'s pre-flight, which treats any
    exception as a broken hook, so it has to train and return normally.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        default_train_fn(*args, **kwargs)
        if self.calls > 1:
            raise optuna.TrialPruned()


def test_pruned_weights_are_retained_when_enabled(run_study, tmp_path):
    pool = CheckpointPool(
        pool_dir=tmp_path / "retain", pruned_pool_size=4, seed=0,
    )
    study = run_study(train_fn=_PruneAfterTraining(), checkpoint_pool=pool)

    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]
    assert pruned, "expected every trial to prune"
    # A study may run more pruned trials than the cap, so the pool holds a
    # sample of them rather than all -- but every entry must be one.
    assert set(pool.pruned_trial_numbers) <= {t.number for t in pruned}
    assert len(pool.pruned_trial_numbers) == min(
        pool.pruned_pool_size, len(pruned)
    )
    for number in pool.pruned_trial_numbers:
        dest = pool.pruned_pool_dir / f"trial_{number:04d}"
        assert (dest / "weights.weights.h5").is_file()


def test_pruned_weights_are_discarded_by_default(run_study, tmp_path):
    pool = CheckpointPool(pool_dir=tmp_path / "discard")
    study = run_study(train_fn=_PruneAfterTraining(), checkpoint_pool=pool)

    assert any(
        t.state == optuna.trial.TrialState.PRUNED for t in study.trials
    )
    assert pool.pruned_trial_numbers == []
    assert not pool.pruned_pool_dir.exists()


def _sidecar(dest) -> dict[str, Any]:
    return json.loads((dest / "checkpoint.json").read_text())


def test_pool_level_extra_metadata_reaches_completed_checkpoints(
    run_study, tmp_path
):
    """#130: a pool-level setting reaches what ``optimize()`` writes."""
    seen: list[int] = []

    def extra(approximator, trial_number):
        seen.append(trial_number)
        return {"domain_fact": trial_number * 10}

    pool = CheckpointPool(pool_dir=tmp_path / "meta", extra_metadata=extra)
    study = run_study(checkpoint_pool=pool)

    complete = {
        t.number for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    }
    assert pool.trial_numbers, "expected the pool to keep a trial"
    assert set(pool.trial_numbers) <= complete
    for number in pool.trial_numbers:
        meta = _sidecar(tmp_path / "meta" / f"trial_{number:04d}")
        assert meta["domain_fact"] == number * 10
        assert meta["state"] == "complete"
    assert set(pool.trial_numbers) <= set(seen)


def test_pool_level_extra_metadata_reaches_pruned_checkpoints(
    run_study, tmp_path
):
    pool = CheckpointPool(
        pool_dir=tmp_path / "meta_pruned", pruned_pool_size=4, seed=0,
        extra_metadata={"domain_fact": "x"},
    )
    run_study(train_fn=_PruneAfterTraining(), checkpoint_pool=pool)

    assert pool.pruned_trial_numbers, "expected a retained pruned trial"
    for number in pool.pruned_trial_numbers:
        meta = _sidecar(pool.pruned_pool_dir / f"trial_{number:04d}")
        assert meta["domain_fact"] == "x"
        assert meta["state"] == "pruned"


def test_failing_extra_metadata_on_a_completed_trial_stops_the_run(
    run_study, tmp_path
):
    """Documented fail-loud behaviour: no checkpoint without its metadata."""
    calls = {"n": 0}

    def broken(approximator, trial_number):
        calls["n"] += 1
        raise RuntimeError("cannot compute moments")

    pool = CheckpointPool(pool_dir=tmp_path / "broken", extra_metadata=broken)
    with pytest.raises(RuntimeError, match="cannot compute moments"):
        run_study(checkpoint_pool=pool, n_trials=3)
    assert calls["n"] == 1
    assert pool.trial_numbers == []
