"""Pruned-trial weight retention through a real ``optimize()`` (#106).

The unit tests exercise ``CheckpointPool.save_pruned`` and
``GenericObjective._retain_pruned`` directly.  What they cannot show is
that the objective's ``except optuna.TrialPruned`` blocks actually reach
that call, and reach it while the approximator is still alive -- which is
the whole point of the issue, and the easiest thing to regress by
reordering ``cleanup_trial()``.
"""

from __future__ import annotations

from typing import Any

import optuna

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
