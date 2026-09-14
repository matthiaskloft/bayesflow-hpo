"""A real ``optimize()`` run with the training space's couplings enabled.

The unit tests for ``TrainingSpace`` stop at ``space.sample()``, which is the
half of the mechanism that cannot go wrong quietly.  The half that can is
everything downstream: a derived ``initial_lr`` has to survive the objective's
fallbacks and reach the optimizer, and a derived ``num_batches`` has to be the
step count the trial actually trains for -- and both are *absent* from
``trial.params``, because Optuna records only what it sampled.

Must fail if: the derived learning rate stops reaching the schedule (the
objective then compiles at its hardcoded ``1e-3`` fallback and merely logs a
warning); the config fallback overrides a derived ``num_batches``; or derived
values stop being recorded, which silently truncates ``best_config()`` to the
sampled half of the configuration.
"""

from __future__ import annotations

from typing import Any

import pytest

from bayesflow_hpo.results.extraction import best_config
from bayesflow_hpo.search_spaces.base import FloatDimension, IntDimension
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.search_spaces.inference.flow_matching import FlowMatchingSpace
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace
from bayesflow_hpo.search_spaces.training import TrainingSpace

from .conftest import assert_trials_succeeded

REFERENCE_BATCH = 32
BATCH_SIZE = 64
LR_REF = 1e-3
EPOCHS = 2
SIMULATION_BUDGET = BATCH_SIZE * EPOCHS * 4


def _coupled_search_space() -> CompositeSearchSpace:
    """The tiny pinned space, with both training couplings switched on."""
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
            initial_lr=FloatDimension("initial_lr", constant=LR_REF),
            batch_size=IntDimension("batch_size", constant=BATCH_SIZE),
            epochs=IntDimension("epochs", constant=EPOCHS),
            lr_reference_batch_size=REFERENCE_BATCH,
            simulation_budget=SIMULATION_BUDGET,
        ),
    )


def test_derived_training_values_reach_the_trial_and_the_results(
    run_study: Any,
) -> None:
    """The scaled rate and the budgeted step count are what the trial used."""
    study = run_study(n_trials=1, search_space=_coupled_search_space())

    assert_trials_succeeded(study, expected=1)
    trial = study.trials[0]

    # The optimizer was compiled at the scaled rate, not the sampled `lr_ref`
    # and not the objective's 1e-3 fallback.
    expected_lr = LR_REF * BATCH_SIZE / REFERENCE_BATCH
    assert trial.user_attrs["peak_learning_rate"] == pytest.approx(expected_lr)

    # `num_batches` came from the budget, not from the `num_batches=4` that
    # the `run_study` fixture passes to `optimize()`.
    expected_batches = SIMULATION_BUDGET // (BATCH_SIZE * EPOCHS)
    assert trial.user_attrs["num_batches"] == expected_batches
    assert trial.user_attrs["epochs"] == EPOCHS
    assert trial.user_attrs["simulations"] == SIMULATION_BUDGET

    # Neither derived value is an Optuna parameter, so results helpers have to
    # go and get them.
    assert "initial_lr" not in trial.params
    assert "num_batches" not in trial.params

    config = best_config(study)
    assert config["num_batches"] == expected_batches
    assert float(config["initial_lr"]) == pytest.approx(expected_lr)
