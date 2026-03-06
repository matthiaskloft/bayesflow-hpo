"""Workflow construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bayesflow as bf
import keras


def _compile_candidate_for_compat(candidate: Any, optimizer: Any) -> None:
    """Try common compile signatures without raising on incompatible variants."""
    compile_fn = getattr(candidate, "compile", None)
    if compile_fn is None:
        return

    try:
        compile_fn()
        return
    except TypeError:
        pass

    try:
        compile_fn(optimizer=optimizer)
        return
    except TypeError:
        pass

    try:
        compile_fn(optimizer)
    except TypeError:
        return


@dataclass
class WorkflowBuildConfig:
    """Runtime settings used to instantiate a BayesFlow workflow.

    Parameters
    ----------
    inference_conditions
        Names of conditioning variables (if any).
    checkpoint_name
        Base name for temporary checkpoints.
    batches_per_epoch
        Batches per epoch — also determines the decay schedule when
        using ExponentialDecay (kept for backward compatibility).
    optimizer
        Custom Keras optimizer.  When ``None`` (default), a cosine-
        annealed Adam optimizer is created automatically.
    epochs
        Total training epochs — used to set ``T_max`` for cosine
        annealing.  Default 200.
    """

    inference_conditions: list[str] | None = None
    checkpoint_name: str = "bayesflow_hpo_trial"
    batches_per_epoch: int = 50
    optimizer: Any | None = None
    epochs: int = 200


def build_workflow(
    simulator: bf.simulators.Simulator,
    adapter: bf.adapters.Adapter,
    inference_network: bf.networks.InferenceNetwork,
    summary_network: bf.networks.SummaryNetwork | None,
    params: dict[str, Any],
    config: WorkflowBuildConfig,
) -> bf.BasicWorkflow:
    """Create and compile a `bf.BasicWorkflow` for one trial.

    By default uses **Adam with CosineDecay**, decaying the learning
    rate from ``initial_lr`` to near-zero over ``config.epochs`` epochs.
    If ``config.optimizer`` is provided it is used directly.
    """
    initial_lr = float(params["initial_lr"])

    if config.optimizer is not None:
        optimizer = config.optimizer
    else:
        total_steps = max(1, config.batches_per_epoch * config.epochs)
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
        optimizer=optimizer,
        initial_learning_rate=initial_lr,
        inference_conditions=config.inference_conditions,
        checkpoint_name=config.checkpoint_name,
    )

    _compile_candidate_for_compat(workflow, optimizer)
    approximator = getattr(workflow, "approximator", None)
    if approximator is not None:
        _compile_candidate_for_compat(approximator, optimizer)

    return workflow
