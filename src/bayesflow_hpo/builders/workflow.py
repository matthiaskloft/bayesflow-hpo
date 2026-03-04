"""Workflow construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bayesflow as bf
import keras


@dataclass
class WorkflowBuildConfig:
    """Runtime settings used to instantiate a BayesFlow workflow."""

    inference_conditions: list[str] | None = None
    checkpoint_name: str = "bayesflow_hpo_trial"


def build_workflow(
    simulator: bf.simulators.Simulator,
    adapter: bf.adapters.Adapter,
    inference_network: bf.networks.InferenceNetwork,
    summary_network: bf.networks.SummaryNetwork | None,
    params: dict[str, Any],
    config: WorkflowBuildConfig,
) -> bf.BasicWorkflow:
    """Create and compile a `bf.BasicWorkflow` for one trial."""
    batch_size = int(params.get("batch_size", 256))
    decay_steps = max(1, batch_size)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(params["initial_lr"]),
        decay_steps=decay_steps,
        decay_rate=float(params.get("decay_rate", 0.95)),
        staircase=True,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
        optimizer=optimizer,
        inference_conditions=config.inference_conditions,
        checkpoint_name=config.checkpoint_name,
    )

    try:
        workflow.approximator.compile(optimizer=optimizer)
    except Exception as exc:
        raise RuntimeError(
            "Failed to compile workflow approximator in build_workflow()."
        ) from exc

    return workflow
