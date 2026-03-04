"""Composite and selection search spaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bayesflow_hpo.search_spaces.base import SearchSpace
from bayesflow_hpo.search_spaces.training import TrainingSpace


@dataclass
class CompositeSearchSpace:
    """Combines inference, summary and training search spaces."""

    inference_space: SearchSpace
    summary_space: SearchSpace | None = None
    training_space: TrainingSpace = field(default_factory=TrainingSpace)

    def sample(self, trial: Any) -> dict[str, Any]:
        params = self.inference_space.sample(trial)
        if self.summary_space is not None:
            params.update(self.summary_space.sample(trial))

        training_params = self.training_space.sample(trial)
        params.update(self.training_space.defaults())
        params.update(training_params)
        return params


@dataclass
class NetworkSelectionSpace:
    """Let Optuna choose the inference network type."""

    candidates: dict[str, SearchSpace]

    def sample(self, trial: Any) -> dict[str, Any]:
        if not self.candidates:
            raise ValueError("NetworkSelectionSpace requires at least one candidate.")
        network_type = trial.suggest_categorical(
            "inference_network_type",
            list(self.candidates.keys()),
        )
        params = self.candidates[network_type].sample(trial)
        params["_inference_network_type"] = network_type
        return params

    def build(self, params: dict[str, Any]) -> Any:
        if "_inference_network_type" not in params:
            message = (
                "Missing '_inference_network_type' in params "
                "for NetworkSelectionSpace.build()."
            )
            raise ValueError(
                message
            )
        network_type = params["_inference_network_type"]
        if network_type not in self.candidates:
            raise KeyError(f"Unknown inference network type: {network_type}")
        return self.candidates[network_type].build(params)


@dataclass
class SummarySelectionSpace:
    """Let Optuna choose the summary network type."""

    candidates: dict[str, SearchSpace]

    def sample(self, trial: Any) -> dict[str, Any]:
        if not self.candidates:
            raise ValueError("SummarySelectionSpace requires at least one candidate.")
        network_type = trial.suggest_categorical(
            "summary_network_type",
            list(self.candidates.keys()),
        )
        params = self.candidates[network_type].sample(trial)
        params["_summary_network_type"] = network_type
        return params

    def build(self, params: dict[str, Any]) -> Any:
        if "_summary_network_type" not in params:
            message = (
                "Missing '_summary_network_type' in params "
                "for SummarySelectionSpace.build()."
            )
            raise ValueError(
                message
            )
        network_type = params["_summary_network_type"]
        if network_type not in self.candidates:
            raise KeyError(f"Unknown summary network type: {network_type}")
        return self.candidates[network_type].build(params)
