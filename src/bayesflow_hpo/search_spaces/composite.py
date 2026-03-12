"""Composite and selection search spaces.

These classes combine individual network search spaces into a single
object that the objective function can call once to sample all
hyperparameters for one trial.

- **CompositeSearchSpace** — bundles inference + summary + training spaces.
- **NetworkSelectionSpace** — lets Optuna choose *which* inference network
  to use (categorical choice among candidates).
- **SummarySelectionSpace** — same idea for summary networks.

Design decision: the ``_inference_network_type`` / ``_summary_network_type``
keys are prefixed with underscore to signal they are routing metadata
(not actual model hyperparameters) and to avoid collisions with user-
defined parameter names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bayesflow_hpo.search_spaces.base import SearchSpace
from bayesflow_hpo.search_spaces.training import TrainingSpace


@dataclass
class CompositeSearchSpace:
    """Combines inference, summary, and training search spaces.

    Parameters
    ----------
    inference_space
        Search space for the inference (posterior) network.
    summary_space
        Optional search space for the summary (embedding) network.
        When ``None``, the workflow is configured without a summary net.
    training_space
        Search space for optimizer/training knobs (learning rate,
        batch size, etc.).  Defaults to :class:`TrainingSpace`.
    """

    inference_space: SearchSpace
    summary_space: SearchSpace | None = None
    training_space: TrainingSpace = field(default_factory=TrainingSpace)

    def sample(self, trial: Any) -> dict[str, Any]:
        """Sample hyperparameters from all sub-spaces into one dict.

        Training defaults (e.g. ``batch_size=256``) are applied first,
        then overwritten by any actively-tuned training dimensions.
        This ensures every downstream consumer always sees a complete
        parameter dict.

        Parameters
        ----------
        trial
            An ``optuna.Trial`` instance.

        Returns
        -------
        dict[str, Any]
            Merged parameter dict from all sub-spaces.
        """
        params = self.inference_space.sample(trial)
        if self.summary_space is not None:
            params.update(self.summary_space.sample(trial))

        # Apply training defaults first, then overwrite with tuned values.
        training_params = self.training_space.sample(trial)
        params.update(self.training_space.defaults())
        params.update(training_params)
        return params


@dataclass
class NetworkSelectionSpace:
    """Let Optuna choose the inference network type."""

    candidates: dict[str, SearchSpace]

    def sample(self, trial: Any) -> dict[str, Any]:
        """Let Optuna choose a network type, then sample its parameters.

        The chosen type is stored as ``_inference_network_type`` in the
        returned dict so that :meth:`build` can route to the correct
        candidate.

        Parameters
        ----------
        trial
            An ``optuna.Trial`` instance.

        Returns
        -------
        dict[str, Any]
            Parameters from the chosen candidate, plus the routing key.
        """
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
        """Build the inference network selected during sampling.

        Delegates to the candidate search space whose name matches
        ``params["_inference_network_type"]``.

        Parameters
        ----------
        params
            Trial parameters (must include ``_inference_network_type``).

        Returns
        -------
        bf.networks.InferenceNetwork
            Constructed inference network.
        """
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
        """Let Optuna choose a summary network type, then sample its parameters.

        Parameters
        ----------
        trial
            An ``optuna.Trial`` instance.

        Returns
        -------
        dict[str, Any]
            Parameters from the chosen candidate, plus the routing key
            ``_summary_network_type``.
        """
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
        """Build the summary network selected during sampling.

        Parameters
        ----------
        params
            Trial parameters (must include ``_summary_network_type``).

        Returns
        -------
        bf.networks.SummaryNetwork
            Constructed summary network.
        """
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
