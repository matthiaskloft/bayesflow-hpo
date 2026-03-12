"""Training hyperparameter search space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    FloatDimension,
    IntDimension,
)


@dataclass
class TrainingSpace(BaseSearchSpace):
    """Search space for optimizer/training knobs.

    Default dimensions
    ------------------
    initial_lr : float
        Initial learning rate (1e-4--5e-3, log scale).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    batch_size : int
        Training batch size (32--1024, step 32). Defaults to 256.
    """

    initial_lr: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "initial_lr", low=1e-4, high=5e-3, log=True
        )
    )
    batch_size: IntDimension = field(
        default_factory=lambda: IntDimension(
            "batch_size", low=32, high=1024, step=32, enabled=False
        )
    )

    def defaults(self) -> dict[str, Any]:
        """Defaults when optional dimensions are not tuned."""
        return {
            "batch_size": 256,
        }
