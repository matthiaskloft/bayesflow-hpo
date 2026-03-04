"""Training hyperparameter search space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    Dimension,
    FloatDimension,
    IntDimension,
)


@dataclass
class TrainingSpace(BaseSearchSpace):
    """Search space for optimizer/training knobs."""

    include_optional: bool = False

    initial_lr: FloatDimension = field(
        default_factory=lambda: FloatDimension("initial_lr", low=1e-4, high=5e-3, log=True)
    )
    batch_size: IntDimension = field(
        default_factory=lambda: IntDimension("batch_size", low=32, high=1024, step=32, default=False)
    )
    decay_rate: FloatDimension = field(
        default_factory=lambda: FloatDimension("decay_rate", low=0.8, high=0.99, default=False)
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [self.initial_lr, self.batch_size, self.decay_rate]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def defaults(self) -> dict[str, Any]:
        """Defaults when optional dimensions are not tuned."""
        return {
            "batch_size": 256,
            "decay_rate": 0.95,
        }
