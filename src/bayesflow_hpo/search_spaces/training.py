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

    initial_lr: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "initial_lr", low=1e-4, high=5e-3, log=True
        )
    )
    batch_size: IntDimension = field(
        default_factory=lambda: IntDimension(
            "batch_size", constant=256
        )
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [self.initial_lr, self.batch_size]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)
