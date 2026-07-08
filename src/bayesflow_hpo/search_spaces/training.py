"""Training hyperparameter search space."""

from __future__ import annotations

from dataclasses import dataclass, field

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
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
