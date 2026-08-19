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
    """Search space for optimizer/training knobs.

    Batch size and peak learning rate are jointly tunable because their useful
    relationship depends on the workload and compute budget (Smith et al.,
    2018; Shallue et al., 2019).

    References
    ----------
    https://doi.org/10.48550/arXiv.1711.00489
    https://www.jmlr.org/papers/v20/18-789.html
    """

    initial_lr: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "initial_lr", low=1e-4, high=1e-2, log=True
        )
    )
    batch_size: IntDimension = field(
        default_factory=lambda: IntDimension(
            "batch_size", low=32, high=256, step=32
        )
    )
