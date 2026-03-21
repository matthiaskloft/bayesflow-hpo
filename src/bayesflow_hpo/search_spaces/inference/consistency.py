"""Search space for BayesFlow ConsistencyModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    Dimension,
    FloatDimension,
    IntDimension,
)


def _compute_total_steps(params: dict[str, Any]) -> int:
    if "cm_total_steps" in params:
        return max(1, int(params["cm_total_steps"]))
    if "total_steps" in params:
        return max(1, int(params["total_steps"]))

    epochs = int(params.get("epochs", params.get("n_epochs", 200)))
    num_batches = int(params.get("num_batches", 50))
    return max(1, epochs * num_batches)


@dataclass
class ConsistencyModelSpace(BaseSearchSpace):
    """Search space for `bf.networks.ConsistencyModel`.

    Default dimensions (tuned)
    --------------------------
    cm_subnet_width, cm_subnet_depth, cm_dropout.

    Fixed dimensions (widen to tune)
    --------------------------------
    cm_max_time : int
        Fixed at ``200``.
    cm_sigma2 : float
        Fixed at ``1.0``.
    cm_s0 : int
        Fixed at ``10``.
    cm_s1 : int
        Fixed at ``50``.
    """

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "cm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("cm_subnet_depth", low=1, high=6)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("cm_dropout", low=0.0, high=0.2)
    )
    max_time: IntDimension = field(
        default_factory=lambda: IntDimension("cm_max_time", constant=200)
    )
    sigma2: FloatDimension = field(
        default_factory=lambda: FloatDimension("cm_sigma2", constant=1.0)
    )
    s0: IntDimension = field(
        default_factory=lambda: IntDimension("cm_s0", constant=10)
    )
    s1: IntDimension = field(
        default_factory=lambda: IntDimension("cm_s1", constant=50)
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.subnet_width,
            self.subnet_depth,
            self.dropout,
            self.max_time,
            self.sigma2,
            self.s0,
            self.s1,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.ConsistencyModel:
        self._validate(params)

        width = int(params["cm_subnet_width"])
        depth = int(params["cm_subnet_depth"])
        total_steps = _compute_total_steps(params)

        return bf.networks.ConsistencyModel(
            total_steps=total_steps,
            subnet_kwargs={
                "widths": tuple([width] * depth),
                "dropout": float(params["cm_dropout"]),
            },
            max_time=int(params["cm_max_time"]),
            sigma2=float(params["cm_sigma2"]),
            s0=int(params["cm_s0"]),
            s1=int(params["cm_s1"]),
        )
