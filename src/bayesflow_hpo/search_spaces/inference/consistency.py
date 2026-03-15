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
    batches_per_epoch = int(params.get("batches_per_epoch", 50))
    return max(1, epochs * batches_per_epoch)


@dataclass
class ConsistencyModelSpace(BaseSearchSpace):
    """Search space for `bf.networks.ConsistencyModel`."""

    include_optional: bool = False

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
        default_factory=lambda: IntDimension(
            "cm_max_time", low=50, high=500, enabled=False
        )
    )
    sigma2: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "cm_sigma2", low=0.1, high=2.0, enabled=False
        )
    )
    s0: IntDimension = field(
        default_factory=lambda: IntDimension("cm_s0", low=2, high=30, enabled=False)
    )
    s1: IntDimension = field(
        default_factory=lambda: IntDimension("cm_s1", low=20, high=100, enabled=False)
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

        kwargs: dict[str, Any] = {
            "total_steps": total_steps,
            "subnet_kwargs": {
                "widths": tuple([width] * depth),
                "dropout": float(params["cm_dropout"]),
            },
        }
        if "cm_max_time" in params:
            kwargs["max_time"] = int(params["cm_max_time"])
        if "cm_sigma2" in params:
            kwargs["sigma2"] = float(params["cm_sigma2"])
        if "cm_s0" in params:
            kwargs["s0"] = int(params["cm_s0"])
        if "cm_s1" in params:
            kwargs["s1"] = int(params["cm_s1"])

        return bf.networks.ConsistencyModel(**kwargs)
