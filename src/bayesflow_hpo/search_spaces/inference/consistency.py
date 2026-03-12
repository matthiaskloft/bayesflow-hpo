"""Search space for BayesFlow ConsistencyModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
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
    """Search space for `bf.networks.ConsistencyModel`.

    Default dimensions
    ------------------
    cm_subnet_width : int
        MLP width (32--256, step 32).
    cm_subnet_depth : int
        MLP depth (1--4).
    cm_dropout : float
        Dropout rate (0.0--0.2).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    cm_max_time : int
        Maximum diffusion time (50--500).
    cm_sigma2 : float
        Noise variance (0.1--2.0).
    cm_s0 : int
        Initial schedule discretisation (2--30).
    cm_s1 : int
        Final schedule discretisation (20--100).
    """

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "cm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("cm_subnet_depth", low=1, high=4)
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

    def build(self, params: dict[str, Any]) -> bf.networks.ConsistencyModel:
        self._validate(params)

        width = int(params["cm_subnet_width"])
        depth = int(params["cm_subnet_depth"])
        total_steps = _compute_total_steps(params)

        return bf.networks.ConsistencyModel(
            total_steps=total_steps,
            max_time=float(params.get("cm_max_time", 200)),
            sigma2=float(params.get("cm_sigma2", 1.0)),
            s0=float(params.get("cm_s0", 10)),
            s1=float(params.get("cm_s1", 50)),
            subnet_kwargs={
                "widths": tuple([width] * depth),
                "dropout": float(params["cm_dropout"]),
            },
        )
