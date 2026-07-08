"""Search space for BayesFlow StableConsistencyModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    FloatDimension,
    IntDimension,
)


@dataclass
class StableConsistencyModelSpace(BaseSearchSpace):
    """Search space for `bf.networks.StableConsistencyModel`.

    Fixed dimensions (widen to tune)
    --------------------------------
    scm_sigma : float
        Fixed at ``1.0``.
    """

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "scm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("scm_subnet_depth", low=1, high=6)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("scm_dropout", low=0.0, high=0.2)
    )
    sigma: FloatDimension = field(
        default_factory=lambda: FloatDimension("scm_sigma", constant=1.0)
    )

    def build(self, params: dict[str, Any]) -> bf.networks.StableConsistencyModel:
        self._validate(params)

        width = int(params["scm_subnet_width"])
        depth = int(params["scm_subnet_depth"])

        return bf.networks.StableConsistencyModel(
            subnet_kwargs={
                "widths": tuple([width] * depth),
                "dropout": float(params["scm_dropout"]),
            },
            sigma=float(params["scm_sigma"]),
        )
