"""Search space for BayesFlow StableConsistencyModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    Dimension,
    FloatDimension,
    IntDimension,
    validate_required_params,
)


@dataclass
class StableConsistencyModelSpace(BaseSearchSpace):
    """Search space for `bf.networks.StableConsistencyModel`."""

    include_optional: bool = False

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension("scm_subnet_width", low=32, high=256, step=32)
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("scm_subnet_depth", low=1, high=4)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("scm_dropout", low=0.0, high=0.2)
    )

    sigma: FloatDimension = field(
        default_factory=lambda: FloatDimension("scm_sigma", low=0.1, high=2.0, default=False)
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.subnet_width,
            self.subnet_depth,
            self.dropout,
            self.sigma,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.StableConsistencyModel:
        validate_required_params(
            params,
            ["scm_subnet_width", "scm_subnet_depth", "scm_dropout"],
            "StableConsistencyModelSpace.build",
        )

        width = int(params["scm_subnet_width"])
        depth = int(params["scm_subnet_depth"])
        return bf.networks.StableConsistencyModel(
            sigma=float(params.get("scm_sigma", 1.0)),
            subnet_kwargs={
                "widths": tuple([width] * depth),
                "dropout": float(params["scm_dropout"]),
            },
        )
