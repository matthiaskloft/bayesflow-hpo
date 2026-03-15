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
)


@dataclass
class StableConsistencyModelSpace(BaseSearchSpace):
    """Search space for `bf.networks.StableConsistencyModel`."""

    include_optional: bool = False

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
        default_factory=lambda: FloatDimension(
            "scm_sigma", low=0.1, high=2.0, enabled=False
        )
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
        self._validate(params)

        width = int(params["scm_subnet_width"])
        depth = int(params["scm_subnet_depth"])
        kwargs: dict[str, Any] = {
            "subnet_kwargs": {
                "widths": tuple([width] * depth),
                "dropout": float(params["scm_dropout"]),
            },
        }
        if "scm_sigma" in params:
            kwargs["sigma"] = float(params["scm_sigma"])

        return bf.networks.StableConsistencyModel(**kwargs)
