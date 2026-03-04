"""Search space for BayesFlow DiffusionModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    CategoricalDimension,
    Dimension,
    FloatDimension,
    IntDimension,
    validate_required_params,
)


@dataclass
class DiffusionModelSpace(BaseSearchSpace):
    """Search space for `bf.networks.DiffusionModel`."""

    include_optional: bool = False

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "dm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("dm_subnet_depth", low=1, high=4)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("dm_dropout", low=0.0, high=0.2)
    )
    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "dm_activation", choices=["mish", "silu"]
        )
    )

    noise_schedule: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "dm_noise_schedule", choices=["edm", "cosine"], default=False
        )
    )
    prediction_type: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "dm_prediction_type", choices=["F", "velocity", "noise", "x"], default=False
        )
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.subnet_width,
            self.subnet_depth,
            self.dropout,
            self.activation,
            self.noise_schedule,
            self.prediction_type,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.DiffusionModel:
        validate_required_params(
            params,
            ["dm_subnet_width", "dm_subnet_depth", "dm_dropout", "dm_activation"],
            "DiffusionModelSpace.build",
        )

        width = int(params["dm_subnet_width"])
        depth = int(params["dm_subnet_depth"])
        return bf.networks.DiffusionModel(
            noise_schedule=params.get("dm_noise_schedule", "edm"),
            prediction_type=params.get("dm_prediction_type", "F"),
            subnet_kwargs={
                "widths": tuple([width] * depth),
                "activation": params["dm_activation"],
                "dropout": float(params["dm_dropout"]),
            },
        )
