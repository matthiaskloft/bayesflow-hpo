"""Search space for BayesFlow FlowMatching."""

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
class FlowMatchingSpace(BaseSearchSpace):
    """Search space for `bf.networks.FlowMatching`."""

    include_optional: bool = False

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "fm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("fm_subnet_depth", low=1, high=4)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("fm_dropout", low=0.0, high=0.2)
    )
    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_activation", choices=["mish", "silu"]
        )
    )

    use_optimal_transport: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_use_ot", choices=[True, False], default=False
        )
    )
    time_alpha: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "fm_time_alpha", low=0.0, high=2.0, default=False
        )
    )
    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.subnet_width,
            self.subnet_depth,
            self.dropout,
            self.activation,
            self.use_optimal_transport,
            self.time_alpha,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.FlowMatching:
        validate_required_params(
            params,
            ["fm_subnet_width", "fm_subnet_depth", "fm_dropout", "fm_activation"],
            "FlowMatchingSpace.build",
        )

        width = int(params["fm_subnet_width"])
        depth = int(params["fm_subnet_depth"])
        return bf.networks.FlowMatching(
            use_optimal_transport=bool(params.get("fm_use_ot", False)),
            time_power_law_alpha=float(params.get("fm_time_alpha", 0.0)),
            loss_fn="mse",
            subnet_kwargs={
                "widths": tuple([width] * depth),
                "activation": params["fm_activation"],
                "dropout": float(params["fm_dropout"]),
            },
        )
