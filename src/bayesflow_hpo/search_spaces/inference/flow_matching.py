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
)


@dataclass
class FlowMatchingSpace(BaseSearchSpace):
    """Search space for `bf.networks.FlowMatching`.

    Default dimensions (tuned)
    --------------------------
    fm_subnet_width : int
        MLP width (32--256, step 32).
    fm_subnet_depth : int
        MLP depth (1--6).  BayesFlow default TimeMLP uses 5 layers.
    fm_dropout : float
        Dropout rate (0.0--0.2).

    Fixed dimensions (widen to tune)
    --------------------------------
    fm_activation : str
        Subnet activation. Fixed at ``"mish"`` (TimeMLP default).
    fm_use_optimal_transport : bool
        Optimal transport. Fixed at ``False``.
    fm_time_power_law_alpha : float
        Time power-law alpha. Fixed at ``0.0``.
    fm_time_embedding_dim : int
        Time embedding dimensionality. Fixed at ``8``.
    """

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "fm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("fm_subnet_depth", low=1, high=6)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("fm_dropout", low=0.0, high=0.2)
    )
    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_activation", constant="mish"
        )
    )
    use_optimal_transport: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_use_optimal_transport", constant=False
        )
    )
    time_alpha: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "fm_time_power_law_alpha", constant=0.0
        )
    )
    time_embedding_dim: IntDimension = field(
        default_factory=lambda: IntDimension(
            "fm_time_embedding_dim", constant=8
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
            self.time_embedding_dim,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.FlowMatching:
        self._validate(params)

        width = int(params["fm_subnet_width"])
        depth = int(params["fm_subnet_depth"])

        subnet_kwargs: dict[str, Any] = {
            "widths": tuple([width] * depth),
            "dropout": float(params["fm_dropout"]),
            "activation": params["fm_activation"],
            "time_embedding_dim": int(params["fm_time_embedding_dim"]),
        }

        return bf.networks.FlowMatching(
            subnet_kwargs=subnet_kwargs,
            use_optimal_transport=bool(params["fm_use_optimal_transport"]),
            time_power_law_alpha=float(params["fm_time_power_law_alpha"]),
        )
