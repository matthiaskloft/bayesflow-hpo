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

    Default ranges
    --------------
    fm_subnet_width : int
        MLP width (32--256, step 32).
    fm_subnet_depth : int
        MLP depth (1--4).
    fm_dropout : float
        Dropout rate (0.0--0.2).
    fm_activation : str
        **Optional** (off by default). Falls back to BayesFlow's TimeMLP
        default ``"mish"``.

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    fm_use_ot, fm_time_alpha.
    """

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
            "fm_activation", choices=["mish", "silu"], enabled=False
        )
    )

    use_optimal_transport: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_use_ot", choices=[True, False], enabled=False
        )
    )
    time_alpha: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "fm_time_alpha", low=0.0, high=2.0, enabled=False
        )
    )
    time_embedding_dim: IntDimension = field(
        default_factory=lambda: IntDimension(
            "fm_time_embedding_dim", low=8, high=64, step=4, enabled=False
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
        }
        if "fm_activation" in params:
            subnet_kwargs["activation"] = params["fm_activation"]
        if "fm_time_embedding_dim" in params:
            subnet_kwargs["time_embedding_dim"] = int(
                params["fm_time_embedding_dim"]
            )

        kwargs: dict[str, Any] = {
            "subnet_kwargs": subnet_kwargs,
        }
        if "fm_use_ot" in params:
            kwargs["use_optimal_transport"] = bool(params["fm_use_ot"])
        if "fm_time_alpha" in params:
            kwargs["time_power_law_alpha"] = float(params["fm_time_alpha"])

        return bf.networks.FlowMatching(**kwargs)
