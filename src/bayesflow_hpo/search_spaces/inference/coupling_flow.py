"""Search space for BayesFlow CouplingFlow."""

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
class CouplingFlowSpace(BaseSearchSpace):
    """Search space for `bf.networks.CouplingFlow`.

    Default ranges
    --------------
    cf_depth : int
        Number of coupling layers (2--8).
    cf_subnet_width : int
        MLP width per coupling subnet (32--256, step 32).
    cf_subnet_depth : int
        MLP depth per coupling subnet (1--3).
    cf_dropout : float
        Dropout rate (0.0--0.3).
    cf_activation : str
        **Optional** (off by default). Falls back to BayesFlow's MLP default
        ``"mish"``.

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    cf_transform, cf_permutation, cf_actnorm.
    """

    include_optional: bool = False

    depth: IntDimension = field(
        default_factory=lambda: IntDimension("cf_depth", low=2, high=8)
    )
    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "cf_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("cf_subnet_depth", low=1, high=3)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("cf_dropout", low=0.0, high=0.3)
    )
    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_activation", choices=["silu", "relu", "mish"], default=False
        )
    )

    transform: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_transform", choices=["affine", "spline"], default=False
        )
    )
    permutation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_permutation", choices=["random", "orthogonal"], default=False
        )
    )
    use_actnorm: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_actnorm", choices=[True, False], default=False
        )
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.depth,
            self.subnet_width,
            self.subnet_depth,
            self.dropout,
            self.activation,
            self.transform,
            self.permutation,
            self.use_actnorm,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.CouplingFlow:
        validate_required_params(
            params,
            [
                "cf_depth",
                "cf_subnet_width",
                "cf_subnet_depth",
                "cf_dropout",
            ],
            "CouplingFlowSpace.build",
        )

        width = int(params["cf_subnet_width"])
        n_layers = int(params["cf_subnet_depth"])
        return bf.networks.CouplingFlow(
            depth=int(params["cf_depth"]),
            transform=params.get("cf_transform", "affine"),
            permutation=params.get("cf_permutation", "random"),
            use_actnorm=bool(params.get("cf_actnorm", True)),
            subnet_kwargs={
                "widths": tuple([width] * n_layers),
                "activation": params.get("cf_activation", "mish"),
                "dropout": float(params["cf_dropout"]),
            },
        )
