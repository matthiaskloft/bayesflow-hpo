"""Search space for BayesFlow SetTransformer."""

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
class SetTransformerSpace(BaseSearchSpace):
    """Search space for `bf.networks.SetTransformer`."""

    include_optional: bool = False

    summary_dim: IntDimension = field(
        default_factory=lambda: IntDimension("st_summary_dim", low=8, high=64, step=8)
    )
    embed_dim: IntDimension = field(
        default_factory=lambda: IntDimension("st_embed_dim", low=32, high=256, step=32)
    )
    num_heads: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "st_num_heads", choices=[1, 2, 4, 8]
        )
    )
    num_layers: IntDimension = field(
        default_factory=lambda: IntDimension("st_num_layers", low=1, high=4)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("st_dropout", low=0.0, high=0.3)
    )

    mlp_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "st_mlp_width", low=64, high=512, step=64, enabled=False
        )
    )
    mlp_depth: IntDimension = field(
        default_factory=lambda: IntDimension(
            "st_mlp_depth", low=1, high=4, enabled=False
        )
    )
    num_inducing: IntDimension = field(
        default_factory=lambda: IntDimension(
            "st_num_inducing", low=8, high=64, step=8, enabled=False
        )
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.summary_dim,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.dropout,
            self.mlp_width,
            self.mlp_depth,
            self.num_inducing,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.SetTransformer:
        validate_required_params(
            params,
            [
                "st_summary_dim",
                "st_embed_dim",
                "st_num_heads",
                "st_num_layers",
                "st_dropout",
            ],
            "SetTransformerSpace.build",
        )

        num_layers = int(params["st_num_layers"])
        embed_dim = int(params["st_embed_dim"])
        num_heads = int(params["st_num_heads"])
        mlp_width = int(params.get("st_mlp_width", 2 * embed_dim))
        mlp_depth = int(params.get("st_mlp_depth", 2))

        return bf.networks.SetTransformer(
            summary_dim=int(params["st_summary_dim"]),
            embed_dims=tuple([embed_dim] * num_layers),
            num_heads=tuple([num_heads] * num_layers),
            mlp_depths=tuple([mlp_depth] * num_layers),
            mlp_widths=tuple([mlp_width] * num_layers),
            dropout=float(params["st_dropout"]),
            num_inducing_points=(
                int(params["st_num_inducing"]) if "st_num_inducing" in params else None
            ),
        )
