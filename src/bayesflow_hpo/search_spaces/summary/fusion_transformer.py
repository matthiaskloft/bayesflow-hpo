"""Search space for BayesFlow FusionTransformer."""

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
class FusionTransformerSpace(BaseSearchSpace):
    """Search space for `bf.networks.FusionTransformer`."""

    include_optional: bool = False

    summary_dim: IntDimension = field(
        default_factory=lambda: IntDimension("ft_summary_dim", low=8, high=64, step=8)
    )
    embed_dim: IntDimension = field(
        default_factory=lambda: IntDimension("ft_embed_dim", low=32, high=256, step=32)
    )
    num_heads: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ft_num_heads", choices=[1, 2, 4, 8]
        )
    )
    num_layers: IntDimension = field(
        default_factory=lambda: IntDimension("ft_num_layers", low=1, high=4)
    )
    template_dim: IntDimension = field(
        default_factory=lambda: IntDimension(
            "ft_template_dim", low=32, high=256, step=32
        )
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("ft_dropout", low=0.0, high=0.3)
    )

    template_type: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ft_template_type", choices=["lstm", "gru"], default=False
        )
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.summary_dim,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.template_dim,
            self.dropout,
            self.template_type,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.FusionTransformer:
        self._validate(params)

        num_layers = int(params["ft_num_layers"])
        embed_dim = int(params["ft_embed_dim"])
        num_heads = int(params["ft_num_heads"])

        return bf.networks.FusionTransformer(
            summary_dim=int(params["ft_summary_dim"]),
            embed_dims=tuple([embed_dim] * num_layers),
            num_heads=tuple([num_heads] * num_layers),
            template_dim=int(params["ft_template_dim"]),
            dropout=float(params["ft_dropout"]),
            template_type=params.get("ft_template_type", "lstm"),
        )
