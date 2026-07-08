"""Search space for BayesFlow FusionTransformer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    BoolDimension,
    CategoricalDimension,
    FloatDimension,
    IntDimension,
)


@dataclass
class FusionTransformerSpace(BaseSearchSpace):
    """Search space for `bf.networks.FusionTransformer`.

    Fixed dimensions (widen to tune)
    --------------------------------
    ft_mlp_width : int
        Feed-forward MLP width. Fixed at ``128``.
    ft_mlp_depth : int
        Feed-forward MLP depth. Fixed at ``2``.
    ft_bidirectional : bool
        Bidirectional template (LSTM/GRU). Fixed at ``True``.
    ft_template_type : str
        Template type. Fixed at ``"lstm"``.
    """

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
    mlp_width: IntDimension = field(
        default_factory=lambda: IntDimension("ft_mlp_width", constant=128)
    )
    mlp_depth: IntDimension = field(
        default_factory=lambda: IntDimension("ft_mlp_depth", constant=2)
    )
    bidirectional: BoolDimension = field(
        default_factory=lambda: BoolDimension(
            "ft_bidirectional", constant=True
        )
    )
    template_type: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ft_template_type", constant="lstm"
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.FusionTransformer:
        self._validate(params)

        num_layers = int(params["ft_num_layers"])
        embed_dim = int(params["ft_embed_dim"])
        num_heads = int(params["ft_num_heads"])
        mlp_width = int(params["ft_mlp_width"])
        mlp_depth = int(params["ft_mlp_depth"])

        return bf.networks.FusionTransformer(
            summary_dim=int(params["ft_summary_dim"]),
            embed_dims=tuple([embed_dim] * num_layers),
            num_heads=tuple([num_heads] * num_layers),
            mlp_widths=tuple([mlp_width] * num_layers),
            mlp_depths=tuple([mlp_depth] * num_layers),
            template_dim=int(params["ft_template_dim"]),
            dropout=float(params["ft_dropout"]),
            template_type=params["ft_template_type"],
            bidirectional=bool(params["ft_bidirectional"]),
        )
