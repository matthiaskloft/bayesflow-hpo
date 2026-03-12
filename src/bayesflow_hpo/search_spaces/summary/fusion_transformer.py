"""Search space for BayesFlow FusionTransformer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    CategoricalDimension,
    FloatDimension,
    IntDimension,
)


@dataclass
class FusionTransformerSpace(BaseSearchSpace):
    """Search space for `bf.networks.FusionTransformer`.

    Default dimensions
    ------------------
    ft_summary_dim : int
        Output summary dimensionality (8--64, step 8).
    ft_embed_dim : int
        Embedding width (32--256, step 32).
    ft_num_heads : int
        Number of attention heads (1, 2, 4, or 8).
    ft_num_layers : int
        Number of transformer layers (1--4).
    ft_template_dim : int
        Template network width (32--256, step 32).
    ft_dropout : float
        Dropout rate (0.0--0.3).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    ft_template_type : str
        Template recurrent cell type (``"lstm"`` or ``"gru"``).
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

    template_type: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ft_template_type", choices=["lstm", "gru"], enabled=False
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.FusionTransformer:
        self._validate(params)

        num_layers = int(params["ft_num_layers"])
        embed_dim = int(params["ft_embed_dim"])
        num_heads = int(params["ft_num_heads"])

        kwargs: dict[str, Any] = {
            "summary_dim": int(params["ft_summary_dim"]),
            "embed_dims": tuple([embed_dim] * num_layers),
            "num_heads": tuple([num_heads] * num_layers),
            "template_dim": int(params["ft_template_dim"]),
            "dropout": float(params["ft_dropout"]),
        }
        if "ft_template_type" in params:
            kwargs["template_type"] = params["ft_template_type"]

        return bf.networks.FusionTransformer(**kwargs)
