"""Search space for BayesFlow SetTransformer."""

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
class SetTransformerSpace(BaseSearchSpace):
    """Search space for `bf.networks.SetTransformer`.

    Default dimensions
    ------------------
    st_summary_dim : int
        Output summary dimensionality (8--64, step 8).
    st_embed_dim : int
        Embedding width (32--256, step 32).
    st_num_heads : int
        Number of attention heads (1, 2, 4, or 8).
    st_num_layers : int
        Number of transformer layers (1--4).
    st_dropout : float
        Dropout rate (0.0--0.3).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    st_mlp_width : int
        Feed-forward MLP width (64--512, step 64).
    st_mlp_depth : int
        Feed-forward MLP depth (1--4).
    st_num_inducing : int
        Number of inducing points for ISAB (8--64, step 8).
    """

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

    def build(self, params: dict[str, Any]) -> bf.networks.SetTransformer:
        self._validate(params)

        num_layers = int(params["st_num_layers"])
        embed_dim = int(params["st_embed_dim"])
        num_heads = int(params["st_num_heads"])

        kwargs: dict[str, Any] = {
            "summary_dim": int(params["st_summary_dim"]),
            "embed_dims": tuple([embed_dim] * num_layers),
            "num_heads": tuple([num_heads] * num_layers),
            "dropout": float(params["st_dropout"]),
        }
        if "st_mlp_width" in params:
            mlp_width = int(params["st_mlp_width"])
            kwargs["mlp_widths"] = tuple(
                [mlp_width] * num_layers
            )
        if "st_mlp_depth" in params:
            mlp_depth = int(params["st_mlp_depth"])
            kwargs["mlp_depths"] = tuple(
                [mlp_depth] * num_layers
            )
        if "st_num_inducing" in params:
            kwargs["num_inducing_points"] = int(
                params["st_num_inducing"]
            )

        return bf.networks.SetTransformer(**kwargs)
