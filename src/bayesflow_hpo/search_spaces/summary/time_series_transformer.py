"""Search space for BayesFlow TimeSeriesTransformer."""

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
class TimeSeriesTransformerSpace(BaseSearchSpace):
    """Search space for `bf.networks.TimeSeriesTransformer`.

    Default dimensions
    ------------------
    tst_summary_dim : int
        Output summary dimensionality (8--64, step 8).
    tst_embed_dim : int
        Embedding width (32--256, step 32).
    tst_num_heads : int
        Number of attention heads (1, 2, 4, or 8).
    tst_num_layers : int
        Number of transformer layers (1--4).
    tst_dropout : float
        Dropout rate (0.0--0.3).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    tst_mlp_width : int
        Feed-forward MLP width (64--512, step 64). Defaults to
        ``2 * embed_dim``.
    tst_time_embed : str
        Time embedding type (``"time2vec"`` or ``"sinusoidal"``).
    """

    summary_dim: IntDimension = field(
        default_factory=lambda: IntDimension("tst_summary_dim", low=8, high=64, step=8)
    )
    embed_dim: IntDimension = field(
        default_factory=lambda: IntDimension("tst_embed_dim", low=32, high=256, step=32)
    )
    num_heads: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "tst_num_heads", choices=[1, 2, 4, 8]
        )
    )
    num_layers: IntDimension = field(
        default_factory=lambda: IntDimension("tst_num_layers", low=1, high=4)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("tst_dropout", low=0.0, high=0.3)
    )

    mlp_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "tst_mlp_width", low=64, high=512, step=64, enabled=False
        )
    )
    time_embed: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "tst_time_embed", choices=["time2vec", "sinusoidal"], enabled=False
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.TimeSeriesTransformer:
        self._validate(params)

        num_layers = int(params["tst_num_layers"])
        embed_dim = int(params["tst_embed_dim"])
        num_heads = int(params["tst_num_heads"])
        mlp_width = int(params.get("tst_mlp_width", 2 * embed_dim))

        return bf.networks.TimeSeriesTransformer(
            summary_dim=int(params["tst_summary_dim"]),
            embed_dims=tuple([embed_dim] * num_layers),
            num_heads=tuple([num_heads] * num_layers),
            mlp_widths=tuple([mlp_width] * num_layers),
            dropout=float(params["tst_dropout"]),
            time_embedding=params.get("tst_time_embed", "time2vec"),
        )
