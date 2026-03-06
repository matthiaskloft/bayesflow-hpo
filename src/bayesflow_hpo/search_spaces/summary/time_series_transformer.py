"""Search space for BayesFlow TimeSeriesTransformer."""

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
class TimeSeriesTransformerSpace(BaseSearchSpace):
    """Search space for `bf.networks.TimeSeriesTransformer`."""

    include_optional: bool = False

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
            "tst_mlp_width", low=64, high=512, step=64, default=False
        )
    )
    time_embed: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "tst_time_embed", choices=["time2vec", "sinusoidal"], default=False
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
            self.time_embed,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.TimeSeriesTransformer:
        validate_required_params(
            params,
            [
                "tst_summary_dim",
                "tst_embed_dim",
                "tst_num_heads",
                "tst_num_layers",
                "tst_dropout",
            ],
            "TimeSeriesTransformerSpace.build",
        )

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
