"""Search space for BayesFlow DeepSet."""

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
class DeepSetSpace(BaseSearchSpace):
    """Search space for `bf.networks.DeepSet`."""

    include_optional: bool = False

    summary_dim: IntDimension = field(
        default_factory=lambda: IntDimension("ds_summary_dim", low=4, high=32)
    )
    depth: IntDimension = field(
        default_factory=lambda: IntDimension("ds_depth", low=1, high=4)
    )
    width: IntDimension = field(
        default_factory=lambda: IntDimension("ds_width", low=32, high=256, step=32)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("ds_dropout", low=0.0, high=0.3)
    )

    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ds_activation", choices=["silu", "mish"], default=False
        )
    )
    spectral_norm: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ds_spectral_norm", choices=[True, False], default=False
        )
    )
    pooling: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension("ds_pooling", choices=["mean"], default=False)
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.summary_dim,
            self.depth,
            self.width,
            self.dropout,
            self.activation,
            self.spectral_norm,
            self.pooling,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.DeepSet:
        validate_required_params(
            params,
            ["ds_summary_dim", "ds_depth", "ds_width", "ds_dropout"],
            "DeepSetSpace.build",
        )

        width = int(params["ds_width"])
        return bf.networks.DeepSet(
            summary_dim=int(params["ds_summary_dim"]),
            depth=int(params["ds_depth"]),
            mlp_widths_equivariant=(width,) * 2,
            mlp_widths_invariant_inner=(width,) * 2,
            mlp_widths_invariant_outer=(width,) * 2,
            mlp_widths_invariant_last=(width,) * 2,
            activation=params.get("ds_activation", "silu"),
            spectral_normalization=bool(params.get("ds_spectral_norm", False)),
            inner_pooling=params.get("ds_pooling", "mean"),
            output_pooling=params.get("ds_pooling", "mean"),
            dropout=float(params["ds_dropout"]),
        )
