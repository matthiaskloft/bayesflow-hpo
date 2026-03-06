"""Search space for BayesFlow TimeSeriesNetwork."""

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
class TimeSeriesNetworkSpace(BaseSearchSpace):
    """Search space for `bf.networks.TimeSeriesNetwork`."""

    include_optional: bool = False

    summary_dim: IntDimension = field(
        default_factory=lambda: IntDimension("tsn_summary_dim", low=8, high=64, step=8)
    )
    recurrent_dim: IntDimension = field(
        default_factory=lambda: IntDimension(
            "tsn_recurrent_dim", low=32, high=256, step=32
        )
    )
    filters: IntDimension = field(
        default_factory=lambda: IntDimension("tsn_filters", low=16, high=128, step=16)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("tsn_dropout", low=0.0, high=0.3)
    )

    recurrent_type: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "tsn_recurrent_type", choices=["gru", "lstm"], default=False
        )
    )
    bidirectional: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "tsn_bidirectional", choices=[True, False], default=False
        )
    )
    skip_steps: IntDimension = field(
        default_factory=lambda: IntDimension(
            "tsn_skip_steps", low=1, high=8, default=False
        )
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.summary_dim,
            self.recurrent_dim,
            self.filters,
            self.dropout,
            self.recurrent_type,
            self.bidirectional,
            self.skip_steps,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.TimeSeriesNetwork:
        validate_required_params(
            params,
            ["tsn_summary_dim", "tsn_recurrent_dim", "tsn_filters", "tsn_dropout"],
            "TimeSeriesNetworkSpace.build",
        )

        return bf.networks.TimeSeriesNetwork(
            summary_dim=int(params["tsn_summary_dim"]),
            recurrent_dim=int(params["tsn_recurrent_dim"]),
            filters=int(params["tsn_filters"]),
            dropout=float(params["tsn_dropout"]),
            recurrent_type=params.get("tsn_recurrent_type", "gru"),
            bidirectional=bool(params.get("tsn_bidirectional", True)),
            skip_steps=int(params.get("tsn_skip_steps", 4)),
        )
