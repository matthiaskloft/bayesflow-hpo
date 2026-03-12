"""Search space for BayesFlow TimeSeriesNetwork."""

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
class TimeSeriesNetworkSpace(BaseSearchSpace):
    """Search space for `bf.networks.TimeSeriesNetwork`.

    Default dimensions
    ------------------
    tsn_summary_dim : int
        Output summary dimensionality (8--64, step 8).
    tsn_recurrent_dim : int
        Recurrent hidden size (32--256, step 32).
    tsn_filters : int
        Convolutional filter count (16--128, step 16).
    tsn_dropout : float
        Dropout rate (0.0--0.3).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    tsn_recurrent_type : str
        Recurrent cell type (``"gru"`` or ``"lstm"``).
    tsn_bidirectional : bool
        Whether to use a bidirectional recurrent layer.
    tsn_skip_steps : int
        Skip-connection stride (1--8).
    """

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
            "tsn_recurrent_type", choices=["gru", "lstm"], enabled=False
        )
    )
    bidirectional: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "tsn_bidirectional", choices=[True, False], enabled=False
        )
    )
    skip_steps: IntDimension = field(
        default_factory=lambda: IntDimension(
            "tsn_skip_steps", low=1, high=8, enabled=False
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.TimeSeriesNetwork:
        self._validate(params)

        return bf.networks.TimeSeriesNetwork(
            summary_dim=int(params["tsn_summary_dim"]),
            recurrent_dim=int(params["tsn_recurrent_dim"]),
            filters=int(params["tsn_filters"]),
            dropout=float(params["tsn_dropout"]),
            recurrent_type=params.get("tsn_recurrent_type", "gru"),
            bidirectional=bool(params.get("tsn_bidirectional", True)),
            skip_steps=int(params.get("tsn_skip_steps", 4)),
        )
