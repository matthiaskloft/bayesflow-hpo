"""Search space for BayesFlow TimeSeriesNetwork."""

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
class TimeSeriesNetworkSpace(BaseSearchSpace):
    """Search space for `bf.networks.TimeSeriesNetwork`.

    Default dimensions (tuned)
    --------------------------
    tsn_summary_dim, tsn_recurrent_dim, tsn_filters, tsn_dropout.

    Fixed dimensions (widen to tune)
    --------------------------------
    tsn_recurrent_type : str
        Recurrent cell type. Fixed at ``"gru"``.
    tsn_bidirectional : bool
        Bidirectional recurrent layer. Fixed at ``True``.
    tsn_skip_steps : int
        Skip-connection stride. Fixed at ``4``.
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
            "tsn_recurrent_type", constant="gru"
        )
    )
    bidirectional: BoolDimension = field(
        default_factory=lambda: BoolDimension(
            "tsn_bidirectional", constant=True
        )
    )
    skip_steps: IntDimension = field(
        default_factory=lambda: IntDimension("tsn_skip_steps", constant=4)
    )

    def build(self, params: dict[str, Any]) -> bf.networks.TimeSeriesNetwork:
        """Construct a ``bf.networks.TimeSeriesNetwork`` from sampled parameters.

        Parameters
        ----------
        params
            Hyperparameter dict from :meth:`sample`.

        Returns
        -------
        bf.networks.TimeSeriesNetwork
            Configured time-series summary network.
        """
        self._validate(params)

        return bf.networks.TimeSeriesNetwork(
            summary_dim=int(params["tsn_summary_dim"]),
            recurrent_dim=int(params["tsn_recurrent_dim"]),
            filters=int(params["tsn_filters"]),
            dropout=float(params["tsn_dropout"]),
            recurrent_type=params["tsn_recurrent_type"],
            bidirectional=bool(params["tsn_bidirectional"]),
            skip_steps=int(params["tsn_skip_steps"]),
        )
