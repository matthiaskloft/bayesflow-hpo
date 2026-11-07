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

    Default dimensions (tuned)
    --------------------------
    tst_summary_dim, tst_embed_dim, tst_num_heads, tst_num_layers, tst_dropout.

    Fixed dimensions (widen to tune)
    --------------------------------
    tst_expansion_factor : float
        Width multiplier of the gated feed-forward block (BayesFlow >= 2.1
        replaced ``mlp_width``/``mlp_depth`` with this). Fixed at ``4.0``.
    tst_time_embedding : str
        Time embedding type. Fixed at ``"time2vec"``.
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
    expansion_factor: FloatDimension = field(
        default_factory=lambda: FloatDimension("tst_expansion_factor", constant=4.0)
    )
    time_embed: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "tst_time_embedding", constant="time2vec"
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.TimeSeriesTransformer:
        """Construct a ``bf.networks.TimeSeriesTransformer`` from sampled parameters.

        Parameters
        ----------
        params
            Hyperparameter dict from :meth:`sample`.

        Returns
        -------
        bf.networks.TimeSeriesTransformer
            Configured time-series transformer summary network.
        """
        self._validate(params)

        num_layers = int(params["tst_num_layers"])
        embed_dim = int(params["tst_embed_dim"])
        num_heads = int(params["tst_num_heads"])

        return bf.networks.TimeSeriesTransformer(
            summary_dim=int(params["tst_summary_dim"]),
            embed_dims=tuple([embed_dim] * num_layers),
            num_heads=tuple([num_heads] * num_layers),
            dropout=float(params["tst_dropout"]),
            expansion_factor=float(params["tst_expansion_factor"]),
            time_embedding=params["tst_time_embedding"],
        )
