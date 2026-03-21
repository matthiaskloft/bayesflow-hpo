"""Search space for BayesFlow DeepSet."""

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
class DeepSetSpace(BaseSearchSpace):
    """Search space for `bf.networks.DeepSet`.

    Default dimensions (tuned)
    --------------------------
    ds_summary_dim : int
        Output summary dimensionality (4--64, step 4).
    ds_depth : int
        Number of DeepSet processing blocks (1--4).
    ds_width : int
        MLP width for all sub-MLPs (32--256, step 32).
    ds_dropout : float
        Dropout rate (0.0--0.3).

    Fixed dimensions (widen to tune)
    --------------------------------
    ds_activation : str
        Activation function. Fixed at ``"silu"``.
    ds_spectral_normalization : bool
        Spectral normalization. Fixed at ``False``.
    ds_inner_pooling : str
        Inner pooling strategy. Fixed at ``"mean"``.
    ds_output_pooling : str
        Output pooling strategy. Fixed at ``"mean"``.

    Architecture notes
    ------------------
    The ``invariant_outer`` MLP uses ``(width, summary_dim)`` to act as a
    bottleneck, matching BayesFlow's default architecture.  All other MLPs
    use ``(width, width)``.
    """

    summary_dim: IntDimension = field(
        default_factory=lambda: IntDimension("ds_summary_dim", low=4, high=64, step=4)
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
            "ds_activation", constant="silu"
        )
    )
    spectral_norm: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ds_spectral_normalization", constant=False
        )
    )
    inner_pooling: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ds_inner_pooling", constant="mean"
        )
    )
    output_pooling: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ds_output_pooling", constant="mean"
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.DeepSet:
        """Construct a ``bf.networks.DeepSet`` from sampled parameters.

        Parameters
        ----------
        params
            Hyperparameter dict from :meth:`sample`.

        Returns
        -------
        bf.networks.DeepSet
            Configured DeepSet summary network.
        """
        self._validate(params)

        width = int(params["ds_width"])
        summary_dim = int(params["ds_summary_dim"])

        return bf.networks.DeepSet(
            summary_dim=summary_dim,
            depth=int(params["ds_depth"]),
            mlp_widths_equivariant=(width, width),
            mlp_widths_invariant_inner=(width, width),
            mlp_widths_invariant_outer=(width, summary_dim),
            mlp_widths_invariant_last=(width, width),
            dropout=float(params["ds_dropout"]),
            activation=params["ds_activation"],
            spectral_normalization=bool(params["ds_spectral_normalization"]),
            inner_pooling=params["ds_inner_pooling"],
            output_pooling=params["ds_output_pooling"],
        )
