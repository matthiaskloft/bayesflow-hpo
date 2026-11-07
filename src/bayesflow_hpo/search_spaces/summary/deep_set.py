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
        Hidden width of the MLP inside every equivariant layer (32--256,
        step 32); the MLP has two hidden layers of this width.
    ds_dropout : float
        Dropout rate (0.0--0.3).

    Fixed dimensions (widen to tune)
    --------------------------------
    ds_activation : str
        Activation function. Fixed at ``"silu"``.
    ds_embed_dim : int
        Working width shared by all equivariant layers. Fixed at ``64``
        (BayesFlow default).
    ds_inner_pooling : str
        Pooling used inside the equivariant layers. Fixed at ``"mean"``.

    Architecture notes
    ------------------
    BayesFlow 2.1 rewrote ``DeepSet`` (stacked equivariant layers, then
    pooling by multi-head attention and a linear projection to
    ``summary_dim``). The 2.0.x arguments ``mlp_widths_equivariant``,
    ``mlp_widths_invariant_*``, ``output_pooling`` and
    ``spectral_normalization`` no longer exist there and are silently
    dropped, so ``ds_output_pooling`` and ``ds_spectral_normalization`` were
    removed from this space (bayesflow-hpo 0.8.0). ``ds_width`` now sets the
    equivariant-layer MLP ``mlp_widths=(width, width)``
    (``bayesflow/networks/summary/deep_set/deep_set.py``, verified against
    bayesflow 2.1.0, 2026-11-07).
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
    embed_dim: IntDimension = field(
        default_factory=lambda: IntDimension("ds_embed_dim", constant=64)
    )
    inner_pooling: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ds_inner_pooling", constant="mean"
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
            embed_dim=int(params["ds_embed_dim"]),
            mlp_widths=(width, width),
            dropout=float(params["ds_dropout"]),
            activation=params["ds_activation"],
            inner_pooling=params["ds_inner_pooling"],
        )
