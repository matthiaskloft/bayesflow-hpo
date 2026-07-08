"""Search space for BayesFlow CouplingFlow."""

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
class CouplingFlowSpace(BaseSearchSpace):
    """Search space for `bf.networks.CouplingFlow`.

    Default dimensions (tuned)
    --------------------------
    cf_depth : int
        Number of coupling layers (2--8).
    cf_subnet_width : int
        MLP width per coupling subnet (32--256, step 32).
    cf_subnet_depth : int
        MLP depth per coupling subnet (1--3).
    cf_dropout : float
        Dropout rate (0.0--0.3).

    Fixed dimensions (widen to tune)
    --------------------------------
    cf_activation : str
        Subnet activation. Fixed at BayesFlow default (not exposed by BF
        as a top-level kwarg; uses subnet default).
    cf_transform : str
        Coupling transform type. Fixed at ``"affine"``.
    cf_permutation : str
        Permutation strategy. Fixed at ``"random"``.
    cf_use_actnorm : bool
        Activation normalization. Fixed at ``True``.
    """

    depth: IntDimension = field(
        default_factory=lambda: IntDimension("cf_depth", low=2, high=8)
    )
    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "cf_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("cf_subnet_depth", low=1, high=3)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("cf_dropout", low=0.0, high=0.3)
    )
    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_activation", constant="silu"
        )
    )
    transform: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_transform", constant="affine"
        )
    )
    permutation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_permutation", constant="random"
        )
    )
    use_actnorm: BoolDimension = field(
        default_factory=lambda: BoolDimension(
            "cf_use_actnorm", constant=True
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.CouplingFlow:
        """Construct a ``bf.networks.CouplingFlow`` from sampled parameters.

        Parameters
        ----------
        params
            Hyperparameter dict from :meth:`sample`.

        Returns
        -------
        bf.networks.CouplingFlow
            Configured coupling flow network.
        """
        self._validate(params)

        width = int(params["cf_subnet_width"])
        n_layers = int(params["cf_subnet_depth"])

        subnet_kwargs: dict[str, Any] = {
            "widths": tuple([width] * n_layers),
            "dropout": float(params["cf_dropout"]),
            "activation": params["cf_activation"],
        }

        return bf.networks.CouplingFlow(
            depth=int(params["cf_depth"]),
            subnet_kwargs=subnet_kwargs,
            transform=params["cf_transform"],
            permutation=params["cf_permutation"],
            use_actnorm=bool(params["cf_use_actnorm"]),
        )
