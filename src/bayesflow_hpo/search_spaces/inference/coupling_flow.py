"""Search space for BayesFlow CouplingFlow."""

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
class CouplingFlowSpace(BaseSearchSpace):
    """Search space for `bf.networks.CouplingFlow`.

    Default dimensions
    ------------------
    cf_depth : int
        Number of coupling layers (2--8).
    cf_subnet_width : int
        MLP width per coupling subnet (32--256, step 32).
    cf_subnet_depth : int
        MLP depth per coupling subnet (1--3).
    cf_dropout : float
        Dropout rate (0.0--0.3).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    cf_activation : str
        Subnet activation function. Defaults to ``"mish"`` when omitted.
    cf_transform : str
        Coupling transform type (``"affine"`` or ``"spline"``).
    cf_permutation : str
        Permutation strategy (``"random"`` or ``"orthogonal"``).
    cf_actnorm : bool
        Whether to use activation normalization between coupling layers.
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
            "cf_activation", choices=["silu", "relu", "mish"], enabled=False
        )
    )

    transform: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_transform", choices=["affine", "spline"], enabled=False
        )
    )
    permutation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_permutation", choices=["random", "orthogonal"], enabled=False
        )
    )
    use_actnorm: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_actnorm", choices=[True, False], enabled=False
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.CouplingFlow:
        """Construct a ``bf.networks.CouplingFlow`` from sampled parameters.

        The subnet MLP uses a uniform-width architecture: all hidden
        layers share ``cf_subnet_width``.  Optional parameters (transform,
        permutation, actnorm) are only passed when present in *params*,
        allowing BayesFlow to apply its own defaults.

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

        # Uniform-width MLP: every hidden layer has the same width.
        subnet_kwargs: dict[str, Any] = {
            "widths": tuple([width] * n_layers),
            "dropout": float(params["cf_dropout"]),
        }
        if "cf_activation" in params:
            subnet_kwargs["activation"] = params["cf_activation"]

        kwargs: dict[str, Any] = {
            "depth": int(params["cf_depth"]),
            "subnet_kwargs": subnet_kwargs,
        }
        # Optional structural choices — omitted keys fall back to BayesFlow defaults.
        if "cf_transform" in params:
            kwargs["transform"] = params["cf_transform"]
        if "cf_permutation" in params:
            kwargs["permutation"] = params["cf_permutation"]
        if "cf_actnorm" in params:
            kwargs["use_actnorm"] = bool(params["cf_actnorm"])

        return bf.networks.CouplingFlow(**kwargs)
