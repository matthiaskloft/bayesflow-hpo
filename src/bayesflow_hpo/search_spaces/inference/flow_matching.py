"""Search space for BayesFlow FlowMatching."""

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
class FlowMatchingSpace(BaseSearchSpace):
    """Search space for `bf.networks.FlowMatching`.

    Default dimensions
    ------------------
    fm_subnet_width : int
        MLP width (32--256, step 32).
    fm_subnet_depth : int
        MLP depth (1--4).
    fm_dropout : float
        Dropout rate (0.0--0.2).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    fm_activation : str
        Subnet activation function. Defaults to ``"mish"`` when omitted.
    fm_use_ot : bool
        Whether to use optimal transport for improved training stability.
        Increases training time (~2.5x) but may speed up inference.
    fm_time_alpha : float
        Power-law exponent for the time distribution during training.
        Controls sampling bias: ``p(t) ∝ t^(1/(1+α))``. Default
        ``α=0`` corresponds to uniform sampling.
    fm_time_embedding_dim : int
        Dimensionality of the Fourier time embedding in the subnet
        (8--64, step 4). BayesFlow defaults to 32 when omitted.
    """

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "fm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("fm_subnet_depth", low=1, high=4)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("fm_dropout", low=0.0, high=0.2)
    )
    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_activation", choices=["mish", "silu"], enabled=False
        )
    )

    use_optimal_transport: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_use_ot", choices=[True, False], enabled=False
        )
    )
    time_alpha: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "fm_time_alpha", low=0.0, high=2.0, enabled=False
        )
    )
    time_embedding_dim: IntDimension = field(
        default_factory=lambda: IntDimension(
            "fm_time_embedding_dim", low=8, high=64, step=4, enabled=False
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.FlowMatching:
        """Construct a ``bf.networks.FlowMatching`` from sampled parameters.

        Parameters
        ----------
        params
            Hyperparameter dict from :meth:`sample`.

        Returns
        -------
        bf.networks.FlowMatching
            Configured flow matching network.
        """
        self._validate(params)

        width = int(params["fm_subnet_width"])
        depth = int(params["fm_subnet_depth"])

        subnet_kwargs: dict[str, Any] = {
            "widths": tuple([width] * depth),
            "dropout": float(params["fm_dropout"]),
        }
        if "fm_activation" in params:
            subnet_kwargs["activation"] = params["fm_activation"]
        if "fm_time_embedding_dim" in params:
            subnet_kwargs["time_embedding_dim"] = int(
                params["fm_time_embedding_dim"]
            )

        kwargs: dict[str, Any] = {
            "subnet_kwargs": subnet_kwargs,
        }
        if "fm_use_ot" in params:
            kwargs["use_optimal_transport"] = bool(params["fm_use_ot"])
        if "fm_time_alpha" in params:
            kwargs["time_power_law_alpha"] = float(
                params["fm_time_alpha"]
            )

        return bf.networks.FlowMatching(**kwargs)
