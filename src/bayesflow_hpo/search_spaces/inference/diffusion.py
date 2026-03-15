"""Search space for BayesFlow DiffusionModel."""

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
class DiffusionModelSpace(BaseSearchSpace):
    """Search space for `bf.networks.DiffusionModel`.

    Default dimensions
    ------------------
    dm_subnet_width : int
        MLP width (32--256, step 32).
    dm_subnet_depth : int
        MLP depth (1--6).  BayesFlow default TimeMLP uses 5 layers.
    dm_dropout : float
        Dropout rate (0.0--0.2).
    dm_activation : str
        Subnet activation function (``"mish"`` or ``"silu"``).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    dm_noise_schedule : str
        Noise schedule type (``"edm"`` or ``"cosine"``).
    dm_prediction_type : str
        Prediction target (``"F"``, ``"velocity"``, ``"noise"``, ``"x"``).
    """

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "dm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("dm_subnet_depth", low=1, high=6)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("dm_dropout", low=0.0, high=0.2)
    )
    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "dm_activation", choices=["mish", "silu"]
        )
    )

    noise_schedule: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "dm_noise_schedule", choices=["edm", "cosine"], enabled=False
        )
    )
    prediction_type: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "dm_prediction_type", choices=["F", "velocity", "noise", "x"], enabled=False
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.DiffusionModel:
        """Construct a ``bf.networks.DiffusionModel`` from sampled parameters.

        Parameters
        ----------
        params
            Hyperparameter dict from :meth:`sample`.

        Returns
        -------
        bf.networks.DiffusionModel
            Configured diffusion model.
        """
        self._validate(params)

        width = int(params["dm_subnet_width"])
        depth = int(params["dm_subnet_depth"])

        kwargs: dict[str, Any] = {
            "subnet_kwargs": {
                "widths": tuple([width] * depth),
                "activation": params["dm_activation"],
                "dropout": float(params["dm_dropout"]),
            },
        }
        if "dm_noise_schedule" in params:
            kwargs["noise_schedule"] = params["dm_noise_schedule"]
        if "dm_prediction_type" in params:
            kwargs["prediction_type"] = params["dm_prediction_type"]

        return bf.networks.DiffusionModel(**kwargs)
