"""Search space for BayesFlow StableConsistencyModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    FloatDimension,
    IntDimension,
)


@dataclass
class StableConsistencyModelSpace(BaseSearchSpace):
    """Search space for `bf.networks.StableConsistencyModel`.

    Default dimensions
    ------------------
    scm_subnet_width : int
        MLP width (32--256, step 32).
    scm_subnet_depth : int
        MLP depth (1--4).
    scm_dropout : float
        Dropout rate (0.0--0.2).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    scm_sigma : float
        Noise scale (0.1--2.0).
    """

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "scm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("scm_subnet_depth", low=1, high=4)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("scm_dropout", low=0.0, high=0.2)
    )

    sigma: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "scm_sigma", low=0.1, high=2.0, enabled=False
        )
    )

    def build(self, params: dict[str, Any]) -> bf.networks.StableConsistencyModel:
        """Construct a ``bf.networks.StableConsistencyModel`` from sampled parameters.

        Parameters
        ----------
        params
            Hyperparameter dict from :meth:`sample`.

        Returns
        -------
        bf.networks.StableConsistencyModel
            Configured stable consistency model.
        """
        self._validate(params)

        width = int(params["scm_subnet_width"])
        depth = int(params["scm_subnet_depth"])
        kwargs: dict[str, Any] = {
            "subnet_kwargs": {
                "widths": tuple([width] * depth),
                "dropout": float(params["scm_dropout"]),
            },
        }
        if "scm_sigma" in params:
            kwargs["sigma"] = float(params["scm_sigma"])

        return bf.networks.StableConsistencyModel(**kwargs)
