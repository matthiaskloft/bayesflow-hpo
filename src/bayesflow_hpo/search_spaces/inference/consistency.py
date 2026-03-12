"""Search space for BayesFlow ConsistencyModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    FloatDimension,
    IntDimension,
)


def _compute_total_steps(params: dict[str, Any]) -> int:
    """Derive total training steps for the consistency model schedule.

    ConsistencyModel needs ``total_steps`` at construction time to set up
    its internal discretisation schedule.  This helper resolves the value
    from multiple possible sources in priority order:

    1. Explicit ``cm_total_steps`` (user override)
    2. Generic ``total_steps`` from training config
    3. ``epochs * batches_per_epoch`` (default: 200 * 50 = 10 000)
    """
    if "cm_total_steps" in params:
        return max(1, int(params["cm_total_steps"]))
    if "total_steps" in params:
        return max(1, int(params["total_steps"]))

    epochs = int(params.get("epochs", params.get("n_epochs", 200)))
    batches_per_epoch = int(params.get("batches_per_epoch", 50))
    return max(1, epochs * batches_per_epoch)


@dataclass
class ConsistencyModelSpace(BaseSearchSpace):
    """Search space for `bf.networks.ConsistencyModel`.

    Default dimensions
    ------------------
    cm_subnet_width : int
        MLP width (32--256, step 32).
    cm_subnet_depth : int
        MLP depth (1--4).
    cm_dropout : float
        Dropout rate (0.0--0.2).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    cm_max_time : int
        Maximum diffusion time (50--500).
    cm_sigma2 : float
        Noise variance (0.1--2.0).
    cm_s0 : int
        Initial schedule discretisation (2--30).
    cm_s1 : int
        Final schedule discretisation (20--100).
    """

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "cm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("cm_subnet_depth", low=1, high=4)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("cm_dropout", low=0.0, high=0.2)
    )

    max_time: IntDimension = field(
        default_factory=lambda: IntDimension(
            "cm_max_time", low=50, high=500, enabled=False
        )
    )
    sigma2: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "cm_sigma2", low=0.1, high=2.0, enabled=False
        )
    )
    s0: IntDimension = field(
        default_factory=lambda: IntDimension("cm_s0", low=2, high=30, enabled=False)
    )
    s1: IntDimension = field(
        default_factory=lambda: IntDimension("cm_s1", low=20, high=100, enabled=False)
    )

    def build(self, params: dict[str, Any]) -> bf.networks.ConsistencyModel:
        """Construct a ``bf.networks.ConsistencyModel`` from sampled parameters.

        ``total_steps`` is derived automatically from training config
        (see :func:`_compute_total_steps`) because the consistency model
        schedule depends on the total training budget.

        Parameters
        ----------
        params
            Hyperparameter dict from :meth:`sample`.  Should also contain
            ``epochs`` and ``batches_per_epoch`` for step computation.

        Returns
        -------
        bf.networks.ConsistencyModel
            Configured consistency model.
        """
        self._validate(params)

        width = int(params["cm_subnet_width"])
        depth = int(params["cm_subnet_depth"])
        total_steps = _compute_total_steps(params)

        kwargs: dict[str, Any] = {
            "total_steps": total_steps,
            "subnet_kwargs": {
                "widths": tuple([width] * depth),
                "dropout": float(params["cm_dropout"]),
            },
        }
        if "cm_max_time" in params:
            kwargs["max_time"] = float(params["cm_max_time"])
        if "cm_sigma2" in params:
            kwargs["sigma2"] = float(params["cm_sigma2"])
        if "cm_s0" in params:
            kwargs["s0"] = float(params["cm_s0"])
        if "cm_s1" in params:
            kwargs["s1"] = float(params["cm_s1"])

        return bf.networks.ConsistencyModel(**kwargs)
