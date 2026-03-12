"""Search space for BayesFlow DeepSet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    CategoricalDimension,
    Dimension,
    FloatDimension,
    IntDimension,
    validate_required_params,
)


@dataclass
class DeepSetSpace(BaseSearchSpace):
    """Search space for `bf.networks.DeepSet`.

    Default ranges
    --------------
    ds_summary_dim : int
        Output summary dimensionality (4--64).
    ds_depth : int
        Number of DeepSet processing blocks (1--4).
    ds_width : int
        MLP width for all sub-MLPs (32--256, step 32).
    ds_dropout : float
        Dropout rate (0.0--0.3).

    Optional dimensions (enabled via ``include_optional=True``)
    -----------------------------------------------------------
    ds_activation : str
        Falls back to BayesFlow default ``"silu"``.
    ds_spectral_norm : bool
        Falls back to ``False``.
    ds_inner_pooling, ds_output_pooling : str
        Falls back to ``"mean"``. Choices: ``["mean", "max"]``.

    Architecture notes
    ------------------
    The ``invariant_outer`` MLP uses ``(width, summary_dim)`` to act as a
    bottleneck, matching BayesFlow's default architecture.  All other MLPs
    use ``(width, width)``.
    """

    include_optional: bool = False

    summary_dim: IntDimension = field(
        default_factory=lambda: IntDimension("ds_summary_dim", low=4, high=64)
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
            "ds_activation", choices=["silu", "mish"], enabled=False
        )
    )
    spectral_norm: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ds_spectral_norm", choices=[True, False], enabled=False
        )
    )
    inner_pooling: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ds_inner_pooling", choices=["mean", "max"], enabled=False
        )
    )
    output_pooling: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "ds_output_pooling", choices=["mean", "max"], enabled=False
        )
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.summary_dim,
            self.depth,
            self.width,
            self.dropout,
            self.activation,
            self.spectral_norm,
            self.inner_pooling,
            self.output_pooling,
        ]

    def sample(self, trial: Any) -> dict[str, Any]:
        return BaseSearchSpace.sample(self, trial)

    def build(self, params: dict[str, Any]) -> bf.networks.DeepSet:
        validate_required_params(
            params,
            ["ds_summary_dim", "ds_depth", "ds_width", "ds_dropout"],
            "DeepSetSpace.build",
        )

        width = int(params["ds_width"])
        summary_dim = int(params["ds_summary_dim"])
        return bf.networks.DeepSet(
            summary_dim=summary_dim,
            depth=int(params["ds_depth"]),
            mlp_widths_equivariant=(width, width),
            mlp_widths_invariant_inner=(width, width),
            mlp_widths_invariant_outer=(width, summary_dim),
            mlp_widths_invariant_last=(width, width),
            activation=params.get("ds_activation", "silu"),
            spectral_normalization=bool(params.get("ds_spectral_norm", False)),
            inner_pooling=params.get("ds_inner_pooling", "mean"),
            output_pooling=params.get("ds_output_pooling", "mean"),
            dropout=float(params["ds_dropout"]),
        )
