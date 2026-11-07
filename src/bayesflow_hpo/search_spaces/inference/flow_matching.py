"""Search space for BayesFlow FlowMatching."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import bayesflow as bf

from bayesflow_hpo.search_spaces.base import (
    BaseSearchSpace,
    BoolDimension,
    CategoricalDimension,
    FloatDimension,
    IntDimension,
    signature_default,
)


def _timemlp_default(param: str, fallback: Any) -> Any:
    """Read a default value used by ``FlowMatching``'s TimeMLP safely.

    ``TIME_MLP_DEFAULTS`` (BayesFlow >= 2.1) takes precedence over the
    ``TimeMLP`` signature because ``FlowMatching`` merges it into
    ``subnet_kwargs`` before building the subnet.
    """
    try:
        from bayesflow.networks.defaults import TIME_MLP_DEFAULTS

        if param in TIME_MLP_DEFAULTS:
            return TIME_MLP_DEFAULTS[param]
    except ImportError:
        pass
    try:
        value = inspect.signature(bf.networks.TimeMLP).parameters[param].default
    except (KeyError, ValueError, TypeError):
        return fallback
    return fallback if value is inspect._empty else value


def _flowmatching_integrate_default(param: str, fallback: Any) -> Any:
    """Read default integration settings from ``bf.networks.FlowMatching``."""
    try:
        from bayesflow.networks.defaults import FLOW_MATCHING_INTEGRATE_DEFAULTS
    except ImportError:
        return fallback
    return FLOW_MATCHING_INTEGRATE_DEFAULTS.get(param, fallback)


@dataclass
class FlowMatchingSpace(BaseSearchSpace):
    """Search space for `bf.networks.FlowMatching`.

    Default dimensions (tuned): subnet width/depth and dropout.

    Untuned dimensions are synchronized to BayesFlow defaults at runtime so
    they remain semantically neutral across BayesFlow updates:

    - TimeMLP-related constants are read from
      ``bayesflow.networks.defaults.TIME_MLP_DEFAULTS`` (the dict
      ``FlowMatching`` merges under ``subnet_kwargs`` in BayesFlow 2.1),
      then from ``inspect.signature(bf.networks.TimeMLP)``
    - Integration constants are read from
      ``bayesflow.networks.defaults.FLOW_MATCHING_INTEGRATE_DEFAULTS``

    If these runtime lookups are unavailable, safe fallback constants are
    used. Use :meth:`fast`, :meth:`balanced`, :meth:`quality`, or
    :meth:`preset` for speed/quality-oriented profiles.
    """

    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension(
            "fm_subnet_width", low=32, high=256, step=32
        )
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("fm_subnet_depth", low=1, high=6)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("fm_dropout", low=0.0, high=0.2)
    )
    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_activation",
            constant=_timemlp_default("activation", "mish"),
        )
    )
    use_optimal_transport: BoolDimension = field(
        default_factory=lambda: BoolDimension(
            "fm_use_optimal_transport", constant=False
        )
    )
    time_alpha: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "fm_time_power_law_alpha",
            constant=signature_default(
                bf.networks.FlowMatching, "time_power_law_alpha", -0.5
            ),
        )
    )
    time_embedding_dim: IntDimension = field(
        default_factory=lambda: IntDimension(
            "fm_time_embedding_dim",
            constant=int(_timemlp_default("time_embedding_dim", 32)),
        )
    )
    integrate_method: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_integrate_method",
            constant=_flowmatching_integrate_default("method", "tsit5"),
        )
    )
    integrate_steps: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_integrate_steps",
            constant=_flowmatching_integrate_default("steps", "adaptive"),
        )
    )
    merge: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_merge",
            constant=_timemlp_default("merge", "concat"),
        )
    )
    norm: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_norm",
            constant=_timemlp_default("norm", "layer"),
        )
    )
    residual: BoolDimension = field(
        default_factory=lambda: BoolDimension(
            "fm_residual",
            constant=bool(_timemlp_default("residual", True)),
        )
    )
    kernel_initializer: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "fm_kernel_initializer",
            constant=_timemlp_default("kernel_initializer", "he_normal"),
        )
    )

    @classmethod
    def preset(cls, profile: str = "default") -> FlowMatchingSpace:
        """Create a FlowMatching space with an ergonomic profile."""
        key = profile.strip().lower()
        if key == "default":
            return cls()
        if key == "fast":
            return cls.fast()
        if key == "balanced":
            return cls.balanced()
        if key == "quality":
            return cls.quality()
        raise ValueError(
            "Unknown FlowMatchingSpace profile "
            f"{profile!r}. Expected one of: default, fast, balanced, quality."
        )

    @classmethod
    def fast(cls) -> FlowMatchingSpace:
        """Latency-focused profile with lean subnet and fixed-step solvers."""
        return cls(
            subnet_width=IntDimension("fm_subnet_width", low=32, high=128, step=32),
            subnet_depth=IntDimension("fm_subnet_depth", low=1, high=3),
            dropout=FloatDimension("fm_dropout", constant=0.0),
            time_embedding_dim=IntDimension("fm_time_embedding_dim", constant=16),
            integrate_method=CategoricalDimension(
                "fm_integrate_method", choices=["euler", "tsit5"]
            ),
            integrate_steps=CategoricalDimension(
                "fm_integrate_steps", choices=[8, 16, 24]
            ),
            merge=CategoricalDimension("fm_merge", constant="add"),
            norm=CategoricalDimension("fm_norm", choices=[None, "layer"]),
            residual=BoolDimension("fm_residual", constant=False),
        )

    @classmethod
    def balanced(cls) -> FlowMatchingSpace:
        """Balanced profile for mixed speed/quality exploration."""
        return cls(
            subnet_width=IntDimension("fm_subnet_width", low=32, high=256, step=32),
            subnet_depth=IntDimension("fm_subnet_depth", low=1, high=5),
            dropout=FloatDimension("fm_dropout", low=0.0, high=0.1),
            integrate_method=CategoricalDimension(
                "fm_integrate_method", choices=["euler", "tsit5"]
            ),
            integrate_steps=CategoricalDimension(
                "fm_integrate_steps", choices=[16, 24, 32]
            ),
            merge=CategoricalDimension("fm_merge", choices=["add", "concat"]),
            norm=CategoricalDimension("fm_norm", choices=[None, "layer"]),
        )

    @classmethod
    def quality(cls) -> FlowMatchingSpace:
        """Quality-oriented profile with larger subnet and finer integration."""
        return cls(
            subnet_width=IntDimension("fm_subnet_width", low=96, high=320, step=32),
            subnet_depth=IntDimension("fm_subnet_depth", low=3, high=8),
            dropout=FloatDimension("fm_dropout", low=0.0, high=0.1),
            use_optimal_transport=BoolDimension("fm_use_optimal_transport"),
            time_embedding_dim=IntDimension(
                "fm_time_embedding_dim", low=32, high=96, step=32
            ),
            integrate_method=CategoricalDimension(
                "fm_integrate_method", constant="tsit5"
            ),
            integrate_steps=CategoricalDimension(
                "fm_integrate_steps", choices=[32, 48, 64]
            ),
        )

    def build(self, params: dict[str, Any]) -> bf.networks.FlowMatching:
        self._validate(params)

        width = int(params["fm_subnet_width"])
        depth = int(params["fm_subnet_depth"])

        subnet_kwargs: dict[str, Any] = {
            "widths": tuple([width] * depth),
            "dropout": float(params["fm_dropout"]),
            "activation": params["fm_activation"],
            "time_embedding_dim": int(params["fm_time_embedding_dim"]),
            "merge": params["fm_merge"],
            "norm": params["fm_norm"],
            "residual": bool(params["fm_residual"]),
            "kernel_initializer": params["fm_kernel_initializer"],
        }

        integrate_kwargs: dict[str, Any] = {
            "method": params["fm_integrate_method"],
            "steps": params["fm_integrate_steps"],
        }

        return bf.networks.FlowMatching(
            subnet_kwargs=subnet_kwargs,
            integrate_kwargs=integrate_kwargs,
            use_optimal_transport=bool(params["fm_use_optimal_transport"]),
            time_power_law_alpha=float(params["fm_time_power_law_alpha"]),
        )
