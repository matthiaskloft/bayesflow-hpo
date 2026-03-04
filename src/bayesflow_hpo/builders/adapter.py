"""Generic BayesFlow adapter construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import bayesflow as bf
import numpy as np
from bayesflow.adapters.transforms.transform import Transform
from bayesflow.utils.serialization import serializable, serialize


@serializable("bayesflow_hpo.builders")
class PriorStandardize(Transform):
    """Standardize one inference variable by per-sample prior location/scale."""

    def __init__(
        self,
        param_key: str,
        scale_key: str,
        loc_key: str | None = None,
    ):
        super().__init__()
        self.param_key = param_key
        self.scale_key = scale_key
        self.loc_key = loc_key
        self._cached_scales: np.ndarray | None = None
        self._cached_locs: np.ndarray | None = None

    def get_config(self) -> dict:
        return serialize(
            {
                "param_key": self.param_key,
                "scale_key": self.scale_key,
                "loc_key": self.loc_key,
            }
        )

    def extra_repr(self) -> str:
        parts = f"param={self.param_key!r}, scale={self.scale_key!r}"
        if self.loc_key is not None:
            parts += f", loc={self.loc_key!r}"
        return parts

    @staticmethod
    def _broadcast_to(arr: np.ndarray, target: np.ndarray) -> np.ndarray:
        while arr.ndim < target.ndim:
            arr = arr[..., np.newaxis]
        return arr

    def _resolve(
        self,
        data: dict[str, np.ndarray],
        key: str | None,
        cache: np.ndarray | None,
        default: float | None,
    ) -> np.ndarray | None:
        if key is not None and key in data:
            return np.asarray(data[key])
        if cache is not None:
            return cache
        if default is None:
            return None
        return np.asarray(default)

    def forward(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        del kwargs
        data = data.copy()

        if self.scale_key in data:
            self._cached_scales = np.array(data[self.scale_key], copy=True)

        if self.loc_key is not None and self.loc_key in data:
            self._cached_locs = np.array(data[self.loc_key], copy=True)

        if self.param_key in data and self._cached_scales is not None:
            param = data[self.param_key]
            scale = self._broadcast_to(self._cached_scales.copy(), param)
            loc = self._resolve(data, self.loc_key, self._cached_locs, 0.0)
            assert loc is not None
            loc = self._broadcast_to(loc, param)
            data[self.param_key] = (param - loc) / scale

        return data

    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        del kwargs
        data = data.copy()

        if self.param_key not in data:
            return data

        scale = self._resolve(data, self.scale_key, self._cached_scales, None)
        if scale is None:
            return data

        param = data[self.param_key]
        scale = self._broadcast_to(scale, param)
        loc = self._resolve(data, self.loc_key, self._cached_locs, 0.0)
        assert loc is not None
        loc = self._broadcast_to(loc, param)

        data[self.param_key] = param * scale + loc
        return data


@dataclass
class AdapterSpec:
    """Declarative specification for constructing a BayesFlow adapter."""

    set_keys: list[str]
    param_keys: list[str]
    context_keys: list[str]
    standardize_keys: list[str] = field(default_factory=list)
    prior_standardize: dict[str, tuple[str | None, str]] = field(default_factory=dict)
    broadcast_specs: dict[str, str] = field(default_factory=dict)
    context_transforms: dict[str, tuple[Callable, Callable]] = field(
        default_factory=dict
    )
    output_dtype: str = "float32"


def create_adapter(spec: AdapterSpec) -> bf.Adapter:
    """Build a `bf.Adapter` from an `AdapterSpec`."""
    adapter = bf.Adapter()

    for param_key, (loc_key, scale_key) in spec.prior_standardize.items():
        adapter.transforms.append(
            PriorStandardize(param_key, scale_key=scale_key, loc_key=loc_key)
        )

    for ctx_key, target_key in spec.broadcast_specs.items():
        adapter = adapter.broadcast(ctx_key, to=target_key)

    if spec.standardize_keys:
        adapter = adapter.standardize(spec.standardize_keys, mean=0, std=1)

    adapter = adapter.as_set(spec.set_keys)

    for key, (forward_fn, _inverse_fn) in spec.context_transforms.items():
        adapter = adapter.apply(include=key, forward=forward_fn)

    adapter = adapter.convert_dtype(from_dtype="float64", to_dtype=spec.output_dtype)

    if len(spec.param_keys) == 1:
        adapter = adapter.rename(from_key=spec.param_keys[0], to_key="inference_variables")
    else:
        adapter = adapter.concatenate(spec.param_keys, into="inference_variables", axis=-1)

    adapter = adapter.concatenate(spec.set_keys, into="summary_variables", axis=-1)
    adapter = adapter.concatenate(spec.context_keys, into="inference_conditions", axis=-1)
    return adapter