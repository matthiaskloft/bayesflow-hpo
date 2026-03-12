"""Base search-space abstractions and default Optuna sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol


@dataclass
class IntDimension:
    """Integer hyperparameter dimension."""

    name: str
    low: int
    high: int
    step: int | None = None
    log: bool = False
    enabled: bool = True


@dataclass
class FloatDimension:
    """Float hyperparameter dimension."""

    name: str
    low: float
    high: float
    log: bool = False
    enabled: bool = True


@dataclass
class CategoricalDimension:
    """Categorical hyperparameter dimension."""

    name: str
    choices: Sequence[str | int | float | bool | None]
    enabled: bool = True


Dimension = IntDimension | FloatDimension | CategoricalDimension

_DIMENSION_TYPES = (IntDimension, FloatDimension, CategoricalDimension)


class SearchSpace(Protocol):
    """Protocol for network-specific search spaces."""

    @property
    def dimensions(self) -> list[Dimension]:
        """Return all tunable dimensions for this search space."""

    def sample(self, trial: Any) -> dict[str, Any]:
        """Sample hyperparameters from an Optuna trial."""

    def build(self, params: dict[str, Any]) -> Any:
        """Build the corresponding network from sampled params."""


@dataclass
class BaseSearchSpace:
    """Base class with automatic ``dimensions``, ``sample``, and validation.

    Subclasses declare hyperparameters as dataclass fields of type
    :class:`IntDimension`, :class:`FloatDimension`, or
    :class:`CategoricalDimension`.  The ``dimensions`` property, ``sample``
    method, and ``_validate`` helper are derived automatically — subclasses
    only need to implement ``build``.

    Dimensions with ``default=False`` are *optional*: they are only sampled
    when ``include_optional=True``.
    """

    include_optional: bool = False

    @property
    def dimensions(self) -> list[Dimension]:
        cls = type(self)
        if cls is not BaseSearchSpace:
            # Detect subclasses that forgot the @dataclass decorator.
            # Their dimension annotations won't appear in fields().
            own_annotations = cls.__dict__.get("__annotations__", {})
            field_names = {f.name for f in fields(self)}
            missing = [
                name
                for name in own_annotations
                if name not in field_names
                and isinstance(getattr(cls, name, None), _DIMENSION_TYPES)
            ]
            if missing:
                raise TypeError(
                    f"{cls.__name__} must be decorated with @dataclass "
                    f"to use BaseSearchSpace's automatic dimension discovery."
                )
        all_fields = fields(self)
        return [
            getattr(self, f.name)
            for f in all_fields
            if isinstance(getattr(self, f.name), _DIMENSION_TYPES)
        ]

    def _validate(self, params: dict[str, Any]) -> None:
        """Raise ``ValueError`` if any required dimension key is missing."""
        required = [d.name for d in self.dimensions if d.default]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.build missing required parameters: "
                f"{', '.join(sorted(missing))}"
            )

    def sample(self, trial: Any) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for dim in self.dimensions:
            if not dim.enabled and not self.include_optional:
                continue

            if isinstance(dim, IntDimension):
                kwargs: dict[str, Any] = {"log": dim.log}
                if dim.step is not None:
                    kwargs["step"] = dim.step
                params[dim.name] = trial.suggest_int(
                    dim.name, dim.low, dim.high, **kwargs
                )
            elif isinstance(dim, FloatDimension):
                params[dim.name] = trial.suggest_float(
                    dim.name,
                    dim.low,
                    dim.high,
                    log=dim.log,
                )
            elif isinstance(dim, CategoricalDimension):
                params[dim.name] = trial.suggest_categorical(
                    dim.name, list(dim.choices)
                )
            else:
                raise TypeError(f"Unsupported dimension type: {type(dim)!r}")

        return params
