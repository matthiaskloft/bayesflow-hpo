"""Base search-space abstractions and default Optuna sampling.

This module defines the building blocks for hyperparameter search spaces:

- **Dimension dataclasses** (``IntDimension``, ``FloatDimension``,
  ``CategoricalDimension``) describe individual tunable knobs.
- **SearchSpace protocol** defines the three-method interface every
  network search space must satisfy: ``dimensions``, ``sample``, ``build``.
- **BaseSearchSpace** provides automatic ``dimensions`` discovery,
  ``sample`` dispatch, and validation from dataclass fields — concrete
  spaces only need to implement ``build``.

Design decision: dimensions are *declared as dataclass fields* rather than
returned from a method because this lets users override ranges by simply
passing new ``IntDimension(...)`` values at construction time, without
subclassing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol


@dataclass
class IntDimension:
    """Integer hyperparameter dimension.

    Parameters
    ----------
    name
        Optuna parameter name (must be unique within a search space).
    low, high
        Inclusive lower and upper bounds.
    step
        Optional step size for discrete grids (e.g. ``step=32`` for
        widths).  When ``None``, any integer in [low, high] is valid.
    log
        Sample on a log scale (useful for learning rates or wide ranges).
    enabled
        When ``False``, this dimension is only sampled if the parent
        space has ``include_optional=True``.
    """

    name: str
    low: int
    high: int
    step: int | None = None
    log: bool = False
    enabled: bool = True


@dataclass
class FloatDimension:
    """Float hyperparameter dimension.

    Parameters
    ----------
    name
        Optuna parameter name (must be unique within a search space).
    low, high
        Inclusive lower and upper bounds.
    log
        Sample on a log scale (common for learning rates).
    enabled
        When ``False``, only sampled if ``include_optional=True``.
    """

    name: str
    low: float
    high: float
    log: bool = False
    enabled: bool = True


@dataclass
class CategoricalDimension:
    """Categorical hyperparameter dimension.

    Parameters
    ----------
    name
        Optuna parameter name (must be unique within a search space).
    choices
        Possible values.  Optuna picks uniformly among them.
    enabled
        When ``False``, only sampled if ``include_optional=True``.
    """

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

    Dimensions with ``enabled=False`` are *optional*: they are only sampled
    when ``include_optional=True``.
    """

    include_optional: bool = False

    @property
    def dimensions(self) -> list[Dimension]:
        """Collect all ``Dimension`` fields from this dataclass instance.

        Iterates over dataclass fields and returns those whose runtime
        value is an ``IntDimension``, ``FloatDimension``, or
        ``CategoricalDimension``.  This auto-discovery avoids requiring
        subclasses to manually list their dimensions.
        """
        try:
            all_fields = fields(self)
        except TypeError:
            raise TypeError(
                f"{type(self).__name__} must be decorated with @dataclass "
                f"to use BaseSearchSpace's automatic dimension discovery."
            ) from None
        return [
            getattr(self, f.name)
            for f in all_fields
            if isinstance(getattr(self, f.name), _DIMENSION_TYPES)
        ]

    def _validate(self, params: dict[str, Any]) -> None:
        """Raise ``ValueError`` if any required dimension key is missing."""
        required = [d.name for d in self.dimensions if d.enabled]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.build missing required parameters: "
                f"{', '.join(sorted(missing))}"
            )

    def sample(self, trial: Any) -> dict[str, Any]:
        """Sample hyperparameters from an Optuna trial.

        Dispatches each dimension to the appropriate
        ``trial.suggest_*`` method.  Disabled dimensions are skipped
        unless ``self.include_optional`` is ``True``.

        Parameters
        ----------
        trial
            An ``optuna.Trial`` instance.

        Returns
        -------
        dict[str, Any]
            Mapping from dimension name to sampled value.
        """
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
