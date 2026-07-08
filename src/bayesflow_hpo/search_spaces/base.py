"""Base search-space abstractions and default Optuna sampling.

This module defines the building blocks for hyperparameter search spaces:

- **Dimension dataclasses** (``IntDimension``, ``FloatDimension``,
  ``CategoricalDimension``, ``BoolDimension``) describe individual tunable
  knobs.
- **SearchSpace protocol** defines the three-method interface every
  network search space must satisfy: ``dimensions``, ``sample``, ``build``.
- **BaseSearchSpace** provides automatic ``dimensions`` discovery,
  ``sample`` dispatch, and validation from dataclass fields — concrete
  spaces only need to implement ``build``.

Design decision: dimensions are *declared as dataclass fields* rather than
returned from a method because this lets users override ranges by simply
passing new ``IntDimension(...)`` values at construction time, without
subclassing.

Dimensions can use ``constant=<value>`` to fix a parameter at a specific
value without going through Optuna's ``suggest_*`` machinery.  The
``constant`` and ``low``/``high`` (or ``choices``) fields are mutually
exclusive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Protocol

# Sentinel to distinguish "no constant set" from "constant is None".
_UNSET: Any = object()


@dataclass
class IntDimension:
    """Integer hyperparameter dimension.

    Parameters
    ----------
    name
        Optuna parameter name (must be unique within a search space).
    low, high
        Inclusive lower and upper bounds.  Required when ``constant``
        is not set.
    step
        Optional step size for discrete grids (e.g. ``step=32`` for
        widths).  When ``None``, any integer in [low, high] is valid.
    log
        Sample on a log scale (useful for learning rates or wide ranges).
    constant
        Fix this dimension to a specific value.  When set, the dimension
        is not sampled via Optuna — the value is injected directly.
        Mutually exclusive with ``low``/``high``.
    """

    name: str
    low: int | None = None
    high: int | None = None
    step: int | None = None
    log: bool = False
    constant: Any = field(default=_UNSET)

    def __post_init__(self) -> None:
        has_range = self.low is not None or self.high is not None
        has_constant = self.constant is not _UNSET
        if has_range and has_constant:
            raise ValueError(
                f"IntDimension({self.name!r}): cannot set both "
                f"constant and low/high."
            )
        if not has_range and not has_constant:
            raise ValueError(
                f"IntDimension({self.name!r}): must set either "
                f"constant or low/high."
            )
        if self.log and self.step is not None and self.step != 1:
            raise ValueError(
                f"IntDimension({self.name!r}): log=True is incompatible "
                f"with step={self.step}. Optuna requires step=1 (or None) "
                f"for log-scale integer sampling."
            )


@dataclass
class FloatDimension:
    """Float hyperparameter dimension.

    Parameters
    ----------
    name
        Optuna parameter name (must be unique within a search space).
    low, high
        Inclusive lower and upper bounds.  Required when ``constant``
        is not set.
    log
        Sample on a log scale (common for learning rates).
    constant
        Fix this dimension to a specific value.  Mutually exclusive
        with ``low``/``high``.
    """

    name: str
    low: float | None = None
    high: float | None = None
    log: bool = False
    constant: Any = field(default=_UNSET)

    def __post_init__(self) -> None:
        has_range = self.low is not None or self.high is not None
        has_constant = self.constant is not _UNSET
        if has_range and has_constant:
            raise ValueError(
                f"FloatDimension({self.name!r}): cannot set both "
                f"constant and low/high."
            )
        if not has_range and not has_constant:
            raise ValueError(
                f"FloatDimension({self.name!r}): must set either "
                f"constant or low/high."
            )


@dataclass
class CategoricalDimension:
    """Categorical hyperparameter dimension.

    Parameters
    ----------
    name
        Optuna parameter name (must be unique within a search space).
    choices
        Possible values.  Optuna picks uniformly among them.  Required
        when ``constant`` is not set.
    constant
        Fix this dimension to a specific value.  Mutually exclusive
        with ``choices``.
    """

    name: str
    choices: Sequence[str | int | float | bool | None] | None = None
    constant: Any = field(default=_UNSET)

    def __post_init__(self) -> None:
        has_choices = self.choices is not None
        has_constant = self.constant is not _UNSET
        if has_choices and has_constant:
            raise ValueError(
                f"CategoricalDimension({self.name!r}): cannot set both "
                f"constant and choices."
            )
        if not has_choices and not has_constant:
            raise ValueError(
                f"CategoricalDimension({self.name!r}): must set either "
                f"constant or choices."
            )


@dataclass
class BoolDimension:
    """Boolean hyperparameter dimension.

    Parameters
    ----------
    name
        Optuna parameter name (must be unique within a search space).
    constant
        Fix this dimension to a specific value.  When unset, the
        dimension is tunable over ``{True, False}``.
    """

    name: str
    constant: Any = field(default=_UNSET)


Dimension = IntDimension | FloatDimension | CategoricalDimension | BoolDimension

_DIMENSION_TYPES = (IntDimension, FloatDimension, CategoricalDimension, BoolDimension)


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
    :class:`IntDimension`, :class:`FloatDimension`,
    :class:`CategoricalDimension`, or :class:`BoolDimension`.  The
    ``dimensions`` property, ``sample`` method, and ``_validate`` helper
    are derived automatically — subclasses only need to implement
    ``build``.

    Dimensions with ``constant`` set are injected directly into the params
    dict without going through Optuna.  Use the ``.constants`` property to
    retrieve all fixed values.
    """

    @property
    def dimensions(self) -> list[Dimension]:
        """Collect all ``Dimension`` fields from this dataclass instance.

        Iterates over dataclass fields and returns those whose runtime
        value is an ``IntDimension``, ``FloatDimension``,
        ``CategoricalDimension``, or ``BoolDimension``.  This auto-discovery
        avoids requiring
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

    @property
    def constants(self) -> dict[str, Any]:
        """Return ``{name: value}`` for all fixed dimensions."""
        return {
            d.name: d.constant
            for d in self.dimensions
            if d.constant is not _UNSET
        }

    def _validate(self, params: dict[str, Any]) -> None:
        """Raise ``ValueError`` if any dimension key is missing."""
        required = [d.name for d in self.dimensions]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.build missing required parameters: "
                f"{', '.join(sorted(missing))}"
            )

    def sample(self, trial: Any) -> dict[str, Any]:
        """Sample hyperparameters from an Optuna trial.

        Dispatches each dimension to the appropriate
        ``trial.suggest_*`` method.  Dimensions with ``constant`` set
        are injected directly without calling Optuna.

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
            # Constants bypass Optuna entirely.
            if dim.constant is not _UNSET:
                params[dim.name] = dim.constant
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
            elif isinstance(dim, BoolDimension):
                params[dim.name] = trial.suggest_categorical(
                    dim.name, [True, False]
                )
            else:
                raise TypeError(f"Unsupported dimension type: {type(dim)!r}")

        return params
