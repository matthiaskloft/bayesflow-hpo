"""Base search-space abstractions and default Optuna sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class IntDimension:
    """Integer hyperparameter dimension."""

    name: str
    low: int
    high: int
    step: int | None = None
    log: bool = False
    default: bool = True


@dataclass
class FloatDimension:
    """Float hyperparameter dimension."""

    name: str
    low: float
    high: float
    log: bool = False
    default: bool = True


@dataclass
class CategoricalDimension:
    """Categorical hyperparameter dimension."""

    name: str
    choices: Sequence[str | int | float | bool | None]
    default: bool = True


Dimension = IntDimension | FloatDimension | CategoricalDimension


class SearchSpace(Protocol):
    """Protocol for network-specific search spaces."""

    @property
    def dimensions(self) -> list[Dimension]:
        """Return all tunable dimensions for this search space."""

    def sample(self, trial: Any) -> dict[str, Any]:
        """Sample hyperparameters from an Optuna trial."""

    def build(self, params: dict[str, Any]) -> Any:
        """Build the corresponding network from sampled params."""


class BaseSearchSpace:
    """Base class with automatic `trial.suggest_*` dispatch."""

    def __init__(self, include_optional: bool = False):
        self.include_optional = include_optional

    @property
    def dimensions(self) -> list[Dimension]:
        raise NotImplementedError

    def sample(self, trial: Any) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for dim in self.dimensions:
            if not dim.default and not self.include_optional:
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


def validate_required_params(
    params: dict[str, Any],
    required_keys: list[str],
    context: str,
) -> None:
    """Validate that all required parameter keys are present."""
    missing = [key for key in required_keys if key not in params]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing required parameters for {context}: {missing_display}"
        )
