"""Type aliases for custom approximator hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bayesflow_hpo.validation.data import ValidationDataset

# build_approximator_fn: (hparams) -> approximator
BuildApproximatorFn = Callable[[dict[str, Any]], Any]

# train_fn: (approximator, simulator, hparams, callbacks) -> None
TrainFn = Callable[[Any, Any, dict[str, Any], list], None]

# validate_fn: (approximator, validation_data, n_posterior_samples) -> metrics
ValidateFn = Callable[[Any, ValidationDataset, int], dict[str, float]]
