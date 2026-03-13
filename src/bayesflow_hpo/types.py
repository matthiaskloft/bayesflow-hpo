"""Type aliases for custom approximator hooks.

These use ``Any`` for the approximator type because custom hooks may
return non-standard approximator subclasses (e.g.
``EquivariantIRTApproximator``).  The simulator type uses the concrete
BayesFlow type since all simulators share the same interface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import bayesflow as bf

from bayesflow_hpo.validation.data import ValidationDataset

# build_approximator_fn: (hparams) -> approximator
# Returns Any because custom hooks may produce non-standard approximator types.
BuildApproximatorFn = Callable[[dict[str, Any]], Any]

# train_fn: (approximator, simulator, hparams, callbacks) -> None
TrainFn = Callable[[Any, bf.simulators.Simulator, dict[str, Any], list], None]

# validate_fn: (approximator, validation_data, n_posterior_samples) -> metrics
ValidateFn = Callable[[Any, ValidationDataset, int], dict[str, float]]
