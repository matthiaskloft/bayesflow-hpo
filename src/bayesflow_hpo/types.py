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
TrainFn = Callable[[Any, bf.simulators.Simulator, dict[str, Any], list[Any]], None]

# validate_fn: (approximator, validation_data, n_posterior_samples) -> metrics
#
# Contract
# --------
# The returned dict MUST contain all keys listed in ``objective_metrics``
# (by default ``["calibration_error", "nrmse"]``).  Missing or non-finite
# values are replaced with a penalty value and a warning is logged (see
# ``_validate_metric_keys``).  Extra keys are silently ignored — they are
# not passed to Optuna but may be stored as trial user attributes in a
# future version.
#
# Timing caveat
# -------------
# The wall-clock time of this function is used as the trial's inference
# time.  Unlike the default validation path, which isolates pure
# inference timing from metric computation, a custom ``validate_fn``
# lumps both together.  If metric computation is expensive relative to
# inference, the cost objective will overestimate inference time.
#
# Intermediate pruning
# --------------------
# When ``PeriodicValidationCallback`` is active (always, since
# ``validation_data`` is required), this function is also called during
# training at the configured interval with a reduced
# ``n_posterior_samples``.  The returned first-metric value is used for
# median-based multi-objective pruning.
ValidateFn = Callable[[Any, ValidationDataset, int], dict[str, float]]
