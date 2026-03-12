"""Dry-run validation to catch shape mismatches early.

Runs the full validation pipeline on a tiny slice (2 sims, 10 draws)
of the first condition.  This catches key mismatches, shape errors,
and missing data keys *before* starting the expensive HPO loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.pipeline import run_validation_pipeline
from bayesflow_hpo.validation.result import ValidationResult


def validate_once(
    approximator: Any,
    validation_data: ValidationDataset,
    n_sims: int = 2,
    n_posterior_samples: int = 10,
    metrics: Sequence[str] | None = None,
) -> ValidationResult:
    """Run a lightweight validation pass to verify data compatibility.

    Slices the first condition to *n_sims* rows, draws only
    *n_posterior_samples* samples, and runs the full metric pipeline.
    Any shape mismatch or key error surfaces immediately with a
    descriptive message.

    Parameters
    ----------
    approximator
        Trained BayesFlow approximator.
    validation_data
        The full :class:`ValidationDataset`.
    n_sims
        Number of simulations to use from the first condition.
    n_posterior_samples
        Number of posterior draws (keep small for speed).
    metrics
        Metric names to test (defaults to registry defaults).
    """
    import numpy as np

    if not validation_data.simulations:
        raise ValueError("ValidationDataset has no simulations.")

    first_batch = validation_data.simulations[0]

    # Slice to n_sims rows
    sliced: dict[str, np.ndarray] = {}
    for key, arr in first_batch.items():
        arr = np.asarray(arr)
        sliced[key] = arr[:n_sims] if arr.shape[0] >= n_sims else arr

    mini_dataset = ValidationDataset(
        simulations=[sliced],
        condition_labels=validation_data.condition_labels[:1],
        param_keys=validation_data.param_keys,
        data_keys=validation_data.data_keys,
        seed=validation_data.seed,
    )

    try:
        return run_validation_pipeline(
            approximator=approximator,
            validation_data=mini_dataset,
            n_posterior_samples=n_posterior_samples,
            metrics=metrics,
        )
    except Exception as exc:
        raise RuntimeError(
            "Dry-run validation failed. Check that "
            f"param_keys={validation_data.param_keys} and "
            f"data_keys={validation_data.data_keys} match your "
            f"simulator output and approximator. Error: {exc}"
        ) from exc
