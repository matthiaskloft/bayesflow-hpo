"""Validation pipeline on fixed `ValidationDataset`."""

from __future__ import annotations

import gc
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn
from bayesflow_hpo.validation.metrics import aggregate_metrics, compute_batch_metrics


def _cleanup_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def run_validation_pipeline(
    approximator: Any,
    validation_data: ValidationDataset,
    n_posterior_samples: int = 1000,
    coverage_levels: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run SBC validation on a fixed dataset reused across trials."""
    if coverage_levels is None:
        coverage_levels = [0.5, 0.8, 0.9, 0.95, 0.99]

    infer_fn = make_bayesflow_infer_fn(
        approximator=approximator,
        param_keys=validation_data.param_keys,
        data_keys=validation_data.data_keys,
    )

    timing = {"inference": 0.0, "metrics": 0.0}
    sim_counter = 0
    all_batch_metrics: list[pd.DataFrame] = []

    for cond_id, sim_batch in enumerate(validation_data.simulations):
        t0 = time.time()
        draws = infer_fn(sim_batch, n_posterior_samples)
        timing["inference"] += time.time() - t0

        n_sims = len(np.asarray(sim_batch[validation_data.param_keys[0]]).reshape(-1))

        t1 = time.time()
        if len(validation_data.param_keys) == 1:
            true_values = np.asarray(sim_batch[validation_data.param_keys[0]]).reshape(-1)
            if draws.ndim == 3 and draws.shape[-1] == 1:
                draws = np.squeeze(draws, axis=-1)
            batch_metrics = compute_batch_metrics(
                draws=draws,
                true_values=true_values,
                cond_id=cond_id,
                sim_id_start=sim_counter,
                coverage_levels=list(coverage_levels),
            )
            all_batch_metrics.append(batch_metrics)
        else:
            if draws.ndim != 3:
                raise ValueError(
                    "Expected posterior draws with shape (n_sims, n_samples, n_params) "
                    "for multi-parameter inference."
                )

            for param_index, param_key in enumerate(validation_data.param_keys):
                true_values = np.asarray(sim_batch[param_key]).reshape(-1)
                param_draws = np.asarray(draws[:, :, param_index])
                param_metrics = compute_batch_metrics(
                    draws=param_draws,
                    true_values=true_values,
                    cond_id=cond_id,
                    sim_id_start=sim_counter,
                    coverage_levels=list(coverage_levels),
                )
                param_metrics.insert(0, "param_key", param_key)
                param_metrics.insert(1, "param_index", int(param_index))
                all_batch_metrics.append(param_metrics)

        timing["metrics"] += time.time() - t1
        sim_counter += n_sims
        _cleanup_gpu_memory()

    sim_metrics = pd.concat(all_batch_metrics, ignore_index=True)
    metrics = aggregate_metrics(
        sim_metrics=sim_metrics,
        coverage_levels=list(coverage_levels),
        n_posterior_samples=n_posterior_samples,
    )

    return {"metrics": metrics, "timing": timing}
