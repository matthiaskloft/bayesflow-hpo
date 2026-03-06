"""Validation dataset generation and serialization."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ValidationDataset:
    """Immutable fixed dataset reused across all HPO trials."""

    simulations: list[dict[str, np.ndarray]]
    condition_labels: list[dict[str, float | int]]
    param_keys: list[str]
    data_keys: list[str]
    seed: int


def generate_validation_dataset(
    simulator: Any,
    param_keys: list[str],
    data_keys: list[str],
    condition_grid: dict[str, list[Any]] | None = None,
    sims_per_condition: int = 200,
    seed: int = 42,
) -> ValidationDataset:
    """Generate a fixed validation dataset from simulator + condition grid."""
    rng = np.random.default_rng(seed)

    if condition_grid is None:
        batch_seed = int(rng.integers(0, 2**31))
        sims = simulator.sample(sims_per_condition, seed=batch_seed)
        return ValidationDataset(
            simulations=[sims],
            condition_labels=[{}],
            param_keys=param_keys,
            data_keys=data_keys,
            seed=seed,
        )

    keys = list(condition_grid.keys())
    values = [condition_grid[k] for k in keys]
    grid_points = list(itertools.product(*values))

    simulations: list[dict[str, np.ndarray]] = []
    condition_labels: list[dict[str, float | int]] = []

    for point in grid_points:
        condition = dict(zip(keys, point, strict=False))
        batch_seed = int(rng.integers(0, 2**31))
        sims = simulator.sample(
            sims_per_condition, conditions=condition, seed=batch_seed,
        )
        simulations.append(sims)
        condition_labels.append(condition)

    return ValidationDataset(
        simulations=simulations,
        condition_labels=condition_labels,
        param_keys=param_keys,
        data_keys=data_keys,
        seed=seed,
    )


def save_validation_dataset(dataset: ValidationDataset, path: str | Path) -> None:
    """Save dataset to a directory (`metadata.json` + `arrays.npz`)."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "condition_labels": dataset.condition_labels,
        "param_keys": dataset.param_keys,
        "data_keys": dataset.data_keys,
        "seed": dataset.seed,
        "n_batches": len(dataset.simulations),
    }
    meta_text = json.dumps(metadata, indent=2)
    (out_dir / "metadata.json").write_text(meta_text, encoding="utf-8")

    arrays: dict[str, np.ndarray] = {}
    for batch_idx, batch in enumerate(dataset.simulations):
        for key, value in batch.items():
            arrays[f"b{batch_idx}__{key}"] = np.asarray(value)

    np.savez_compressed(out_dir / "arrays.npz", **arrays)


def make_condition_grid(
    *,
    linspace: dict[str, tuple[float, float, int]] | None = None,
    logspace: dict[str, tuple[float, float, int]] | None = None,
    values: dict[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    """Build a condition grid from convenience specs.

    Parameters
    ----------
    linspace
        ``{name: (start, stop, n_points)}`` — linearly spaced values.
    logspace
        ``{name: (start, stop, n_points)}`` — log-spaced values
        (base 10, start/stop are **not** exponents — they are raw values).
    values
        ``{name: [v1, v2, ...]}`` — explicit value lists.

    Returns
    -------
    dict[str, list]
        Condition grid suitable for :func:`generate_validation_dataset`.
    """
    grid: dict[str, list[Any]] = {}
    if linspace:
        for name, (start, stop, n) in linspace.items():
            grid[name] = np.linspace(start, stop, n).tolist()
    if logspace:
        for name, (start, stop, n) in logspace.items():
            grid[name] = np.geomspace(start, stop, n).tolist()
    if values:
        for name, vals in values.items():
            grid[name] = list(vals)
    return grid


def make_validation_dataset(
    simulator: Any,
    param_keys: list[str],
    data_keys: list[str],
    *,
    linspace: dict[str, tuple[float, float, int]] | None = None,
    logspace: dict[str, tuple[float, float, int]] | None = None,
    values: dict[str, list[Any]] | None = None,
    sims_per_condition: int = 200,
    seed: int = 42,
) -> ValidationDataset:
    """One-step dataset creation combining grid construction + generation.

    Convenience wrapper that calls :func:`make_condition_grid` then
    :func:`generate_validation_dataset`.
    """
    grid = make_condition_grid(
        linspace=linspace, logspace=logspace, values=values,
    ) or None
    return generate_validation_dataset(
        simulator=simulator,
        param_keys=param_keys,
        data_keys=data_keys,
        condition_grid=grid,
        sims_per_condition=sims_per_condition,
        seed=seed,
    )


def load_validation_dataset(path: str | Path) -> ValidationDataset:
    """Load dataset from `save_validation_dataset` output directory."""
    in_dir = Path(path)
    metadata_path = in_dir / "metadata.json"
    arrays_path = in_dir / "arrays.npz"

    if not metadata_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(f"Validation dataset not found at {in_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path, allow_pickle=False)

    n_batches = int(metadata["n_batches"])
    simulations: list[dict[str, np.ndarray]] = []
    for batch_idx in range(n_batches):
        prefix = f"b{batch_idx}__"
        batch: dict[str, np.ndarray] = {}
        for key in arrays.files:
            if key.startswith(prefix):
                batch_name = key[len(prefix) :]
                batch[batch_name] = np.asarray(arrays[key])
        simulations.append(batch)

    return ValidationDataset(
        simulations=simulations,
        condition_labels=list(metadata["condition_labels"]),
        param_keys=list(metadata["param_keys"]),
        data_keys=list(metadata["data_keys"]),
        seed=int(metadata["seed"]),
    )
