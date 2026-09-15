"""Inference adapter from approximator to batch posterior samples.

Creates a closure that translates from the validation pipeline's
``(sim_data, n_samples)`` calling convention to BayesFlow's
``approximator.sample(conditions=..., num_samples=...)`` API.

For multi-parameter models, posterior draws are concatenated on the
last axis so that metrics receive a single ``(n_sims, n_samples, n_params)``
array and can index into individual parameters.

Sampling is chunked over the condition batch. A validation condition holds
``n_sims`` simulations and each is given ``n_posterior_samples`` draws, so an
unchunked call materializes their product at once -- 100,000 draws at the
``optimize()`` defaults, 200,000 at this pipeline's own -- plus every
intermediate the inference network needs for that many rows. An adaptive ODE
sampler keeps several integrator stages live, so the multiplier on the nominal
count is substantial: a FlowMatching + DeepSet approximator (15 parameters, 50
observations) ran 40,000 draws on an RTX 5090 and went out of memory at 60,000.
Neither ``n_sims`` nor ``n_posterior_samples`` is a search-space
hyperparameter, so ``estimate_peak_memory_mb`` -- which covers training only --
cannot reject such a trial, and the failure lands after training has been paid
for. See issue #101.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

#: Default cap on posterior draws materialized by a single ``sample()`` call.
#:
#: Chosen from the measurements in issue #101: 40,000 draws completed on a
#: 32 GiB card for a mid-sized approximator and 60,000 did not, so half the
#: last known-good figure leaves room for wider networks and for a card
#: shared with other work. The cost of a low cap is only more calls, each
#: with the same per-call overhead, while the cost of a high one is a dead
#: trial after full training.
DEFAULT_MAX_SAMPLES_PER_CALL = 20_000


def _slice(value: Any, start: int, count: int) -> Any:
    """Take *count* rows of *value* from *start*, leaving broadcasts whole.

    A 0-d value or one with a leading dimension of 1 is what the
    approximator broadcasts across the batch, so every chunk needs it in
    full; slicing it would hand later chunks an empty array.
    """
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.shape(value)
    shape = tuple(shape)
    if not shape or shape[0] == 1:
        return value
    return value[start : start + count]


def make_bayesflow_infer_fn(
    approximator: Any,
    param_keys: list[str],
    data_keys: list[str],
    available_keys: set[str] | None = None,
    max_samples_per_call: int | None = DEFAULT_MAX_SAMPLES_PER_CALL,
) -> Callable[[dict[str, Any], int], np.ndarray]:
    """Create inference fn: `(sim_data, n_samples) -> draws`.

    For multi-parameter inference, parameters are concatenated on the last axis
    before returning draws of shape `(n_sims, n_samples, n_params)`.

    Parameters
    ----------
    approximator
        Trained BayesFlow approximator.
    param_keys
        Names of parameter variables to extract from posterior draws.
    data_keys
        Names of observable/conditioning variables to pass to the approximator.
    available_keys
        If provided, validate that every entry in *data_keys* exists in this
        set.  Raises ``KeyError`` if any are missing.
    max_samples_per_call
        Maximum number of posterior draws (``rows x n_posterior_samples``)
        requested from the approximator in one ``sample()`` call.  The
        condition batch is split into consecutive slices of
        ``max_samples_per_call // n_posterior_samples`` rows -- at least one
        row, so a single simulation is never split and a sample count above
        the cap is honoured rather than silently reduced.  ``None`` disables
        chunking and restores the single-call behaviour.  The assembled
        return value is identical either way.

    Raises
    ------
    ValueError
        If *max_samples_per_call* is not positive.
    """
    if max_samples_per_call is not None and max_samples_per_call < 1:
        raise ValueError(
            "max_samples_per_call must be >= 1 or None (no chunking), got "
            f"{max_samples_per_call}."
        )

    if available_keys is not None:
        missing = set(data_keys) - available_keys
        if missing:
            raise KeyError(
                f"data_keys {sorted(missing)} not found in validation data. "
                f"Available keys: {sorted(available_keys)}"
            )

    def _assemble(post_draws: Any) -> np.ndarray:
        """Stack one chunk's draws into the metric contract's layout."""
        if len(param_keys) == 1:
            draws = np.asarray(post_draws[param_keys[0]])
            if draws.ndim == 3 and draws.shape[-1] == 1:
                draws = np.squeeze(draws, axis=-1)
            return draws

        draw_parts = [np.asarray(post_draws[key]) for key in param_keys]
        normalized_parts = [
            part[..., None] if part.ndim == 2 else part for part in draw_parts
        ]
        return np.concatenate(normalized_parts, axis=-1)

    def _batch_size(conditions: dict[str, Any]) -> int | None:
        """Rows every batched conditioning value shares, or ``None``.

        ``None`` means "do not chunk this batch": either nothing in it has a
        batch axis, or a shape could not be read at all. Both cases used to
        reach ``sample()`` untouched, and must keep doing so -- shape
        probing is new here and must not turn an input that previously
        worked into an exception.

        A value whose leading dimension is 1 while others are longer is
        treated as BROADCAST and excluded from the count, mirroring what
        the approximator does with it. Counting it would make the batch
        look one row long and silently switch chunking off, which is the
        failure mode the cap exists to prevent.

        Raises
        ------
        ValueError
            If two batched values disagree on their leading dimension.
            Slicing to the shorter of the two would drop the tail of the
            longer one, and the assembled draws would then have fewer rows
            than the condition has simulations -- a mismatch the metrics
            see as misaligned rows, not as an error.
        """
        sizes: dict[str, int] = {}
        for key, value in conditions.items():
            shape = getattr(value, "shape", None)
            if shape is None:
                try:
                    shape = np.shape(value)
                except Exception:  # noqa: BLE001 - probe only, never fatal
                    return None
            if not tuple(shape):
                # 0-d: no batch axis to slice along.
                continue
            sizes[key] = int(tuple(shape)[0])

        batched = {k: n for k, n in sizes.items() if n != 1}
        if not batched:
            return None
        distinct = set(batched.values())
        if len(distinct) > 1:
            raise ValueError(
                "Conditioning values disagree on their batch size: "
                + ", ".join(f"{k}={n}" for k, n in sorted(batched.items()))
                + ". Every conditioning array must have one row per "
                "simulation in the condition (values with a leading "
                "dimension of 1 are treated as broadcast)."
            )
        return distinct.pop()

    def infer_fn(sim_data: dict[str, Any], n_posterior_samples: int) -> np.ndarray:
        conditions = {k: sim_data[k] for k in data_keys}
        n_samples = int(n_posterior_samples)

        n_rows = _batch_size(conditions)
        if max_samples_per_call is None or n_rows is None:
            rows_per_call = n_rows = 0 if n_rows is None else n_rows
        else:
            rows_per_call = max(1, max_samples_per_call // max(n_samples, 1))

        if rows_per_call >= n_rows:
            # Unchunked, and deliberately not routed through the slicing
            # path: slicing would copy every conditioning array and the
            # concatenation would copy the result, for no benefit whenever
            # the whole batch already fits.
            return _assemble(
                approximator.sample(
                    conditions=conditions, num_samples=n_samples,
                )
            )

        chunks = [
            _assemble(
                approximator.sample(
                    conditions={
                        k: _slice(v, start, rows_per_call)
                        for k, v in conditions.items()
                    },
                    num_samples=n_samples,
                )
            )
            for start in range(0, n_rows, rows_per_call)
        ]
        return np.concatenate(chunks, axis=0)

    return infer_fn
