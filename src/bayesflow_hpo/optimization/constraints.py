"""Constraint and budget helpers for trial pre-filtering.

Provides fast heuristic estimates of parameter count and peak memory
usage from hyperparameter dicts *before* building the actual model.
This allows the objective to reject obviously-oversized configs without
allocating GPU memory.

The heuristics are intentionally conservative (tend to overestimate)
because a false positive (rejecting a viable config) is much cheaper
than a false negative (OOM crash mid-training).
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypeAlias

logger = logging.getLogger(__name__)

# (metric_name, threshold, "above"/"below")
# "above" => reject when metric value exceeds threshold
# "below" => reject when metric value falls below threshold
MetricConstraintSpec: TypeAlias = tuple[str, float, Literal["above", "below"]]


def _safe_int(value: Any, default: int) -> int:
    """Convert *value* to int, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    """Convert *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mlp_block_params(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
) -> int:
    """Estimate trainable parameter count for an MLP block.

    Assumes a standard dense MLP: input → [hidden × depth] → output,
    counting weights + biases for each layer.
    """
    if depth <= 0:
        return max(1, input_dim * output_dim + output_dim)

    # Weights + biases for each layer.
    first = input_dim * hidden_dim + hidden_dim
    middle = max(0, depth - 1) * (hidden_dim * hidden_dim + hidden_dim)
    last = hidden_dim * output_dim + output_dim
    return max(1, first + middle + last)


def _estimate_summary_params(params: dict[str, Any]) -> tuple[int, int]:
    """Estimate (param_count, summary_dim) for the summary network.

    Dispatches based on which parameter prefix is present in *params*.
    Falls back to DeepSet estimation if no transformer keys are found.

    Returns
    -------
    tuple[int, int]
        ``(estimated_param_count, summary_dim)`` — the summary_dim is
        needed downstream to estimate the inference network input size.
    """
    if "ft_summary_dim" in params:
        summary_dim = _safe_int(params.get("ft_summary_dim"), 16)
        embed_dim = _safe_int(params.get("ft_embed_dim"), 64)
        layers = _safe_int(params.get("ft_num_layers"), 2)
        template_dim = _safe_int(params.get("ft_template_dim"), 128)
        total = layers * (embed_dim * embed_dim + 2 * embed_dim * template_dim)
        return max(1, total), summary_dim

    if "tst_summary_dim" in params:
        summary_dim = _safe_int(params.get("tst_summary_dim"), 16)
        embed_dim = _safe_int(params.get("tst_embed_dim"), 64)
        layers = _safe_int(params.get("tst_num_layers"), 2)
        heads = _safe_int(params.get("tst_num_heads"), 4)
        mlp_width = _safe_int(params.get("tst_mlp_width"), 2 * embed_dim)
        total = layers * (
            embed_dim * embed_dim + embed_dim * mlp_width + heads * embed_dim
        )
        return max(1, total), summary_dim

    if "tsn_summary_dim" in params:
        summary_dim = _safe_int(params.get("tsn_summary_dim"), 16)
        filters = _safe_int(params.get("tsn_filters"), 32)
        recurrent_dim = _safe_int(params.get("tsn_recurrent_dim"), 128)
        total = (filters * 3 * 3) + (4 * recurrent_dim * recurrent_dim)
        return max(1, total), summary_dim

    if "st_summary_dim" in params:
        summary_dim = _safe_int(params.get("st_summary_dim"), 16)
        embed_dim = _safe_int(params.get("st_embed_dim"), 64)
        layers = _safe_int(params.get("st_num_layers"), 2)
        heads = _safe_int(params.get("st_num_heads"), 4)
        mlp_width = _safe_int(params.get("st_mlp_width"), 2 * embed_dim)
        mlp_depth = _safe_int(params.get("st_mlp_depth"), 2)
        total = layers * (
            embed_dim * embed_dim
            + heads * embed_dim
            + _mlp_block_params(embed_dim, mlp_width, mlp_depth, embed_dim)
        )
        return max(1, total), summary_dim

    summary_dim = _safe_int(params.get("ds_summary_dim"), 8)
    deepset_width = _safe_int(params.get("ds_width"), 64)
    deepset_depth = _safe_int(params.get("ds_depth"), 2)
    # Inner MLP: obs_dim → width (depth hidden layers)
    # + outer MLP: width → summary_dim (depth hidden layers).
    # Use obs_dim=1 as conservative lower bound for the input embedding.
    obs_dim = 1
    inner = _mlp_block_params(obs_dim, deepset_width, deepset_depth, deepset_width)
    outer = _mlp_block_params(deepset_width, deepset_width, deepset_depth, summary_dim)
    return max(1, inner + outer), summary_dim


def _estimate_inference_params(params: dict[str, Any], summary_dim: int) -> int:
    """Estimate trainable parameter count for the inference network.

    The input dimension is ``summary_dim + n_conditions + latent_dim/2``
    because BayesFlow splits the latent space for coupling transforms.
    The output dimension is ``2 * ceil(n_params/2)`` to account for
    affine transform outputs (scale + shift).

    For CouplingFlow, the estimate accounts for BayesFlow's internal
    latent-dimension padding (``max(n_params, 2 * depth)``).
    """
    n_conditions = _safe_int(params.get("n_conditions"), 4)
    n_params = _safe_int(params.get("n_params"), 1)
    input_dim = summary_dim + n_conditions + max(1, n_params // 2)
    output_dim = 2 * max(1, (n_params + 1) // 2)

    if "cf_depth" in params:
        depth = _safe_int(params.get("cf_depth"), 6)
        hidden = _safe_int(params.get("cf_subnet_width"), 128)
        subnet_depth = _safe_int(params.get("cf_subnet_depth"), 2)
        # BayesFlow pads the latent dimension to at least 2 * depth.
        latent_dim = max(n_params, 2 * depth)
        cf_input = summary_dim + n_conditions + latent_dim // 2
        cf_output = 2 * ((latent_dim + 1) // 2)
        subnet = _mlp_block_params(cf_input, hidden, subnet_depth, cf_output)
        return max(1, depth * subnet)

    if "fm_subnet_width" in params:
        hidden = _safe_int(params.get("fm_subnet_width"), 128)
        subnet_depth = _safe_int(params.get("fm_subnet_depth"), 2)
        return _mlp_block_params(input_dim, hidden, subnet_depth, output_dim)

    if "dm_subnet_width" in params:
        hidden = _safe_int(params.get("dm_subnet_width"), 128)
        subnet_depth = _safe_int(params.get("dm_subnet_depth"), 2)
        return _mlp_block_params(input_dim, hidden, subnet_depth, output_dim)

    if "cm_subnet_width" in params:
        hidden = _safe_int(params.get("cm_subnet_width"), 128)
        subnet_depth = _safe_int(params.get("cm_subnet_depth"), 2)
        return _mlp_block_params(input_dim, hidden, subnet_depth, output_dim)

    if "scm_subnet_width" in params:
        hidden = _safe_int(params.get("scm_subnet_width"), 128)
        subnet_depth = _safe_int(params.get("scm_subnet_depth"), 2)
        return _mlp_block_params(input_dim, hidden, subnet_depth, output_dim)

    generic_width = _safe_int(
        params.get("subnet_width", params.get("hidden_dim", params.get("width"))),
        128,
    )
    generic_depth = _safe_int(
        params.get("subnet_depth", params.get("hidden_depth", params.get("depth"))),
        2,
    )
    return _mlp_block_params(input_dim, generic_width, generic_depth, output_dim)


def estimate_param_count(params: dict[str, Any]) -> int:
    """Heuristic parameter estimate from supported search-space keys.

    Combines summary and inference network estimates.  This is a fast
    pre-check that avoids building the actual Keras model — the exact
    count is verified after build in the objective function.
    """
    summary_params, summary_dim = _estimate_summary_params(params)
    inference_params = _estimate_inference_params(params, summary_dim)
    return max(1, int(summary_params + inference_params))


def _subnet_width(params: dict[str, Any]) -> int:
    """Width of the widest subnet named in *params*, or 128."""
    return _safe_int(
        params.get(
            "cf_subnet_width",
            params.get(
                "fm_subnet_width",
                params.get(
                    "dm_subnet_width",
                    params.get(
                        "cm_subnet_width",
                        params.get(
                            "scm_subnet_width",
                            params.get("hidden_dim", params.get("width", 128)),
                        ),
                    ),
                ),
            ),
        ),
        128,
    )


def _subnet_depth(params: dict[str, Any]) -> int:
    """Depth of the subnet named in *params*, or 2."""
    return _safe_int(
        params.get(
            "cf_subnet_depth",
            params.get(
                "fm_subnet_depth",
                params.get(
                    "dm_subnet_depth",
                    params.get(
                        "cm_subnet_depth",
                        params.get("scm_subnet_depth", params.get("depth", 2)),
                    ),
                ),
            ),
        ),
        2,
    )


#: Live state buffers an adaptive ODE sampler holds at once.
#:
#: BayesFlow's default flow-matching integrator is ``tsit5``
#: (``bayesflow/networks/defaults.py``: ``FLOW_MATCHING_INTEGRATE_DEFAULTS =
#: {"method": "tsit5", "steps": "adaptive"}``), and ``tsit5_step``
#: (``bayesflow/utils/integrate.py``) evaluates seven stages ``k1..k7`` that
#: are all alive simultaneously, on top of ``state``, ``new_state`` and the
#: error estimate. Seven is therefore a floor on the multiplier applied to
#: one state-sized tensor, not a guess -- and it is what made the 500x1000
#: case of issue #101 fail inside ``integrate_adaptive -> tsit5_step`` on a
#: ``(500000, 256)`` activation with ~20 GiB free.
_ODE_SAMPLER_STAGES = 7

#: Multiplier reconciling the per-row activation proxy with measurement.
#:
#: Calibrated against the #101 benchmark rather than assumed. That harness
#: (`docs/plans/bench_inference_ratio.py`) sampled a FlowMatching network
#: with subnet widths ``(128, 128)`` and a DeepSet with ``summary_dim=32,
#: depth=2``: the proxy below scores ``(32 + 128) * 2 = 320`` elements per
#: row, which at four bytes and seven live stages is 8.75 KiB per row. The
#: run completed 40,000 rows and went out of memory at 60,000 under a
#: 6.29 GiB process cap, so the true peak is above 107 KiB per row -- a
#: factor of roughly 12.6, which this rounds up. The gap is everything the
#: proxy does not name: per-layer activations inside each subnet call, the
#: time and residual embeddings, the adaptive integrator's error estimate,
#: the returned sample buffer, and allocator fragmentation.
#:
#: It is a floor on the overhead of ONE measured configuration, not a law.
#: Deliberately so: this module rejects rather than crashes, and a rejected
#: viable config costs one trial where an OOM costs a whole training run.
_SAMPLING_OVERHEAD_FACTOR = 13

#: Search-space prefixes whose inference network is sampled by integrating
#: an ODE, and so pays :data:`_ODE_SAMPLER_STAGES`. A coupling flow inverts
#: layer by layer instead, keeping one intermediate live at a time.
_ODE_SAMPLED_PREFIXES = ("fm_", "dm_", "cm_", "scm_")


def estimate_validation_memory_mb(
    params: dict[str, Any],
    n_sims: int,
    n_posterior_samples: int,
    max_samples_per_call: int | None = None,
    dtype_bytes: int = 4,
) -> float:
    """Estimate approximate peak memory of ONE validation ``sample()`` call.

    A different allocation from :func:`estimate_peak_memory_mb`, which
    covers training: sampling holds no gradients and no optimizer state,
    but its batch is ``n_sims x n_posterior_samples`` rows rather than
    ``batch_size``, and an adaptive ODE sampler keeps
    :data:`_ODE_SAMPLER_STAGES` copies of that batch live at once. At the
    ``optimize()`` defaults the sampling batch is 100,000 rows against a
    training batch of a few hundred, which is why a trial could pass the
    training budget and still die in validation (issue #101).

    *max_samples_per_call* is the same cap
    :func:`~bayesflow_hpo.validation.inference.make_bayesflow_infer_fn`
    applies, and is what the estimate is taken over: the peak is one
    CHUNK, not the whole condition. Passing ``None`` estimates the
    unchunked call.

    The heuristic keeps this module's convention of overestimating: a
    rejected viable config costs one trial, an OOM costs a full training
    run. Its per-row proxy is calibrated by
    :data:`_SAMPLING_OVERHEAD_FACTOR` against the measurement in #101.

    Known to be approximate in two directions: the summary network's own
    activations scale with the observation count, which is not a
    search-space key and is not modelled here, and the calibration comes
    from one architecture on one card.

    Parameters
    ----------
    params
        Hyperparameter dict from the search space.
    n_sims
        Simulations in the largest validation condition.
    n_posterior_samples
        Posterior draws requested per simulation.
    max_samples_per_call
        Cap on draws per ``sample()`` call, or ``None`` for no chunking.
    dtype_bytes
        Bytes per element (default 4 for float32).

    Returns
    -------
    float
        Estimated peak megabytes for one sampling call.
    """
    summary_params, summary_dim = _estimate_summary_params(params)
    inference_params = _estimate_inference_params(params, summary_dim)
    total_params = max(1, summary_params + inference_params)

    n_sims = max(1, _safe_int(n_sims, 1))
    n_samples = max(1, _safe_int(n_posterior_samples, 1))
    rows = n_sims * n_samples
    if max_samples_per_call is not None:
        cap = max(1, _safe_int(max_samples_per_call, rows))
        # Mirrors the closure: whole rows, at least one simulation.
        rows = min(rows, max(1, cap // n_samples) * n_samples)

    width = _subnet_width(params)
    depth = max(1, _subnet_depth(params) * max(1, _safe_int(params.get("cf_depth"), 1)))
    stages = (
        _ODE_SAMPLER_STAGES
        if any(key.startswith(_ODE_SAMPLED_PREFIXES) for key in params)
        else 1
    )

    activation_elements = max(1, rows * max(1, summary_dim + width) * depth)
    # Weights only -- no gradients, no optimizer state at sampling time.
    param_bytes = total_params * dtype_bytes
    activation_bytes = (
        activation_elements * dtype_bytes * stages * _SAMPLING_OVERHEAD_FACTOR
    )

    return float((param_bytes + activation_bytes) / (1024**2))


def estimate_peak_memory_mb(
    params: dict[str, Any],
    batch_size: int | None = None,
    dtype_bytes: int = 4,
) -> float:
    """Estimate approximate peak training memory in MB.

    This is a conservative heuristic that combines:

    - **Parameter memory** × 4 (weights + gradients + Adam m + Adam v)
    - **Activation memory** × 3 (forward + backward + workspace)

    The factor of 4 for parameters comes from Adam's two momentum
    buffers plus the gradient buffer.  The activation estimate uses
    ``batch_size × (summary_dim + subnet_width) × depth`` as a rough
    proxy for the largest intermediate tensor.

    Parameters
    ----------
    params
        Hyperparameter dict from the search space.
    batch_size
        Override batch size (otherwise read from ``params``).
    dtype_bytes
        Bytes per element (default 4 for float32).
    """
    summary_params, summary_dim = _estimate_summary_params(params)
    inference_params = _estimate_inference_params(params, summary_dim)
    total_params = max(1, summary_params + inference_params)

    if batch_size is None:
        batch_size = _safe_int(params.get("batch_size"), 256)

    subnet_width = _subnet_width(params)
    subnet_depth = _subnet_depth(params)
    flow_depth = _safe_int(params.get("cf_depth"), 1)
    activation_depth = max(1, subnet_depth * flow_depth)

    activation_elements = max(
        1,
        batch_size * max(1, summary_dim + subnet_width) * activation_depth,
    )

    # Weights + gradients + Adam states (approx 4x parameter memory) plus activations.
    param_bytes = total_params * dtype_bytes * 4
    activation_bytes = activation_elements * dtype_bytes * 3

    return float((param_bytes + activation_bytes) / (1024**2))


def exceeds_memory_budget(
    params: dict[str, Any],
    max_memory_mb: float,
    batch_size: int | None = None,
) -> bool:
    """Return True when the estimated peak memory exceeds a budget."""
    estimated_mb = estimate_peak_memory_mb(params=params, batch_size=batch_size)
    return estimated_mb > float(max_memory_mb)


def _detect_gpu_memory_mb(safety_margin: float = 0.2) -> float | None:
    """Detect free GPU memory in MB with a safety margin applied.

    Uses ``torch.cuda.mem_get_info()`` and returns:

    ``free_bytes * (1 - safety_margin) / (1024.0 ** 2)``

    Returns ``None`` when CUDA is unavailable or cannot be queried.
    """
    if not (0.0 <= float(safety_margin) < 1.0):
        raise ValueError(
            "safety_margin must satisfy 0.0 <= safety_margin < 1.0, "
            f"got {safety_margin!r}"
        )

    try:
        import torch
    except ImportError:
        return None

    try:
        if not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info()
    except (AttributeError, RuntimeError):
        return None

    return float(free_bytes) * (1.0 - float(safety_margin)) / (1024.0 ** 2)
