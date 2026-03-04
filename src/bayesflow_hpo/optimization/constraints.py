"""Constraint and budget helpers for trial pre-filtering."""

from __future__ import annotations

from typing import Any


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mlp_block_params(input_dim: int, hidden_dim: int, depth: int, output_dim: int) -> int:
    if depth <= 0:
        return max(1, input_dim * output_dim)

    first = input_dim * hidden_dim
    middle = max(0, depth - 1) * hidden_dim * hidden_dim
    last = hidden_dim * output_dim
    return max(1, first + middle + last)


def _estimate_summary_params(params: dict[str, Any]) -> tuple[int, int]:
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
        total = layers * (embed_dim * embed_dim + embed_dim * mlp_width + heads * embed_dim)
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
    deepset_params = 4 * deepset_depth * (deepset_width**2)
    deepset_params += deepset_width * summary_dim
    return max(1, deepset_params), summary_dim


def _estimate_inference_params(params: dict[str, Any], summary_dim: int) -> int:
    n_conditions = _safe_int(params.get("n_conditions"), 4)
    n_params = _safe_int(params.get("n_params"), 1)
    input_dim = summary_dim + n_conditions + max(1, n_params // 2)
    output_dim = 2 * max(1, (n_params + 1) // 2)

    if "cf_depth" in params:
        depth = _safe_int(params.get("cf_depth"), 6)
        hidden = _safe_int(params.get("cf_subnet_width"), 128)
        subnet_depth = _safe_int(params.get("cf_subnet_depth"), 2)
        subnet = _mlp_block_params(input_dim, hidden, subnet_depth, output_dim)
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
    """Heuristic parameter estimate from supported search-space keys."""
    summary_params, summary_dim = _estimate_summary_params(params)
    inference_params = _estimate_inference_params(params, summary_dim)
    regularization = int(1000 * _safe_float(params.get("initial_lr"), 1e-3))
    return max(1, int(summary_params + inference_params + regularization))


def estimate_peak_memory_mb(
    params: dict[str, Any],
    batch_size: int | None = None,
    dtype_bytes: int = 4,
) -> float:
    """Estimate approximate peak training memory in MB.

    This is a conservative heuristic that combines parameter memory,
    optimizer state, gradients, and a rough activation budget.
    """
    summary_params, summary_dim = _estimate_summary_params(params)
    inference_params = _estimate_inference_params(params, summary_dim)
    total_params = max(1, summary_params + inference_params)

    if batch_size is None:
        batch_size = _safe_int(params.get("batch_size"), 256)

    subnet_width = _safe_int(
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
    subnet_depth = _safe_int(
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
