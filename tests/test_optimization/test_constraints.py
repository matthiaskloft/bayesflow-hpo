"""Tests for optimization constraints/helpers."""

import pytest

from bayesflow_hpo.optimization.constraints import (
    estimate_param_count,
    estimate_peak_memory_mb,
    exceeds_memory_budget,
)


def test_estimate_param_count_positive():
    params = {
        "ds_summary_dim": 8,
        "ds_width": 64,
        "ds_depth": 2,
        "cf_depth": 6,
        "cf_subnet_width": 128,
    }
    estimated = estimate_param_count(params)
    assert estimated > 0


@pytest.mark.parametrize(
    "params",
    [
        {"dm_subnet_width": 64, "dm_subnet_depth": 2},
        {"cm_subnet_width": 64, "cm_subnet_depth": 2},
        {"scm_subnet_width": 64, "scm_subnet_depth": 2},
        {"st_summary_dim": 16, "st_embed_dim": 64, "st_num_heads": 4, "st_num_layers": 2},
        {
            "tst_summary_dim": 16,
            "tst_embed_dim": 64,
            "tst_num_heads": 4,
            "tst_num_layers": 2,
        },
        {
            "ft_summary_dim": 16,
            "ft_embed_dim": 64,
            "ft_num_heads": 4,
            "ft_num_layers": 2,
            "ft_template_dim": 128,
        },
        {"tsn_summary_dim": 16, "tsn_filters": 32, "tsn_recurrent_dim": 128},
    ],
)
def test_estimate_param_count_phase2_networks_positive(params):
    assert estimate_param_count(params) > 0


def test_estimate_peak_memory_mb_positive():
    params = {
        "ds_summary_dim": 16,
        "ds_width": 64,
        "ds_depth": 2,
        "cf_depth": 6,
        "cf_subnet_width": 128,
        "cf_subnet_depth": 2,
        "batch_size": 256,
    }
    assert estimate_peak_memory_mb(params) > 0.0


def test_exceeds_memory_budget_threshold_behavior():
    params = {
        "ds_summary_dim": 16,
        "ds_width": 64,
        "ds_depth": 2,
        "cf_depth": 6,
        "cf_subnet_width": 128,
        "cf_subnet_depth": 2,
        "batch_size": 256,
    }
    estimate = estimate_peak_memory_mb(params)
    assert exceeds_memory_budget(params, max_memory_mb=max(estimate - 1e-6, 0.0))
    assert not exceeds_memory_budget(params, max_memory_mb=estimate + 1.0)
