"""Tests for optimization constraints/helpers."""

import types

import pytest

from bayesflow_hpo.optimization.constraints import (
    _detect_gpu_memory_mb,
    estimate_param_count,
    estimate_peak_memory_mb,
    estimate_validation_memory_mb,
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
        {
            "st_summary_dim": 16,
            "st_embed_dim": 64,
            "st_num_heads": 4,
            "st_num_layers": 2,
        },
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


def test_detect_gpu_memory_mb_returns_none_without_torch(monkeypatch):
    import builtins
    import sys

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert _detect_gpu_memory_mb() is None


@pytest.mark.parametrize("bad_margin", [-0.1, 1.0, 1.2])
def test_detect_gpu_memory_mb_invalid_safety_margin_raises(bad_margin):
    with pytest.raises(ValueError, match="safety_margin must satisfy"):
        _detect_gpu_memory_mb(safety_margin=bad_margin)


def test_detect_gpu_memory_mb_returns_none_when_cuda_unavailable(monkeypatch):
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)
    assert _detect_gpu_memory_mb() is None


def test_detect_gpu_memory_mb_returns_none_on_runtime_error(monkeypatch):
    def _raise_runtime_error():
        raise RuntimeError("cuda init failed")

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=_raise_runtime_error,
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)
    assert _detect_gpu_memory_mb() is None


def test_detect_gpu_memory_mb_returns_none_without_mem_get_info(monkeypatch):
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)
    assert _detect_gpu_memory_mb() is None

def test_detect_gpu_memory_mb_applies_safety_margin(monkeypatch):
    free_bytes = 2 * 1024 * 1024 * 1024  # 2 GiB
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (free_bytes, free_bytes),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)
    detected = _detect_gpu_memory_mb(safety_margin=0.25)
    expected = free_bytes * (1.0 - 0.25) / (1024.0**2)
    assert detected == pytest.approx(expected)


class TestValidationMemoryEstimate:
    """Sampling is a different allocation from training (#101)."""

    #: The architecture of the #101 benchmark harness
    #: (``docs/plans/bench_inference_ratio.py``): FlowMatching with subnet
    #: widths ``(128, 128)`` over a DeepSet with ``summary_dim=32,
    #: depth=2``, 15 parameters.
    BENCH_PARAMS = {
        "fm_subnet_width": 128,
        "fm_subnet_depth": 2,
        "ds_summary_dim": 32,
        "ds_depth": 2,
        "n_params": 15,
    }

    def test_matches_the_measured_oom_threshold(self):
        """Calibration is pinned to the measurement it came from.

        The benchmark completed 40,000 draws under a 6.29 GiB process cap
        and went out of memory at 60,000. An estimate that brackets those
        two is the whole claim this function makes; drifting off it makes
        `max_memory_mb` dishonest again in one direction or useless in the
        other.
        """
        cap_mb = 6.29 * 1024

        ok = estimate_validation_memory_mb(
            self.BENCH_PARAMS,
            n_sims=40_000,
            n_posterior_samples=1,
            max_samples_per_call=None,
        )
        oom = estimate_validation_memory_mb(
            self.BENCH_PARAMS,
            n_sims=60_000,
            n_posterior_samples=1,
            max_samples_per_call=None,
        )
        assert ok < cap_mb < oom

    def test_scales_with_the_sample_product(self):
        base = estimate_validation_memory_mb(
            self.BENCH_PARAMS, n_sims=100, n_posterior_samples=100,
            max_samples_per_call=None,
        )
        doubled = estimate_validation_memory_mb(
            self.BENCH_PARAMS, n_sims=200, n_posterior_samples=100,
            max_samples_per_call=None,
        )
        assert doubled > 1.9 * base

    def test_the_cap_is_what_bounds_the_estimate(self):
        """The peak is one chunk, so the cap -- not the condition -- sets it."""
        capped = estimate_validation_memory_mb(
            self.BENCH_PARAMS, n_sims=200, n_posterior_samples=500,
            max_samples_per_call=20_000,
        )
        uncapped = estimate_validation_memory_mb(
            self.BENCH_PARAMS, n_sims=200, n_posterior_samples=500,
            max_samples_per_call=None,
        )
        assert capped < uncapped
        # A larger condition costs nothing more once the cap binds.
        assert capped == estimate_validation_memory_mb(
            self.BENCH_PARAMS, n_sims=2_000, n_posterior_samples=500,
            max_samples_per_call=20_000,
        )

    def test_sample_count_above_the_cap_keeps_one_simulation_whole(self):
        """Mirrors the closure, which never splits a single simulation."""
        estimate = estimate_validation_memory_mb(
            self.BENCH_PARAMS, n_sims=10, n_posterior_samples=50_000,
            max_samples_per_call=20_000,
        )
        one_sim = estimate_validation_memory_mb(
            self.BENCH_PARAMS, n_sims=1, n_posterior_samples=50_000,
            max_samples_per_call=None,
        )
        assert estimate == one_sim

    def test_ode_sampled_networks_cost_more_than_a_coupling_flow(self):
        """Seven live tsit5 stages against one sequential inverse."""
        shared = {"ds_summary_dim": 32, "ds_depth": 2, "n_params": 15}
        flow_matching = estimate_validation_memory_mb(
            {**shared, "fm_subnet_width": 128, "fm_subnet_depth": 2},
            n_sims=100, n_posterior_samples=100, max_samples_per_call=None,
        )
        coupling = estimate_validation_memory_mb(
            {**shared, "cf_subnet_width": 128, "cf_subnet_depth": 2,
             "cf_depth": 1},
            n_sims=100, n_posterior_samples=100, max_samples_per_call=None,
        )
        assert flow_matching > coupling

    def test_training_estimate_is_not_the_sampling_estimate(self):
        """At the `optimize()` defaults, sampling is the larger allocation."""
        params = {**self.BENCH_PARAMS, "batch_size": 256}
        training = estimate_peak_memory_mb(params)
        sampling = estimate_validation_memory_mb(
            params, n_sims=200, n_posterior_samples=500,
            max_samples_per_call=20_000,
        )
        assert sampling > training

    def test_degenerate_counts_do_not_raise(self):
        for kwargs in (
            {"n_sims": 0, "n_posterior_samples": 0},
            {"n_sims": -5, "n_posterior_samples": 10},
            {"n_sims": "bad", "n_posterior_samples": None},
        ):
            assert estimate_validation_memory_mb(
                self.BENCH_PARAMS, max_samples_per_call=20_000, **kwargs
            ) > 0
