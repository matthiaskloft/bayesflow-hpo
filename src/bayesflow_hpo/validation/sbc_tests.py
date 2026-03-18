"""SBC (Simulation-Based Calibration) rank-uniformity tests.

SBC checks whether the posterior approximation is well-calibrated by
verifying that the rank statistics of true values within posterior draws
are uniformly distributed.  Three tests are provided:

- **KS test**: Kolmogorov-Smirnov test against Uniform(0, 1).
- **Chi-squared test**: Binned goodness-of-fit against uniform expected
  counts.  Skipped when bins have fewer than 5 expected observations.
"""

from __future__ import annotations

import numpy as np


def compute_sbc_uniformity_tests(
    ranks: np.ndarray,
    n_posterior_samples: int,
    n_bins: int = 20,
) -> dict[str, float]:
    """Compute KS and chi-squared uniformity tests on SBC ranks."""
    from scipy.stats import chisquare, kstest

    n_sims = len(ranks)
    if n_sims == 0:
        return {
            "sbc_ks_stat": np.nan,
            "sbc_ks_pvalue": np.nan,
            "sbc_chi2_stat": np.nan,
            "sbc_chi2_pvalue": np.nan,
        }

    normalized_ranks = (ranks + 0.5) / (n_posterior_samples + 1)
    ks_stat, ks_pvalue = kstest(normalized_ranks, "uniform")

    n_bins_actual = min(n_bins, n_posterior_samples + 1)
    bin_range = (-0.5, n_posterior_samples + 0.5)
    hist, _ = np.histogram(ranks, bins=n_bins_actual, range=bin_range)
    expected_per_bin = n_sims / n_bins_actual

    if expected_per_bin >= 5:
        expected = [expected_per_bin] * n_bins_actual
        chi2_stat, chi2_pvalue = chisquare(hist, f_exp=expected)
    else:
        chi2_stat, chi2_pvalue = np.nan, np.nan

    return {
        "sbc_ks_stat": float(ks_stat),
        "sbc_ks_pvalue": float(ks_pvalue),
        "sbc_chi2_stat": float(chi2_stat) if not np.isnan(chi2_stat) else np.nan,
        "sbc_chi2_pvalue": float(chi2_pvalue) if not np.isnan(chi2_pvalue) else np.nan,
    }


