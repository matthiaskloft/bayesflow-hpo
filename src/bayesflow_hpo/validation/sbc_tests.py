"""SBC rank-uniformity tests."""

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


def compute_sbc_c2st(
    ranks: np.ndarray,
    n_posterior_samples: int,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Classifier two-sample test for SBC rank uniformity."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return {"sbc_c2st_accuracy": np.nan, "sbc_c2st_sd": np.nan}

    n_sims = len(ranks)
    if n_sims < 2 * n_folds:
        return {"sbc_c2st_accuracy": np.nan, "sbc_c2st_sd": np.nan}

    rng = np.random.RandomState(random_state)
    uniform_ranks = rng.randint(0, n_posterior_samples + 1, size=n_sims)

    x_data = np.concatenate([ranks, uniform_ranks]).reshape(-1, 1)
    y = np.concatenate([np.ones(n_sims), np.zeros(n_sims)])

    clf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=random_state,
    )
    scores = cross_val_score(clf, x_data, y, cv=n_folds, scoring="accuracy")

    return {
        "sbc_c2st_accuracy": float(np.mean(scores)),
        "sbc_c2st_sd": float(np.std(scores)),
    }
