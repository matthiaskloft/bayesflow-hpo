"""Tests for C2ST metrics (L-C2ST, global C2ST, and ValidateFn factory)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from bayesflow_hpo.validation.c2st import (
    GlobalC2STResult,
    LC2STResult,
    global_c2st,
    lc2st,
    make_lc2st_validate_fn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42


def _make_well_calibrated(
    n_sims: int = 100,
    n_samples: int = 50,
    n_params: int = 2,
    n_obs: int = 3,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate well-calibrated synthetic posterior draws.

    true_params ~ N(0, 1), posterior_samples ~ N(true_params, 0.1).
    """
    rng = np.random.default_rng(seed)
    true_params = rng.standard_normal((n_sims, n_params))
    noise = rng.standard_normal((n_sims, n_samples, n_params))
    posterior_samples = true_params[:, None, :] + 0.1 * noise
    observations = rng.standard_normal((n_sims, n_obs))
    return posterior_samples, true_params, observations


def _make_mismatched(
    n_sims: int = 100,
    n_samples: int = 50,
    n_params: int = 2,
    n_obs: int = 3,
    shift: float = 5.0,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate biased posterior draws (shifted away from true params)."""
    rng = np.random.default_rng(seed)
    true_params = rng.standard_normal((n_sims, n_params))
    noise = rng.standard_normal((n_sims, n_samples, n_params))
    posterior_samples = true_params[:, None, :] + shift + 0.1 * noise
    observations = rng.standard_normal((n_sims, n_obs))
    return posterior_samples, true_params, observations


# ---------------------------------------------------------------------------
# L-C2ST tests
# ---------------------------------------------------------------------------


class TestLC2ST:
    """Tests for lc2st() standalone function."""

    def test_lc2st_detects_mismatch(self) -> None:
        """Biased posterior should produce statistic >> 0."""
        posterior, true_params, obs = _make_mismatched(
            n_sims=80, seed=SEED,
        )
        result = lc2st(
            posterior, true_params, obs,
            n_folds=5, n_null_trials=10, seed=SEED,
        )
        assert result.statistic > 0.05
        assert result.p_value is not None
        assert result.p_value < 0.1

    def test_lc2st_consistent_posterior(self) -> None:
        """Well-calibrated posterior should produce statistic near 0."""
        posterior, true_params, obs = _make_well_calibrated(
            n_sims=80, seed=SEED,
        )
        result = lc2st(
            posterior, true_params, obs,
            n_folds=5, n_null_trials=0, seed=SEED,
        )
        # Statistic should be small (near chance level)
        assert result.statistic < 0.05

    def test_lc2st_result_fields(self) -> None:
        """Result dataclass has expected fields and shapes."""
        posterior, true_params, obs = _make_well_calibrated(
            n_sims=40, seed=SEED,
        )
        result = lc2st(
            posterior, true_params, obs,
            n_folds=5, n_null_trials=3, seed=SEED,
        )
        assert isinstance(result, LC2STResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert result.null_statistics.shape == (3,)
        assert result.per_observation_stats.shape == (40,)

    def test_lc2st_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        posterior, true_params, obs = _make_well_calibrated(
            n_sims=40, seed=SEED,
        )
        r1 = lc2st(posterior, true_params, obs, seed=123)
        r2 = lc2st(posterior, true_params, obs, seed=123)
        assert r1.statistic == r2.statistic
        np.testing.assert_array_equal(
            r1.per_observation_stats, r2.per_observation_stats,
        )

    def test_lc2st_single_param(self) -> None:
        """Works with n_params=1."""
        posterior, true_params, obs = _make_well_calibrated(
            n_sims=40, n_params=1, seed=SEED,
        )
        result = lc2st(posterior, true_params, obs, seed=SEED)
        assert isinstance(result.statistic, float)
        assert np.isfinite(result.statistic)

    def test_lc2st_no_null_trials(self) -> None:
        """n_null_trials=0 skips permutation test."""
        posterior, true_params, obs = _make_well_calibrated(
            n_sims=40, seed=SEED,
        )
        result = lc2st(
            posterior, true_params, obs,
            n_null_trials=0, seed=SEED,
        )
        assert result.p_value is None
        assert len(result.null_statistics) == 0


# ---------------------------------------------------------------------------
# Global C2ST tests
# ---------------------------------------------------------------------------


class TestGlobalC2ST:
    """Tests for global_c2st() standalone function."""

    def test_global_c2st_same_distribution(self) -> None:
        """Same distribution should give accuracy near 0.5."""
        rng = np.random.default_rng(SEED)
        samples = rng.standard_normal((200, 3))
        result = global_c2st(
            samples[:100], samples[100:], seed=SEED,
        )
        assert 0.3 < result.accuracy < 0.7
        assert result.p_value > 0.05

    def test_global_c2st_different_distribution(self) -> None:
        """Different distributions should give high accuracy."""
        rng = np.random.default_rng(SEED)
        p = rng.standard_normal((200, 3))
        q = rng.standard_normal((200, 3)) + 3.0
        result = global_c2st(
            p, q,
            clf_kwargs={
                "hidden_layer_sizes": (20,),
                "max_iter": 500,
                "solver": "adam",
            },
            seed=SEED,
        )
        assert result.accuracy > 0.8
        assert result.p_value < 0.01

    def test_global_c2st_result_fields(self) -> None:
        """Result dataclass has expected fields."""
        rng = np.random.default_rng(SEED)
        p = rng.standard_normal((50, 2))
        q = rng.standard_normal((50, 2))
        result = global_c2st(p, q, seed=SEED)
        assert isinstance(result, GlobalC2STResult)
        assert isinstance(result.accuracy, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.n_test, int)
        assert result.n_test > 0

    def test_global_c2st_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        rng = np.random.default_rng(SEED)
        p = rng.standard_normal((50, 2))
        q = rng.standard_normal((50, 2)) + 1.0
        r1 = global_c2st(p, q, seed=99)
        r2 = global_c2st(p, q, seed=99)
        assert r1.accuracy == r2.accuracy
        assert r1.p_value == r2.p_value


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestMakeLC2STValidateFn:
    """Tests for make_lc2st_validate_fn() factory."""

    def test_make_lc2st_validate_fn_returns_callable(self) -> None:
        """Factory returns a callable."""
        fn = make_lc2st_validate_fn()
        assert callable(fn)

    def test_lc2st_validate_fn_output_keys(self) -> None:
        """Returned function produces expected metric keys."""
        from unittest.mock import MagicMock

        from bayesflow_hpo.validation.data import ValidationDataset

        rng = np.random.default_rng(SEED)
        n_sims, n_samples = 30, 20
        true_p1 = rng.standard_normal(n_sims)
        true_p2 = rng.standard_normal(n_sims)
        obs = rng.standard_normal((n_sims, 3))

        # Mock approximator that returns draws matching true params
        mock_approx = MagicMock()
        mock_approx.sample.return_value = {
            "p1": true_p1[:, None] + 0.1 * rng.standard_normal((n_sims, n_samples)),
            "p2": true_p2[:, None] + 0.1 * rng.standard_normal((n_sims, n_samples)),
        }

        val_data = ValidationDataset(
            simulations=[{"p1": true_p1, "p2": true_p2, "x": obs}],
            condition_labels=[{}],
            param_keys=["p1", "p2"],
            data_keys=["x"],
            seed=SEED,
        )

        fn = make_lc2st_validate_fn(
            base_metrics=["calibration_error", "nrmse"],
            n_folds=3,
            seed=SEED,
        )
        result = fn(mock_approx, val_data, n_samples)

        assert isinstance(result, dict)
        assert "lc2st" in result
        assert "calibration_error" in result
        assert "nrmse" in result
        assert np.isfinite(result["lc2st"])

    def test_lc2st_validate_fn_single_param(self) -> None:
        """Factory works with single-parameter models."""
        from unittest.mock import MagicMock

        from bayesflow_hpo.validation.data import ValidationDataset

        rng = np.random.default_rng(SEED)
        n_sims, n_samples = 30, 20
        true_p = rng.standard_normal(n_sims)
        obs = rng.standard_normal((n_sims, 2))

        mock_approx = MagicMock()
        mock_approx.sample.return_value = {
            "theta": true_p[:, None] + 0.1 * rng.standard_normal((n_sims, n_samples)),
        }

        val_data = ValidationDataset(
            simulations=[{"theta": true_p, "x": obs}],
            condition_labels=[{}],
            param_keys=["theta"],
            data_keys=["x"],
            seed=SEED,
        )

        fn = make_lc2st_validate_fn(
            base_metrics=["nrmse"],
            n_folds=3,
            seed=SEED,
        )
        result = fn(mock_approx, val_data, n_samples)

        assert "lc2st" in result
        assert "nrmse" in result
        assert np.isfinite(result["lc2st"])


# ---------------------------------------------------------------------------
# Import guard test
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Tests for sklearn import guard."""

    def test_require_sklearn_error_message(self) -> None:
        """Helpful error when sklearn is missing."""
        mocked = {
            "sklearn": None,
            "sklearn.neural_network": None,
            "sklearn.model_selection": None,
        }
        with patch.dict("sys.modules", mocked):
            with pytest.raises(ImportError, match="bayesflow-hpo\\[sklearn\\]"):
                from bayesflow_hpo.validation.c2st import _require_sklearn
                _require_sklearn()
