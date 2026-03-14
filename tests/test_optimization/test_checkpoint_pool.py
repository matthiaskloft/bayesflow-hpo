"""Tests for bayesflow_hpo.optimization.checkpoint_pool."""

from unittest.mock import MagicMock

import pytest

from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool


@pytest.fixture
def pool_dir(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture
def mock_approximator():
    approx = MagicMock()
    approx.save_weights = MagicMock()
    return approx


class TestCheckpointPool:
    def test_save_first_trial(self, pool_dir, mock_approximator):
        pool = CheckpointPool(pool_dir=pool_dir, pool_size=3)
        saved = pool.maybe_save(0, 0.5, mock_approximator)
        assert saved is True
        mock_approximator.save_weights.assert_called_once()

    def test_best_checkpoint_dir_empty(self, pool_dir):
        pool = CheckpointPool(pool_dir=pool_dir)
        assert pool.best_checkpoint_dir is None

    def test_best_checkpoint_dir_after_save(self, pool_dir, mock_approximator):
        pool = CheckpointPool(pool_dir=pool_dir)
        pool.maybe_save(0, 0.5, mock_approximator)
        assert pool.best_checkpoint_dir is not None
        assert "trial_0000" in str(pool.best_checkpoint_dir)

    def test_trial_numbers_ordering(self, pool_dir, mock_approximator):
        pool = CheckpointPool(pool_dir=pool_dir, pool_size=5)
        pool.maybe_save(3, 0.8, mock_approximator)
        pool.maybe_save(1, 0.2, mock_approximator)
        pool.maybe_save(2, 0.5, mock_approximator)
        # Sorted by objective value (best first)
        assert pool.trial_numbers == [1, 2, 3]

    def test_eviction_at_capacity(self, pool_dir, mock_approximator):
        pool = CheckpointPool(pool_dir=pool_dir, pool_size=2)
        pool.maybe_save(0, 0.3, mock_approximator)
        pool.maybe_save(1, 0.5, mock_approximator)
        pool.maybe_save(2, 0.1, mock_approximator)
        # Trial 1 (worst, 0.5) should be evicted
        assert len(pool.trial_numbers) == 2
        assert 1 not in pool.trial_numbers
        assert pool.trial_numbers == [2, 0]

    def test_reject_worse_than_pool(self, pool_dir, mock_approximator):
        pool = CheckpointPool(pool_dir=pool_dir, pool_size=2)
        pool.maybe_save(0, 0.3, mock_approximator)
        pool.maybe_save(1, 0.5, mock_approximator)
        # This is >= worst in pool (0.5), should be rejected
        saved = pool.maybe_save(2, 0.6, mock_approximator)
        assert saved is False
        assert len(pool.trial_numbers) == 2

    def test_save_weights_failure_returns_false(self, pool_dir):
        approx = MagicMock()
        approx.save_weights.side_effect = RuntimeError("disk full")
        pool = CheckpointPool(pool_dir=pool_dir)
        saved = pool.maybe_save(0, 0.5, approx)
        assert saved is False

    def test_cleanup_removes_dir(self, pool_dir, mock_approximator):
        pool = CheckpointPool(pool_dir=pool_dir, pool_size=3)
        pool.maybe_save(0, 0.5, mock_approximator)
        pool.cleanup()
        assert not pool_dir.exists()
        assert pool.trial_numbers == []

    def test_cleanup_empty_pool(self, pool_dir):
        pool = CheckpointPool(pool_dir=pool_dir)
        pool.cleanup()  # Should not raise
        assert pool.trial_numbers == []
