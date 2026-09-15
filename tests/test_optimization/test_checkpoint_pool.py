"""Tests for bayesflow_hpo.optimization.checkpoint_pool."""

import json
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestPrunedPool:
    """Retention of pruned trials' weights (bayesflow-hpo#106)."""

    def test_disabled_by_default(self, pool_dir, mock_approximator):
        pool = CheckpointPool(pool_dir=pool_dir)
        assert pool.pruned_pool_size == 0
        assert pool.save_pruned(0, mock_approximator, step=3) is False
        assert pool.pruned_trial_numbers == []
        mock_approximator.save_weights.assert_not_called()

    def test_saves_when_enabled(self, pool_dir, mock_approximator):
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=2, seed=0,
        )
        assert pool.save_pruned(7, mock_approximator, step=3) is True
        assert pool.pruned_trial_numbers == [7]
        dest = pool.pruned_pool_dir / "trial_0007"
        assert dest.is_dir()

    def test_metadata_records_the_rung(self, pool_dir, mock_approximator):
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=2, seed=0,
        )
        pool.save_pruned(7, mock_approximator, step=4, objective_value=0.25)
        meta = json.loads(
            (pool.pruned_pool_dir / "trial_0007" / "checkpoint.json").read_text()
        )
        assert meta["state"] == "pruned"
        assert meta["pruned_at_step"] == 4
        assert meta["objective_value"] == pytest.approx(0.25)
        assert meta["weights_file"] == "weights.weights.h5"

    def test_complete_checkpoints_get_metadata_too(
        self, pool_dir, mock_approximator
    ):
        pool = CheckpointPool(pool_dir=pool_dir, pool_size=2)
        pool.maybe_save(3, 0.5, mock_approximator)
        meta = json.loads(
            (pool_dir / "trial_0003" / "checkpoint.json").read_text()
        )
        assert meta["state"] == "complete"
        assert meta["objective_value"] == pytest.approx(0.5)

    def test_pruned_pool_is_separate_from_top_k(
        self, pool_dir, mock_approximator
    ):
        pool = CheckpointPool(
            pool_dir=pool_dir, pool_size=1, pruned_pool_size=3, seed=0,
        )
        pool.maybe_save(0, 0.1, mock_approximator)
        for n in range(1, 6):
            pool.save_pruned(n, mock_approximator, step=1)
        # No pruned trial displaced the scored one, and the caps are
        # enforced independently.
        assert pool.trial_numbers == [0]
        assert len(pool.pruned_trial_numbers) == 3

    def test_respects_cap(self, pool_dir, mock_approximator):
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=2, seed=0,
        )
        for n in range(20):
            pool.save_pruned(n, mock_approximator, step=1)
        assert len(pool.pruned_trial_numbers) == 2
        kept = set(pool.pruned_trial_numbers)
        on_disk = {
            int(d.name.split("_")[1])
            for d in pool.pruned_pool_dir.iterdir()
            if d.is_dir()
        }
        assert on_disk == kept

    def test_retention_is_uniform_over_the_population(
        self, pool_dir, mock_approximator
    ):
        """Every pruned trial is equally likely to survive.

        The point of not using top-k: the retained sample must cover the
        whole quality range, which means a late trial must not be favoured
        over an early one (or vice versa).
        """
        population, cap, replicates = 10, 3, 600
        counts: Counter[int] = Counter()
        for rep in range(replicates):
            pool = CheckpointPool(
                pool_dir=pool_dir / f"rep_{rep}",
                pruned_pool_size=cap,
                seed=rep,
            )
            for n in range(population):
                pool.save_pruned(n, mock_approximator, step=1)
            counts.update(pool.pruned_trial_numbers)

        expected = replicates * cap / population
        for n in range(population):
            assert counts[n] == pytest.approx(expected, rel=0.2)

    def test_save_weights_failure_returns_false(self, pool_dir):
        approx = MagicMock()
        approx.save_weights.side_effect = RuntimeError("disk full")
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=2, seed=0,
        )
        assert pool.save_pruned(0, approx, step=1) is False
        assert pool.pruned_trial_numbers == []

    def test_failed_write_does_not_consume_a_draw(self, pool_dir):
        """A write that never landed must not skew the sample.

        Counting it would make the retained set a uniform sample of the
        offers rather than of the trials that could actually be retained.
        """
        approx = MagicMock()
        approx.save_weights.side_effect = RuntimeError("disk full")
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=2, seed=0,
        )
        for n in range(5):
            pool.save_pruned(n, approx, step=1)
        assert pool._pruned_seen == 0

    def test_reoffering_the_same_trial_keeps_its_checkpoint(
        self, pool_dir, mock_approximator
    ):
        """Drawing its own slot must not delete what was just written."""
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=1, seed=0,
        )
        for _ in range(10):
            pool.save_pruned(4, mock_approximator, step=1)
        assert pool.pruned_trial_numbers == [4]
        assert (pool.pruned_pool_dir / "trial_0004").is_dir()

    def test_reoffer_updates_in_place_below_capacity(
        self, pool_dir, mock_approximator
    ):
        """One trial is one member of the population, not two.

        Appending a second entry for the same trial would let a later
        eviction delete the shared directory while the other entry still
        reported the trial as retained.
        """
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=4, seed=0,
        )
        pool.save_pruned(2, mock_approximator, step=1)
        pool.save_pruned(2, mock_approximator, step=7)

        assert pool.pruned_trial_numbers == [2]
        assert pool._pruned_seen == 1
        meta = json.loads(
            (pool.pruned_pool_dir / "trial_0002" / "checkpoint.json").read_text()
        )
        assert meta["pruned_at_step"] == 7

    def test_failed_weights_write_leaves_no_orphan_directory(self, pool_dir):
        """A saver that writes bytes and then raises must leave nothing.

        The caller never records a failed checkpoint, so an orphan
        directory is unreachable by eviction and grows the pool past its
        cap.
        """
        def _write_then_raise(path):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"partial")
            raise RuntimeError("disk full")

        approx = MagicMock()
        approx.save_weights.side_effect = _write_then_raise
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=2, seed=0,
        )
        for n in range(5):
            assert pool.save_pruned(n, approx, step=1) is False

        assert pool.pruned_trial_numbers == []
        leftovers = (
            list(pool.pruned_pool_dir.iterdir())
            if pool.pruned_pool_dir.exists()
            else []
        )
        assert leftovers == []

    def test_metadata_failure_does_not_publish_the_checkpoint(self, pool_dir):
        """Weights without their rung are not a usable pruned checkpoint."""
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=2, seed=0,
        )
        approx = MagicMock()
        with patch(
            "bayesflow_hpo.optimization.checkpoint_pool.Path.write_text",
            side_effect=OSError("read-only"),
        ):
            assert pool.save_pruned(0, approx, step=1) is False

        assert pool.pruned_trial_numbers == []
        assert not (pool.pruned_pool_dir / "trial_0000").exists()

    def test_failed_reoffer_preserves_the_existing_checkpoint(
        self, pool_dir, mock_approximator
    ):
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=2, seed=0,
        )
        pool.save_pruned(1, mock_approximator, step=3)

        broken = MagicMock()
        broken.save_weights.side_effect = RuntimeError("disk full")
        assert pool.save_pruned(1, broken, step=9) is False

        assert pool.pruned_trial_numbers == [1]
        meta = json.loads(
            (pool.pruned_pool_dir / "trial_0001" / "checkpoint.json").read_text()
        )
        assert meta["pruned_at_step"] == 3

    def test_cleanup_clears_pruned_pool(self, pool_dir, mock_approximator):
        pool = CheckpointPool(
            pool_dir=pool_dir, pruned_pool_size=2, seed=0,
        )
        pool.save_pruned(0, mock_approximator, step=1)
        pool.cleanup()
        assert not pool.pruned_pool_dir.exists()
        assert pool.pruned_trial_numbers == []
