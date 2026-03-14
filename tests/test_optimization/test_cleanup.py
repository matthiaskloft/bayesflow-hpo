"""Tests for bayesflow_hpo.optimization.cleanup."""

from bayesflow_hpo.optimization.cleanup import cleanup_trial


def test_cleanup_trial_runs_without_error():
    """cleanup_trial should not raise even without GPU backends."""
    cleanup_trial()


def test_cleanup_trial_is_idempotent():
    """Calling cleanup_trial multiple times should be safe."""
    cleanup_trial()
    cleanup_trial()
