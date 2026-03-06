"""Tests for the resume parameter in create_study / optimize."""

import optuna
import pytest

from bayesflow_hpo.optimization.study import create_study


def _add_dummy_trial(study: optuna.Study) -> None:
    """Add a single completed trial to *study*."""
    trial = optuna.trial.create_trial(
        params={"x": 0.5},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 1.0)},
        values=(0.5, 100.0),
    )
    study.add_trial(trial)


@pytest.fixture()
def tmp_storage(tmp_path):
    """Return a temporary SQLite storage URL."""
    return f"sqlite:///{tmp_path / 'test.db'}"


class TestResumeFalse:
    """resume=False (load_if_exists=False) should start a fresh study."""

    def test_creates_new_study(self, tmp_storage):
        study = create_study(storage=tmp_storage, load_if_exists=False)
        assert len(study.trials) == 0

    def test_raises_on_duplicate_name(self, tmp_storage):
        create_study(storage=tmp_storage, load_if_exists=False)
        with pytest.raises(optuna.exceptions.DuplicatedStudyError):
            create_study(storage=tmp_storage, load_if_exists=False)


class TestResumeTrue:
    """resume=True (load_if_exists=True) should continue an existing study."""

    def test_preserves_existing_trials(self, tmp_storage):
        study = create_study(storage=tmp_storage, load_if_exists=True)
        _add_dummy_trial(study)
        assert len(study.trials) == 1

        resumed = create_study(storage=tmp_storage, load_if_exists=True)
        assert len(resumed.trials) == 1

    def test_creates_if_not_exists(self, tmp_storage):
        study = create_study(storage=tmp_storage, load_if_exists=True)
        assert len(study.trials) == 0


class TestInMemoryStorage:
    """In-memory storage (storage=None) should always work."""

    def test_in_memory_creates_fresh(self):
        study = create_study(storage=None, load_if_exists=True)
        assert len(study.trials) == 0
