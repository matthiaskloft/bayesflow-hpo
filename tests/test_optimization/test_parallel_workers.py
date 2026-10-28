"""Shared-storage sampler regressions for independently launched workers."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import MagicMock, patch

import optuna
import pytest
from scipy.stats import qmc

from bayesflow_hpo.optimization.study import create_study


def _qmc_worker(storage, barrier):
    study = create_study(
        study_name="parallel", directions=["minimize"], storage=storage,
        qmc_startup_trials=8, sampler_seed=123,
    )
    barrier.wait(timeout=60)
    trial = study.ask()
    read_attrs = study._storage.get_study_system_attrs

    def concurrent_counter_read(study_id):
        attrs = read_attrs(study_id)
        # Force both workers to see the same old counter if the upstream
        # non-atomic allocation path is accidentally reintroduced.
        barrier.wait(timeout=60)
        return attrs

    with patch.object(
        study._storage, "get_study_system_attrs", side_effect=concurrent_counter_read,
    ):
        return trial.number, trial.suggest_float("x", 0, 1)


def test_qmc_workers_use_unique_sobol_indices(tmp_path):
    storage = f"sqlite:///{tmp_path / 'parallel.db'}"
    study = create_study(
        study_name="parallel", directions=["minimize"], storage=storage,
        qmc_startup_trials=8,
    )
    trial = study.ask()
    trial.suggest_float("x", 0, 1)
    study.tell(trial, 0.0)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager:
        barrier = manager.Barrier(2)
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
            futures = [pool.submit(_qmc_worker, storage, barrier) for _ in range(2)]
            points = [future.result(timeout=120) for future in futures]
    expected = qmc.Sobol(d=1, scramble=False).random_base2(2).ravel()
    assert len({value for _, value in points}) == 2
    for number, value in points:
        assert value == expected[number - 1]


def test_qmc_resume_counts_other_workers_completions(tmp_path):
    storage = f"sqlite:///{tmp_path / 'resume.db'}"
    first = create_study(storage=storage, qmc_startup_trials=2)
    second = create_study(storage=storage, qmc_startup_trials=2)
    first.optimize(lambda t: [t.suggest_float("x", 0, 1)] * 2, n_trials=2)
    second.ask()
    assert not second.sampler._is_qmc_phase


@pytest.mark.parametrize("seed", [None, 7, 42])
def test_sampler_seed_and_constant_liar_forwarding(seed):
    with patch("optuna.samplers.TPESampler") as factory:
        create_study(storage=None, sampler_seed=seed, constant_liar=True)
    assert factory.call_args.kwargs["seed"] == seed
    assert factory.call_args.kwargs["constant_liar"] is True


def test_tpe_workers_propose_different_points_with_running_trial(tmp_path):
    storage = f"sqlite:///{tmp_path / 'tpe.db'}"
    first = create_study(
        storage=storage, directions=["minimize"], sampler_seed=11,
        constant_liar=True, sampler_n_startup_trials=2, budget_aware=False,
    )
    first.optimize(lambda t: (t.suggest_float("x", -1, 1) - 0.2) ** 2, n_trials=5)
    second = create_study(
        storage=storage, directions=["minimize"], sampler_seed=22,
        constant_liar=True, sampler_n_startup_trials=2, budget_aware=False,
    )
    running = first.ask()
    x = running.suggest_float("x", -1, 1)
    other = second.ask()
    assert other.suggest_float("x", -1, 1) != x
    assert first.trials[running.number].state == optuna.trial.TrialState.RUNNING


def test_resume_preserves_a_running_trial(tmp_path):
    from bayesflow_hpo.api import _create_and_run_study

    storage = f"sqlite:///{tmp_path / 'active.db'}"
    objective = MagicMock()
    kwargs = dict(
        objective=objective, study_name="active", directions=["minimize"],
        metric_names=["m"], storage=storage, n_trials=0,
        warm_start_from=None, warm_start_top_k=25, max_total_trials=None,
        show_progress_bar=False, sampler_seed=11, constant_liar=True,
    )
    first = _create_and_run_study(resume=False, **kwargs)
    trial = first.ask()
    trial.suggest_float("x", 0, 1)
    with patch("bayesflow_hpo.api.optuna.delete_study") as delete:
        second = _create_and_run_study(resume=True, **kwargs)
    delete.assert_not_called()
    assert second.trials[0].state == optuna.trial.TrialState.RUNNING
    first.tell(trial, 0.5)
    assert second.trials[0].value == 0.5

@pytest.mark.parametrize(
    ("preset", "factory_name"),
    [("random", "RandomSampler"), ("gp", "GPSampler"),
     ("nsga2", "NSGAIISampler"), ("nsga3", "NSGAIIISampler")],
)
def test_all_builtin_presets_receive_worker_seed(preset, factory_name):
    with patch(f"optuna.samplers.{factory_name}") as factory:
        create_study(storage=None, sampler=preset, sampler_seed=123)
    assert factory.call_args.kwargs["seed"] == 123


def test_optimize_forwards_worker_options():
    from bayesflow_hpo.api import optimize

    with (
        patch(
            "bayesflow_hpo.api._infer_and_validate_keys",
            return_value=(["p"], ["x"]),
        ),
        patch("bayesflow_hpo.api._setup_validation_data"),
        patch("bayesflow_hpo.api.check_pipeline"),
        patch("bayesflow_hpo.api._build_objective"),
        patch(
            "bayesflow_hpo.api._derive_directions",
            return_value=(["minimize"], ["m"]),
        ),
        patch("bayesflow_hpo.api._create_and_run_study") as run,
    ):
        optimize(
            simulator=MagicMock(), adapter=MagicMock(), search_space=MagicMock(),
            storage=None, sampler_seed=77, constant_liar=True,
        )
    assert run.call_args.kwargs["sampler_seed"] == 77
    assert run.call_args.kwargs["constant_liar"] is True

@pytest.mark.parametrize("seed", [-1, 2**32, 1.5])
@pytest.mark.parametrize(
    ("sampler", "qmc_startup_trials"),
    [(None, 0), ("tpe", 0), ("random", 0),
     (optuna.samplers.RandomSampler(seed=1), 8)],
)
def test_invalid_seed_preserves_existing_study(
    tmp_path, seed, sampler, qmc_startup_trials,
):
    from bayesflow_hpo.api import _create_and_run_study

    storage = f"sqlite:///{tmp_path / 'preserve.db'}"
    study = create_study(
        study_name="preserve", storage=storage, directions=["minimize"],
    )
    study.optimize(lambda trial: 0.5, n_trials=1)
    with pytest.raises((ValueError, TypeError)):
        _create_and_run_study(
            objective=MagicMock(), study_name="preserve", directions=["minimize"],
            metric_names=["m"], storage=storage, resume=False, sampler=sampler,
            sampler_seed=seed, qmc_startup_trials=qmc_startup_trials,
            n_trials=0, warm_start_from=None,
            warm_start_top_k=25, max_total_trials=None, show_progress_bar=False,
        )
    restored = optuna.load_study(study_name="preserve", storage=storage)
    assert len(restored.trials) == 1
    assert restored.trials[0].value == 0.5


def test_custom_sampler_ignores_invalid_preset_seed():
    from bayesflow_hpo.api import _create_and_run_study

    sampler = optuna.samplers.RandomSampler(seed=1)
    study = _create_and_run_study(
        objective=MagicMock(), study_name="custom", directions=["minimize"],
        metric_names=["m"], storage=None, resume=False, sampler=sampler,
        sampler_seed=-1, n_trials=0, warm_start_from=None,
        warm_start_top_k=25, max_total_trials=None, show_progress_bar=False,
    )
    assert study.sampler is sampler


@pytest.mark.parametrize("seed", [None, 7, 123])
def test_worker_seed_reaches_qmc_warmup(seed):
    from bayesflow_hpo.optimization.study import QMCWarmupSampler

    with patch(
        "bayesflow_hpo.optimization.study.QMCWarmupSampler",
        wraps=QMCWarmupSampler,
    ) as wrapper:
        create_study(storage=None, qmc_startup_trials=8, sampler_seed=seed)
    assert wrapper.call_args.kwargs["seed"] == seed


def test_qmc_worker_seeds_control_independent_draws_and_categorical_ties():
    def draw(seed):
        study = create_study(
            storage=None, directions=["minimize"], qmc_startup_trials=8,
            sampler_seed=seed,
        )
        study.add_trial(optuna.trial.create_trial(
            value=0.0, params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
        ))
        trial = study.ask()
        return (
            trial.suggest_float("x", 0, 1),
            trial.suggest_categorical("family", list("abcdef")),
            trial.suggest_float("conditional", 0, 1),
        )

    first = draw(7)
    second = draw(123)
    assert draw(7) == first
    assert first[0] == second[0]  # The unscrambled Sobol sequence stays shared.
    assert first[1] != second[1]
    assert first[2] != second[2]
