"""Optuna study creation, resume, warm-start, and trial-counting helpers.

This module manages the Optuna study lifecycle.  Key design decisions:

- **Budget-aware sampling**: The TPE sampler receives a ``constraints_func``
  that marks budget-rejected trials as infeasible, teaching it to avoid
  oversized configurations even during startup.

- **Non-rejected trial counting**: ``optimize_until()`` counts *trained*
  trials (not including budget-rejected ones) toward ``n_trials``, because
  budget rejections are essentially free (no GPU time).  A separate hard
  cap prevents infinite loops when the entire search space is infeasible.

- **Warm-start**: Seeding a new study from a previous one lets the sampler
  skip the initial random exploration phase, which is valuable when the
  search space changes slightly between experiments.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import optuna
from optuna.trial import TrialState

from bayesflow_hpo.objectives import mean_objective_score
from bayesflow_hpo.optimization.constraints import MetricConstraintSpec
from bayesflow_hpo.results.extraction import _objective_column_names

logger = logging.getLogger(__name__)


def _is_power_of_two(n: int) -> bool:
    """Return ``True`` if *n* is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0

# Default SQLite storage file used by :func:`create_study`.
DEFAULT_STORAGE = "sqlite:///bayesflow_hpo.db"


_PRE_TRAINING_REJECTIONS = {
    "memory_budget",
    "param_budget",
    "build_failed",
    "compile_failed",
    "param_probe_failed",
}


def _make_constraints_func(
    budget_aware: bool = True,
    soft_thresholds: list[MetricConstraintSpec] | None = None,
) -> Callable[[optuna.trial.FrozenTrial], list[float]]:
    """Build a composed Optuna constraints function.

    The returned function emits:

    - index 0: budget rejection flag (when ``budget_aware`` is True)
    - subsequent indices: soft metric-threshold violations
    """
    thresholds = list(soft_thresholds or [])

    def _constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
        values: list[float] = []

        if budget_aware:
            rejected_reason = trial.user_attrs.get("rejected_reason")
            values.append(
                1.0 if rejected_reason in _PRE_TRAINING_REJECTIONS else 0.0
            )

        for metric, threshold, direction in thresholds:
            raw = trial.user_attrs.get(metric)
            if raw is None:
                values.append(0.0)
                continue
            metric_value = float(raw)
            if direction == "above":
                values.append(max(0.0, metric_value - float(threshold)))
            elif direction == "below":
                values.append(max(0.0, float(threshold) - metric_value))
            else:
                raise ValueError(
                    f"Unsupported constraint direction {direction!r} for metric "
                    f"{metric!r}; expected 'above' or 'below'."
                )
        return values

    return _constraints


def _mean_ranking_key(trial: optuna.trial.FrozenTrial) -> float:
    """Rank by the mean of objective values (excluding cost score).

    Falls back to the first objective value when multi-objective values
    are not available.
    """
    if trial.values:
        return mean_objective_score(trial.values)
    return float("inf")


def _resolve_pruner(name: str) -> optuna.pruners.BasePruner:
    """Resolve a string preset to an Optuna pruner instance.

    Parameters
    ----------
    name
        One of ``"median"``, ``"hyperband"``, or ``"none"``.

    Returns
    -------
    optuna.pruners.BasePruner

    Raises
    ------
    ValueError
        If *name* is not a recognized preset.

    References
    ----------
    Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A.
        (2018). Hyperband: A novel bandit-based approach to hyperparameter
        optimization. *JMLR*, *18*(185), 1--52.
        Section 3.6: η=3 convention.
    Akiba, T., et al. (2019). Optuna. *Proc. 25th ACM SIGKDD*.
    """
    presets = {
        "median": lambda: optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1,
        ),
        "hyperband": lambda: optuna.pruners.HyperbandPruner(
            min_resource=1,
            reduction_factor=3,
        ),
        "none": lambda: optuna.pruners.NopPruner(),
    }
    if name not in presets:
        raise ValueError(
            f"Unknown pruner preset: {name!r}. "
            f"Expected one of {sorted(presets)}."
        )
    return presets[name]()


def _resolve_sampler(
    name: str,
    budget_aware: bool = True,
    soft_thresholds: list[MetricConstraintSpec] | None = None,
) -> optuna.samplers.BaseSampler:
    """Resolve a string preset to a configured Optuna sampler.

    All presets that accept ``constraints_func`` auto-wire the budget
    constraint when *budget_aware* is ``True``, teaching the sampler to
    avoid oversized configurations.

    Parameters
    ----------
    name
        One of ``"tpe"``, ``"gp"``, ``"botorch"``, ``"nsga2"``,
        ``"nsga3"``, ``"auto"``, or ``"random"``.
    budget_aware
        Whether to include budget rejection in the composed
        constraints function for samplers that support it.

    Returns
    -------
    optuna.samplers.BaseSampler

    Raises
    ------
    ValueError
        If *name* is not a recognized preset.
    ImportError
        If *name* requires an optional dependency that is not installed
        (``"botorch"`` needs ``optuna-integration[botorch]``;
        ``"auto"`` needs a newer Optuna version).

    References
    ----------
    Bergstra, J., et al. (2011). Algorithms for hyper-parameter
        optimization. *NeurIPS 24*.
    Balandat, M., et al. (2020). BoTorch. *NeurIPS 33*.
    Deb, K., et al. (2002). NSGA-II. *IEEE TEVC*, *6*(2), 182--197.
    Deb, K., & Jain, H. (2014). NSGA-III. *IEEE TEVC*, *18*(4), 577--601.
    """
    constraints = (
        _make_constraints_func(
            budget_aware=budget_aware,
            soft_thresholds=soft_thresholds,
        )
        if budget_aware or soft_thresholds
        else None
    )

    def _make_tpe() -> optuna.samplers.TPESampler:
        return optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
            n_startup_trials=25,
            warn_independent_sampling=False,
            constraints_func=constraints,
        )

    def _make_gp() -> optuna.samplers.GPSampler:
        return optuna.samplers.GPSampler(
            seed=42,
            n_startup_trials=10,
            constraints_func=constraints,
        )

    def _make_botorch() -> optuna.samplers.BaseSampler:
        try:
            from optuna.integration import BoTorchSampler
        except ImportError:
            raise ImportError(
                'Sampler preset "botorch" requires optuna-integration[botorch]. '
                "Install with: pip install optuna-integration[botorch]"
            ) from None
        return BoTorchSampler(
            seed=42,
            n_startup_trials=10,
            constraints_func=constraints,
        )

    def _make_nsga2() -> optuna.samplers.NSGAIISampler:
        return optuna.samplers.NSGAIISampler(
            population_size=50,
            seed=42,
            constraints_func=constraints,
        )

    def _make_nsga3() -> optuna.samplers.NSGAIIISampler:
        return optuna.samplers.NSGAIIISampler(
            population_size=50,
            seed=42,
            constraints_func=constraints,
        )

    def _make_auto() -> optuna.samplers.BaseSampler:
        try:
            from optuna.samplers import AutoSampler  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError(
                'Sampler preset "auto" requires a newer version of Optuna '
                "that provides AutoSampler. It is not available in "
                f"optuna {optuna.__version__}."
            ) from None
        return AutoSampler(seed=42)

    def _make_random() -> optuna.samplers.RandomSampler:
        return optuna.samplers.RandomSampler(seed=42)

    presets: dict[str, Callable[[], optuna.samplers.BaseSampler]] = {
        "tpe": _make_tpe,
        "gp": _make_gp,
        "botorch": _make_botorch,
        "nsga2": _make_nsga2,
        "nsga3": _make_nsga3,
        "auto": _make_auto,
        "random": _make_random,
    }
    if name not in presets:
        raise ValueError(
            f"Unknown sampler preset: {name!r}. "
            f"Expected one of {sorted(presets)}."
        )
    return presets[name]()


def _resolve_n_startup_trials(sampler: optuna.samplers.BaseSampler) -> int:
    """Infer the number of startup trials from a sampler instance.

    Checks ``n_startup_trials`` first (TPE, GP, BoTorch), then
    ``population_size`` (NSGA-II, NSGA-III), falling back to 10.

    Parameters
    ----------
    sampler
        An Optuna sampler instance.

    Returns
    -------
    int
        The resolved startup trial count.
    """
    # Public attribute (future Optuna versions may expose it).
    n = getattr(sampler, "n_startup_trials", None)
    if n is not None:
        return int(n)
    # Private attribute (TPE, GP in Optuna 4.x store it as _n_startup_trials).
    n = getattr(sampler, "_n_startup_trials", None)
    if n is not None:
        return int(n)
    # NSGA-II/III expose population_size as a public property.
    n = getattr(sampler, "population_size", None)
    if n is not None:
        return int(n)
    return 10


class QMCWarmupSampler(optuna.samplers.BaseSampler):
    """Composite sampler that uses QMC (Sobol) for the first N trials.

    Wraps a main sampler and delegates to ``QMCSampler`` for the first
    *qmc_startup_trials* non-rejected completions, then transparently
    delegates all calls to the main sampler.

    This is an internal class — the public API is the
    ``qmc_startup_trials`` parameter on :func:`create_study` and
    :func:`~bayesflow_hpo.optimize`.

    Parameters
    ----------
    main_sampler
        The sampler to use after the QMC warm-up phase.
    qmc_startup_trials
        Number of non-rejected QMC trials before switching.

    Raises
    ------
    ValueError
        If *qmc_startup_trials* is negative.

    References
    ----------
    Sobol', I. M. (1967). On the distribution of points in a cube and the
        approximate evaluation of integrals. *USSR Computational Mathematics
        and Mathematical Physics*, *7*(4), 86-112.
        https://doi.org/10.1016/0041-5553(67)90144-9

    Joe, S., & Kuo, F. Y. (2008). Constructing Sobol sequences with
        better two-dimensional projections. *SIAM Journal on Scientific
        Computing*, *30*(5), 2635-2654.
        https://doi.org/10.1137/070709359

    Optuna PR #2423: Sobol outperforms Halton in benchmarks.
    Optuna Issue #1797: QMCSampler significantly better than RandomSampler.
    """

    def __init__(
        self,
        main_sampler: optuna.samplers.BaseSampler,
        qmc_startup_trials: int,
    ) -> None:
        if qmc_startup_trials < 0:
            raise ValueError(
                f"qmc_startup_trials must be >= 0, got {qmc_startup_trials}"
            )
        self._main_sampler = main_sampler
        self._qmc_startup_trials = qmc_startup_trials
        self._qmc_sampler = optuna.samplers.QMCSampler(
            qmc_type="sobol",
            scramble=False,
        )
        self._n_qmc_completed: int = 0
        self._pending_qmc_trials: set[int] = set()

    @property
    def _is_qmc_phase(self) -> bool:
        """Whether we are still in the QMC warm-up phase."""
        return self._n_qmc_completed < self._qmc_startup_trials

    @property
    def _active_sampler(self) -> optuna.samplers.BaseSampler:
        """The sampler that should handle the current trial."""
        if self._is_qmc_phase:
            return self._qmc_sampler
        return self._main_sampler

    @property
    def n_startup_trials(self) -> int:
        """Startup trials for pruning warmup alignment.

        Returns the maximum of the QMC quota and the main sampler's
        startup count, so pruning does not activate prematurely.
        """
        main_startup = _resolve_n_startup_trials(self._main_sampler)
        return max(self._qmc_startup_trials, main_startup)

    def infer_relative_search_space(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        return self._active_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        if self._is_qmc_phase:
            self._pending_qmc_trials.add(trial.number)
        return self._active_sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        if self._is_qmc_phase:
            self._pending_qmc_trials.add(trial.number)
        return self._active_sampler.sample_independent(
            study, trial, param_name, param_distribution,
        )

    def before_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        self._active_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        # BaseSampler promises Sequence[float]; narrowing to list would break
        # on any sequence Optuna chooses to pass.
        values: Sequence[float] | None,
    ) -> None:
        was_qmc_trial = trial.number in self._pending_qmc_trials
        if was_qmc_trial:
            self._pending_qmc_trials.discard(trial.number)
            if (
                state == TrialState.COMPLETE
                and "rejected_reason" not in trial.user_attrs
            ):
                self._n_qmc_completed += 1
        # Delegate to the sampler that actually produced this trial,
        # not _active_sampler (which may have flipped mid-increment).
        delegate = self._qmc_sampler if was_qmc_trial else self._main_sampler
        delegate.after_trial(study, trial, state, values)


def create_study(
    study_name: str = "bayesflow_hpo",
    directions: list[str] | None = None,
    metric_names: list[str] | None = None,
    storage: str | None = DEFAULT_STORAGE,
    load_if_exists: bool = True,
    sampler: str | optuna.samplers.BaseSampler | None = None,
    pruner: str | optuna.pruners.BasePruner | None = None,
    warm_start_from: optuna.Study | None = None,
    warm_start_top_k: int = 25,
    budget_aware: bool = True,
    metric_constraints_soft: list[MetricConstraintSpec] | None = None,
    qmc_startup_trials: int = 0,
) -> optuna.Study:
    """Create or resume an Optuna study.

    Parameters
    ----------
    study_name
        Optuna study name (default ``"bayesflow_hpo"``).
    directions
        Optimization directions.  Default ``["minimize", "minimize"]``.
        The caller is responsible for passing the correct number of
        directions matching the objective shape.
    metric_names
        Human-readable names for each objective.
    storage
        Optuna storage URL.  Default ``"sqlite:///bayesflow_hpo.db"``
        for automatic persistence and crash recovery.  Pass ``None``
        for in-memory only.
    load_if_exists
        Resume a study with the same name if it already exists.
    sampler
        Optuna sampler.  Accepts a ``BaseSampler`` instance, a string
        preset, or ``None`` (default ``"tpe"``).

        ============= =================================================
        Preset        Sampler
        ============= =================================================
        ``"tpe"``     ``TPESampler(multivariate=True,
                      n_startup_trials=25)``
        ``"gp"``      ``GPSampler(n_startup_trials=10)``
        ``"botorch"`` ``BoTorchSampler(n_startup_trials=10)`` |br|
                      Requires ``optuna-integration[botorch]``.
        ``"nsga2"``   ``NSGAIISampler(population_size=50)``
        ``"nsga3"``   ``NSGAIIISampler(population_size=50)``
        ``"auto"``    ``AutoSampler()`` |br|
                      Requires a newer Optuna version.
        ``"random"``  ``RandomSampler()``
        ============= =================================================

        All presets use ``seed=42``.  Presets that support
        ``constraints_func`` auto-wire budget constraints when
        ``budget_aware=True``.
    pruner
        Optuna pruner for single-objective studies.  Accepts a
        ``BasePruner`` instance, a string preset, or ``None``
        (default ``MedianPruner``).

        =========== ================================================
        Preset      Pruner
        =========== ================================================
        ``"median"``    ``MedianPruner(n_startup_trials=5,
                        n_warmup_steps=1, interval_steps=1)``
        ``"hyperband"`` ``HyperbandPruner(min_resource=1,
                        reduction_factor=3)``
        ``"none"``      ``NopPruner()``
        =========== ================================================

        ``"hyperband"`` outperforms ``"median"`` with TPE per Optuna
        benchmarks (Li et al., 2018, Section 3.6: η=3 convention).

        **Single-objective only.**  This pruner is consulted via
        ``trial.should_prune()`` only in single-objective studies.
        In multi-objective studies, Optuna does not support
        ``trial.report()`` so this parameter is ignored; pruning is
        handled by the ``pruning_strategy`` parameter on
        ``optimize()``.
    warm_start_from
        Optional source study to seed initial trials from.
    warm_start_top_k
        Number of best trials to copy from the source study.
    budget_aware
        Whether to attach a constraints function that marks
        budget-rejected trials as infeasible for the sampler.
    metric_constraints_soft
        Optional soft metric thresholds passed to sampler presets via
        ``constraints_func``. Supported samplers treat positive values
        as constraint violations and bias sampling away from them. When
        a user-supplied sampler instance is passed, this parameter is
        ignored because sampler internals cannot be patched safely.
    qmc_startup_trials
        Number of initial trials to sample with a Sobol quasi-random
        sequence before the main sampler takes over.  Sobol provides
        better space-filling coverage than random startup, giving
        the main sampler a more informative initial dataset.

        Only non-rejected completions count toward this quota.
        When 0 (default), no QMC wrapper is applied.

        Sobol's low-discrepancy guarantee is optimal at
        n = 2^m; a warning is logged for non-power-of-2 values.

    Raises
    ------
    ValueError
        If *qmc_startup_trials* is negative.
    """
    if directions is None:
        directions = ["minimize", "minimize"]

    if isinstance(sampler, str):
        sampler = _resolve_sampler(
            sampler,
            budget_aware=budget_aware,
            soft_thresholds=metric_constraints_soft,
        )
    elif sampler is None:
        sampler = _resolve_sampler(
            "tpe",
            budget_aware=budget_aware,
            soft_thresholds=metric_constraints_soft,
        )

    if qmc_startup_trials < 0:
        raise ValueError(
            f"qmc_startup_trials must be >= 0, got {qmc_startup_trials}"
        )
    if qmc_startup_trials > 0:
        if not _is_power_of_two(qmc_startup_trials):
            # Sobol' (1967): optimal discrepancy at 2^m points
            logger.warning(
                "qmc_startup_trials=%d is not a power of 2. Sobol's "
                "low-discrepancy guarantee is optimal at n = 2^m "
                "(e.g. 8, 16, 32). Proceeding with the requested value.",
                qmc_startup_trials,
            )
        sampler = QMCWarmupSampler(sampler, qmc_startup_trials)

    if isinstance(pruner, str):
        pruner = _resolve_pruner(pruner)
    elif pruner is None:
        pruner = _resolve_pruner("median")

    create_kwargs: dict[str, Any] = dict(
        study_name=study_name,
        directions=directions,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )
    study = optuna.create_study(**create_kwargs)
    # `metric_names` is NOT a `create_study` parameter -- the guard that used
    # to test for one was never true on any Optuna release, so the assignment
    # below it always ran and set a private, in-memory-only attribute. The
    # real API is `Study.set_metric_names` (Optuna >= 3.2), which persists to
    # storage and survives a reload, so column provenance is recoverable
    # rather than lost the moment the process exits.
    if metric_names:
        try:
            with warnings.catch_warnings():
                # set_metric_names is flagged experimental; the alternative is
                # no persisted provenance at all.
                warnings.simplefilter("ignore")
                study.set_metric_names(list(metric_names))
        except (AttributeError, RuntimeError) as exc:  # pragma: no cover
            logger.debug("Could not persist metric names: %s", exc)
            study._metric_names = metric_names  # type: ignore[attr-defined]
    # A warm start that will copy nothing cannot import a foreign schema,
    # so neither the check nor the encoding propagation below applies:
    # `warm_start_top_k=0` is explicitly supported by `max(0, int(top_k))`,
    # and a source with no COMPLETE trials has nothing to give either.
    will_copy = warm_start_top_k > 0 and warm_start_from is not None and any(
        t.state == TrialState.COMPLETE and t.values is not None
        for t in warm_start_from.trials
    )
    if warm_start_from is not None and will_copy and len(study.trials) == 0:
        # Provenance is validated BEFORE any trial is copied. Copied values
        # carry both their encoding and the metric behind each column, so a
        # target that inherits the encoding alone looks verified while its
        # columns may mean something else entirely: the resume guard would
        # then find no schema, record the *requested* names, and treat the
        # copied values as though they had always represented those metrics.
        source_schema = warm_start_from.user_attrs.get(
            "bayesflow_hpo_objective_schema"
        )
        if isinstance(source_schema, (list, tuple)):
            source_schema = list(source_schema)
        else:
            source_schema = None
        if (
            source_schema is not None
            and metric_names is not None
            and source_schema != list(metric_names)
        ):
            raise ValueError(
                f"Cannot warm-start from study "
                f"{warm_start_from.study_name!r}: it stores objectives "
                f"{source_schema!r}, but this run produces "
                f"{list(metric_names)!r}. Its trials would be copied into "
                "columns that mean something else."
            )

        warm_start_study(
            target_study=study,
            source_study=warm_start_from,
            top_k=warm_start_top_k,
        )
        # Without this the target holds COMPLETE trials and no provenance --
        # the exact signature of a legacy study -- so the resume guard would
        # reject a perfectly valid warm start from an already-re-encoded one.
        source_encoding = warm_start_from.user_attrs.get(
            "bayesflow_hpo_objective_encoding"
        )
        if source_encoding is not None:
            study.set_user_attr(
                "bayesflow_hpo_objective_encoding", source_encoding
            )
        if source_schema is not None:
            study.set_user_attr(
                "bayesflow_hpo_objective_schema", source_schema
            )
    return study


def resume_study(study_name: str, storage: str) -> optuna.Study:
    """Resume a persisted study."""
    return create_study(study_name=study_name, storage=storage, load_if_exists=True)


def warm_start_study(
    target_study: optuna.Study,
    source_study: optuna.Study,
    top_k: int = 25,
) -> int:
    """Seed *target_study* with best completed trials from *source_study*.

    Trials are ranked by the arithmetic mean of their objective values
    (excluding cost score), falling back to the first objective when
    only a single value is available.

    Parameters
    ----------
    target_study
        Study to seed.
    source_study
        Study to copy trials from.
    top_k
        Maximum number of trials to copy.

    Returns
    -------
    int
        Number of trials actually added.
    """
    complete_trials = [
        trial
        for trial in source_study.trials
        if trial.state == TrialState.COMPLETE and trial.values is not None
    ]
    if not complete_trials:
        return 0

    ranked = sorted(complete_trials, key=_mean_ranking_key)

    added = 0
    for trial in ranked[: max(0, int(top_k))]:
        seeded_trial = optuna.trial.create_trial(
            params=trial.params,
            distributions=trial.distributions,
            values=trial.values,
            intermediate_values=trial.intermediate_values,
            user_attrs=trial.user_attrs,
            system_attrs=trial.system_attrs,
            state=TrialState.COMPLETE,
        )
        target_study.add_trial(seeded_trial)
        added += 1

    return added


def count_trained_trials(study: optuna.Study) -> int:
    """Count completed trials that were not rejected by budget checks."""
    return sum(
        1
        for t in study.trials
        if t.state == TrialState.COMPLETE
        and "rejected_reason" not in t.user_attrs
    )


def _count_pruned(study: optuna.Study, since_trial: int = 0) -> int:
    """Count trials pruned by intermediate validation since a given trial number."""
    return sum(
        1
        for t in study.trials
        if t.number >= since_trial and t.state == TrialState.PRUNED
    )


def _count_budget_rejected(study: optuna.Study, since_trial: int = 0) -> int:
    """Count trials rejected by budget checks since a given trial number."""
    return sum(
        1
        for t in study.trials
        if (
            t.number >= since_trial
            and t.user_attrs.get("rejected_reason") in _PRE_TRAINING_REJECTIONS
        )
    )


def _count_failed(study: optuna.Study, since_trial: int = 0) -> int:
    """Count trials that crashed during training since a given trial number."""
    return sum(
        1
        for t in study.trials
        if t.number >= since_trial and t.state == TrialState.FAIL
    )


def _count_failure_reasons(
    study: optuna.Study,
    since_trial: int = 0,
) -> dict[str, int]:
    """Count training errors and rejection reasons for recent trials.

    Groups identical error messages (truncated to 80 chars) so that
    the progress log can detect systemic issues vs. random failures.
    """
    counts: dict[str, int] = {}
    for t in study.trials:
        if t.number < since_trial:
            continue
        reason = t.user_attrs.get("rejected_reason")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
        error = t.user_attrs.get("training_error")
        if error:
            # Group identical error messages.
            key = f"error: {error[:80]}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _best_objective_so_far(
    study: optuna.Study,
    select_by: int = 0,
) -> float | None:
    """Return the best value for the selected objective across trained trials.

    Only considers completed, non-rejected trials.  Returns ``None`` if
    no qualifying trials exist yet.
    """
    best = None
    for t in study.trials:
        if (
            t.state == TrialState.COMPLETE
            and t.values is not None
            and "rejected_reason" not in t.user_attrs
        ):
            val = t.values[select_by]
            if best is None or val < best:
                best = val
    return best


def _best_trial_so_far(
    study: optuna.Study,
    select_by: int = 0,
) -> optuna.trial.FrozenTrial | None:
    """Return the best trained trial by the selected objective.

    Returns ``None`` if no qualifying trials exist yet.
    """
    best_trial = None
    best_val = None
    for t in study.trials:
        if (
            t.state == TrialState.COMPLETE
            and t.values is not None
            and "rejected_reason" not in t.user_attrs
        ):
            val = t.values[select_by]
            if best_val is None or val < best_val:
                best_val = val
                best_trial = t
    return best_trial


def _count_non_rejected(study: optuna.Study) -> int:
    """Count trials that actually attempted training (trained + pruned + failed)."""
    return sum(
        1
        for t in study.trials
        if t.user_attrs.get("rejected_reason") not in _PRE_TRAINING_REJECTIONS
    )


def optimize_until(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], tuple[float, ...]],
    n_trained: int,
    *,
    max_total_trials: int | None = None,
    show_progress_bar: bool = True,
) -> None:
    """Run trials until *n_trained* have actually trained (not budget-rejected).

    Budget-rejected trials do not count toward ``max_total_trials``
    because they are cheap (no training).  A hard safety cap of
    ``5 * max_total_trials`` on *all* trials (including rejected)
    prevents runaway loops when the search space consistently exceeds
    the parameter budget.

    Parameters
    ----------
    study
        The Optuna study to optimize.
    objective
        The objective callable.
    n_trained
        Target number of trials that pass budget checks and complete training.
    max_total_trials
        Cap on non-rejected trials (trained + pruned + failed).
        Defaults to ``3 * n_trained``.
    show_progress_bar
        Whether to show Optuna's progress bar.
    """
    if max_total_trials is None:
        max_total_trials = 3 * n_trained

    # Hard safety cap on ALL trials (including rejected) to prevent
    # infinite loops when every sampled config exceeds the budget.
    hard_cap = 5 * max_total_trials

    trained_before = count_trained_trials(study)
    target = trained_before + n_trained
    total_before = len(study.trials)
    non_rejected_before = _count_non_rejected(study)

    logger.info(
        "Starting HPO: target %d trained trials "
        "(max %d non-rejected, hard cap %d).\n"
        "  trained  = completed training + validation successfully\n"
        "  rejected = skipped before training (model too large or failed to build)\n"
        "  failed   = crashed during training\n"
        "  pruned   = stopped early by intermediate validation (unpromising)",
        n_trained, max_total_trials, hard_cap,
    )

    def _non_rejected_now() -> int:
        return _count_non_rejected(study) - non_rejected_before

    def _total_now() -> int:
        return len(study.trials) - total_before

    while (
        count_trained_trials(study) < target
        and _non_rejected_now() < max_total_trials
        and _total_now() < hard_cap
    ):
        remaining_trained = target - count_trained_trials(study)
        remaining_non_rejected = max_total_trials - _non_rejected_now()
        # Run in small batches to re-check the trained count regularly.
        batch = min(remaining_trained, remaining_non_rejected, max(1, n_trained // 4))
        study.optimize(
            objective,
            n_trials=batch,
            show_progress_bar=show_progress_bar,
            gc_after_trial=True,
        )

        # --- Live progress summary after each batch ---
        trained_now = count_trained_trials(study) - trained_before
        rejected = _count_budget_rejected(study, since_trial=total_before)
        failed = _count_failed(study, since_trial=total_before)
        pruned = _count_pruned(study, since_trial=total_before)
        parts = [f"{trained_now}/{n_trained} trained"]
        if rejected:
            parts.append(f"{rejected} rejected")
        if failed:
            parts.append(f"{failed} failed")
        if pruned:
            parts.append(f"{pruned} pruned")
        best_trial = _best_trial_so_far(study)
        if best_trial is not None:
            obj_cols = _objective_column_names(study)
            for col, val in zip(obj_cols, best_trial.values):
                parts.append(f"best {col}: {val:.4f}")
        logger.info("Progress: %s", " | ".join(parts))

    # --- Final summary ---
    trained_now = count_trained_trials(study) - trained_before
    rejected = _count_budget_rejected(study, since_trial=total_before)
    failed = _count_failed(study, since_trial=total_before)
    pruned = _count_pruned(study, since_trial=total_before)
    total_now = len(study.trials) - total_before
    if rejected > 0 or failed > 0 or pruned > 0:
        parts = [f"{trained_now} trained"]
        if rejected:
            parts.append(f"{rejected} rejected")
        if failed:
            parts.append(f"{failed} failed")
        if pruned:
            parts.append(f"{pruned} pruned")
        logger.info("Completed %s.", ", ".join(parts))

    # --- Failure reason breakdown ---
    reasons = _count_failure_reasons(study, since_trial=total_before)
    if reasons:
        reason_parts = [f"{reason}: {count}" for reason, count in reasons.items()]
        logger.info("Failure breakdown: %s", " | ".join(reason_parts))
        # Warn if a single reason dominates (signals a systemic issue).
        dominant = max(reasons.values())
        if total_now > 2 and dominant / total_now > 0.5:
            logger.warning(
                "Over half of trials failed for the same reason. "
                "Check the most common failure above — it may indicate "
                "a configuration issue rather than bad hyperparameters.",
            )

    if trained_now < n_trained:
        hint_parts = []
        if reasons:
            hint_parts.append(
                "failure breakdown: "
                + ", ".join(f"{r}: {c}" for r, c in reasons.items())
            )
        if pruned:
            hint_parts.append(f"{pruned} trials were pruned")
        hint = (
            f" ({'; '.join(hint_parts)})"
            if hint_parts
            else ""
        )
        if _total_now() >= hard_cap:
            logger.warning(
                "Hit hard safety cap (%d total trials including rejected). "
                "Most sampled configs are being rejected by budget checks%s. "
                "Consider raising max_param_count or narrowing the search space.",
                hard_cap, hint,
            )
        else:
            logger.warning(
                "Reached max_total_trials=%d before achieving %d trained "
                "trials (got %d)%s. Consider raising max_total_trials, "
                "max_param_count, or adjusting the search space.",
                max_total_trials, n_trained, trained_now, hint,
            )
