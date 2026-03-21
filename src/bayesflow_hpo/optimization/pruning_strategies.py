"""Multi-objective pruning strategies for BayesFlow HPO.

Optuna does not support ``trial.report()`` for multi-objective studies
(Issue #3450, open since April 2022).  This module provides three
custom pruning strategies that operate on per-metric user attributes
stored by :class:`PeriodicValidationCallback
<bayesflow_hpo.optimization.validation_callback.PeriodicValidationCallback>`.

Four strategies are available via ``pruning_strategy`` in ``optimize()``:

- ``"none"`` — disable intermediate pruning entirely (no callback created).
- ``"dominance"`` — per-objective normalized median check (AND rule).
  Prunes only if the trial is worse than the median on ALL objectives.
  Simplified adaptation of MO-ASHA's dominance-based promotion
  (Schmucker et al., 2021, Algorithm 1).
- ``"mo-sha"`` — non-dominated sorting at each step; prunes trials in
  the bottom fraction per MO-ASHA Algorithm 2 (Schmucker et al., 2021).
- ``"primary"`` — single-metric median pruning on a user-chosen
  objective, equivalent to Optuna's MedianPruner (Akiba et al., 2019).

References
----------
Schmucker, R., Donini, M., Zafar, M. B., Salinas, D., & Archambeau, C.
    (2021). Multi-objective asynchronous successive halving. *arXiv
    preprint*. https://doi.org/10.48550/arxiv.2106.12639

Emmerich, M. T. M., & Deutz, A. H. (2018). A tutorial on
    multiobjective optimization: Fundamentals and evolutionary methods.
    *Natural Computing*, *17*(3), 585--609.
    https://doi.org/10.1007/s11047-018-9685-y

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and
    elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions
    on Evolutionary Computation*, *6*(2), 182--197.
    https://doi.org/10.1109/4235.996017

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).
    Optuna: A next-generation hyperparameter optimization framework.
    In *Proc. 25th ACM SIGKDD* (pp. 2623--2631).
    https://doi.org/10.1145/3292500.3330701
"""

from __future__ import annotations

import numpy as np
import optuna


def _any_non_finite(values) -> bool:
    """Return True if any value is NaN or Inf."""
    return any(not np.isfinite(v) for v in values)


def should_prune_dominance(
    trial: optuna.Trial,
    scores: dict[str, float],
    step: int,
    n_startup_trials: int,
) -> bool:
    """Per-objective normalized median check with AND rule.

    Gathers per-metric user attributes (``val_{metric}_step_{N}``) from
    completed non-rejected trials.  Normalizes each metric to [0, 1]
    using the observed range to eliminate scale sensitivity, then prunes
    only if the current trial is **worse than the median on ALL
    objectives**.

    This is a simplified adaptation of MO-ASHA's dominance-based
    promotion rule (Schmucker et al., 2021, Algorithm 1).  Range
    normalization addresses the scale sensitivity of scalarization
    approaches: Emmerich & Deutz (2018, Proposition 9) show that linear
    scalarization can only find solutions on convex Pareto fronts;
    Schmucker et al. (2021, Section 6) confirm empirically that
    scalarization "tends to penalize one objective heavier than the
    other" while "globally informed techniques are more robust towards
    objectives of different magnitude."

    Parameters
    ----------
    trial
        The running Optuna trial.
    scores
        Current intermediate metric values ``{metric_name: value}``.
    step
        Monotonic step counter (1-indexed).
    n_startup_trials
        Minimum completed reference trials before pruning activates.

    Returns
    -------
    bool
        ``True`` if the trial should be pruned.
    """
    if n_startup_trials < 1:
        return False
    if _any_non_finite(scores.values()):
        return True

    metrics = list(scores.keys())
    ref_vectors = _gather_reference_vectors(trial, metrics, step)

    if len(ref_vectors) < n_startup_trials:
        return False

    ref_array = np.asarray(ref_vectors)  # (N, M)
    current = np.asarray([scores[m] for m in metrics])  # (M,)

    # Normalize to [0, 1] using observed range per metric.
    # Degenerate range (all same value) uses 1.0 to skip normalization.
    mins = ref_array.min(axis=0)
    ranges = ref_array.max(axis=0) - mins
    safe_ranges = np.where(ranges > 0, ranges, 1.0)

    norm_ref = (ref_array - mins) / safe_ranges
    # Current trial may fall outside [0, 1] — that is correct:
    # < 0 means better than all refs (won't exceed median),
    # > 1 means worse than all refs (will exceed median).
    norm_current = (current - mins) / safe_ranges
    medians = np.median(norm_ref, axis=0)

    # AND rule: prune only if worse than median on ALL objectives.
    return bool(np.all(norm_current > medians))


def should_prune_mo_sha(
    trial: optuna.Trial,
    scores: dict[str, float],
    step: int,
    n_startup_trials: int,
    reduction_factor: int = 3,
) -> bool:
    """Non-dominated sorting + bottom-fraction pruning (MO-ASHA).

    Implements the candidate selection from MO-ASHA Algorithm 2
    (Schmucker et al., 2021): at each rung, select the top
    ``|rung| / eta`` configurations using non-dominated sorting
    (Algorithm 1, ``mo_selector(rung k, |rung k| / eta)``).  Prunes
    trials that fall outside the selected set.

    The default ``reduction_factor=3`` (eta) follows MO-ASHA Algorithm 2
    header ("Data: R, r0, s, eta (default eta = 3)") and Li et al.
    (2018, Section 3.6: "in practice we suggest taking eta to be equal
    to 3 or 4").

    Parameters
    ----------
    trial
        The running Optuna trial.
    scores
        Current intermediate metric values ``{metric_name: value}``.
    step
        Monotonic step counter (1-indexed).
    n_startup_trials
        Minimum completed reference trials before pruning activates.
    reduction_factor
        Fraction denominator for survivor selection (eta).  Default 3.

    Returns
    -------
    bool
        ``True`` if the trial should be pruned.
    """
    if n_startup_trials < 1:
        return False
    if _any_non_finite(scores.values()):
        return True

    metrics = list(scores.keys())
    ref_vectors = _gather_reference_vectors(trial, metrics, step)

    if len(ref_vectors) < n_startup_trials:
        return False

    # Build the full set: reference trials + current trial.
    current = [scores[m] for m in metrics]
    all_vectors = ref_vectors + [current]
    current_idx = len(all_vectors) - 1

    # Select top |all| / eta via non-dominated sorting.
    n_select = max(1, len(all_vectors) // reduction_factor)
    fronts = _non_dominated_sort(np.asarray(all_vectors))

    selected: set[int] = set()
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.update(front)
        else:
            remaining = n_select - len(selected)
            selected.update(front[:remaining])
            break

    return current_idx not in selected


def should_prune_primary(
    trial: optuna.Trial,
    score: float,
    metric: str,
    step: int,
    n_startup_trials: int,
) -> bool:
    """Single-metric median pruning.

    Equivalent to Optuna's ``MedianPruner`` (Akiba et al., 2019)
    applied to a single user-chosen objective.  Reads per-metric user
    attributes (``val_{metric}_step_{N}``) from completed trials.

    Parameters
    ----------
    trial
        The running Optuna trial.
    score
        Current intermediate value for the chosen metric.
    metric
        Name of the primary metric (used to look up
        ``val_{metric}_step_{N}`` attrs on completed trials).
    step
        Monotonic step counter (1-indexed).
    n_startup_trials
        Minimum completed reference trials before pruning activates.

    Returns
    -------
    bool
        ``True`` if the trial should be pruned.
    """
    if n_startup_trials < 1:
        return False

    if not np.isfinite(score):
        return True

    ref_vectors = _gather_reference_vectors(trial, [metric], step)

    if len(ref_vectors) < n_startup_trials:
        return False

    ref_scores = [row[0] for row in ref_vectors]
    median_score = float(np.median(ref_scores))
    return bool(score > median_score)


def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Iteratively extract non-dominated fronts from an objective array.

    Implements the O(MN^2) fast non-dominated sorting from NSGA-II
    (Deb et al., 2002), confirmed via MO-ASHA Algorithm 1 lines 1--6
    (Schmucker et al., 2021): iteratively extract non-dominated fronts
    F1, ..., Fm by removing dominated points.

    All objectives are assumed to be **minimized**.

    Parameters
    ----------
    objectives
        (N, M) array where N is the number of solutions and M the
        number of objectives.

    Returns
    -------
    list[list[int]]
        List of fronts, each front being a list of row indices.
        ``fronts[0]`` is the Pareto-optimal front.
    """
    n = objectives.shape[0]
    remaining = set(range(n))
    fronts: list[list[int]] = []

    while remaining:
        front: list[int] = []
        remaining_list = sorted(remaining)

        for i in remaining_list:
            dominated = False
            for j in remaining_list:
                if i == j:
                    continue
                # j dominates i if j <= i on all objectives and j < i on at least one.
                if np.all(objectives[j] <= objectives[i]) and np.any(
                    objectives[j] < objectives[i]
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        fronts.append(front)
        remaining -= set(front)

    return fronts


def _gather_reference_vectors(
    trial: optuna.Trial,
    metrics: list[str],
    step: int,
) -> list[list[float]]:
    """Gather per-metric scores from completed non-rejected trials.

    Reads ``val_{metric}_step_{step}`` user attributes.  Trials missing
    any metric attr (e.g., old-schema trials with only
    ``val_score_step_*``) are silently skipped.

    Parameters
    ----------
    trial
        The running trial (used to access ``trial.study``).
    metrics
        Ordered list of metric names to gather.
    step
        Step index.

    Returns
    -------
    list[list[float]]
        Each inner list is ``[metric_1_value, ..., metric_M_value]``
        for one completed trial.  Only trials with all metrics present
        and finite are included.
    """
    vectors: list[list[float]] = []

    for t in trial.study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    ):
        if "rejected_reason" in t.user_attrs:
            continue
        if t.number == trial.number:
            continue

        row = _extract_metric_row(t.user_attrs, metrics, step)
        if row is not None:
            vectors.append(row)

    return vectors


def _extract_metric_row(
    user_attrs: dict,
    metrics: list[str],
    step: int,
) -> list[float] | None:
    """Extract a finite metric vector from trial user attributes.

    Returns ``None`` if any metric is missing or non-finite.
    """
    row: list[float] = []
    for m in metrics:
        val = user_attrs.get(f"val_{m}_step_{step}")
        if val is None:
            return None
        fval = float(val)
        if not np.isfinite(fval):
            return None
        row.append(fval)
    return row
