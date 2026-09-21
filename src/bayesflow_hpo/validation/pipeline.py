"""Validation pipeline on fixed ``ValidationDataset``."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from bayesflow_hpo.optimization.cleanup import cleanup_trial
from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.inference import (
    DEFAULT_MAX_SAMPLES_PER_CALL,
    make_bayesflow_infer_fn,
)
from bayesflow_hpo.validation.metrics import (
    Aggregate,
    aggregate_condition_rows,
    compute_condition_metrics,
    normalize_aggregate,
    reduce_metric,
    resolve_reduction,
)
from bayesflow_hpo.validation.registry import (
    _JOINT,
    DEFAULT_METRICS,
    JointMetricConfigurationError,
    JointMetricFn,
    JointMetricInputs,
    is_joint_metric,
    joint_metric_settings,
    output_keys_for,
    resolve_joint_metrics,
    resolve_metrics,
)
from bayesflow_hpo.validation.result import ValidationResult

logger = logging.getLogger(__name__)


def _joint_draws(draws: np.ndarray) -> np.ndarray:
    """Return *draws* as the 3-D array :class:`JointMetricInputs` promises.

    Two separate places collapse the trailing axis for a single-parameter
    study, and only one of them is in this module:

    - ``make_bayesflow_infer_fn`` squeezes inside the closure
      (``validation/inference.py:57-61``), BEFORE the pipeline sees the
      array, so no amount of reordering within the loop establishes the
      invariant;
    - ``run_validation_pipeline`` squeezes again in its single-parameter
      branch, which the joint dispatch now runs ahead of.

    A joint metric that asserts rank -- ``compute_tarp_coverage`` raises on
    ``ndim != 3`` -- would therefore fail on EVERY condition of an ordinary
    scalar-parameter study, which under the D8 guard means a metric that
    looks configured and silently never reports. Re-expanding here makes the
    guarantee the joint path's own rather than an assumption about upstream.

    Parameters
    ----------
    draws
        ``(n_sims, n_samples)`` or ``(n_sims, n_samples, n_params)``.

    Returns
    -------
    np.ndarray
        The same values with a trailing parameter axis.

    Raises
    ------
    ValueError
        If *draws* is neither 2-D nor 3-D.
    """
    arr = np.asarray(draws)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3:
        return arr
    raise ValueError(
        "Expected posterior draws with 2 or 3 dimensions, got shape "
        f"{arr.shape}."
    )


def _run_joint_metrics(
    joint_metric_fns: dict[str, Any],
    *,
    draws: np.ndarray,
    sim_batch: dict[str, Any],
    validation_data: ValidationDataset,
    approximator: Any,
    cond_id: int,
    n_conditions: int,
    failed_joint: dict[str, str],
    emitted_keys: dict[str, set[str]],
) -> dict[str, float]:
    """Evaluate joint metrics for one condition, under a per-metric guard.

    The guard exists because the pipeline has no other one: an exception
    from a single metric on a single condition aborts the whole validation,
    so the trial loses its MARGINAL metrics too and drops to the
    training-loss fallback. Joint metrics are precisely the ones with
    optional dependencies and numerical preconditions -- L-C2ST fits a
    classifier per fold, and TARP with ``standardize=True`` raises on a
    constant dimension -- so that blast radius is not acceptable.

    A failure is recorded in *failed_joint* and the metric is dropped from
    the trial entirely by the caller, rather than omitted for this condition
    alone. Omitting one condition would leave the value dependent on which
    condition failed: `aggregate_condition_rows` takes its key set from the
    first row, so a failure on condition 0 discards every later success,
    while a failure on condition 1 reports a mean over the successes and
    never reaches the metric's registered worst case at all.
    """
    param_keys = tuple(validation_data.param_keys)
    true_values = np.stack(
        [np.asarray(sim_batch[pk]).reshape(-1) for pk in param_keys],
        axis=-1,
    )
    inputs = JointMetricInputs(
        draws=_joint_draws(draws),
        true_values=true_values,
        param_keys=param_keys,
        sim_batch=sim_batch,
        data_keys=tuple(validation_data.data_keys),
        approximator=approximator,
        cond_id=cond_id,
        n_conditions=n_conditions,
    )

    row: dict[str, float] = {}
    for name, fn in joint_metric_fns.items():
        if name in failed_joint:
            # Already invalidated for this trial; computing it again would
            # cost the full price for a value that will be discarded.
            continue
        try:
            result = fn(inputs)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            failed_joint[name] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Joint metric %r failed on condition %d and is invalidated "
                "for this trial: %s",
                name,
                cond_id,
                exc,
            )
            continue
        # A NaN is a failure the metric chose not to raise for, and it
        # must take the same path as one that did. Otherwise
        # `_aggregate_joint_rows` nanmeans it away: a metric returning 0.01
        # on one condition and NaN on another reports 0.01, a flattering
        # finite score where D8 promises whole-trial invalidation. An
        # infinity is left alone -- `log_gamma` shows a metric can mean it.
        #
        # Converted FIRST, then inspected. An earlier version tested
        # `isinstance(value, float | int)` before converting, which is not
        # the same set: `np.float64` subclasses Python float but
        # `np.float32` and `np.float16` do not, so a NaN from a float32
        # reduction -- an ordinary result of reducing a float32 array --
        # passed the guard untouched and was then converted to a Python NaN
        # one line later, restoring the exact bug the guard was added for.
        if not isinstance(result, Mapping):
            # `result.items()` below would raise AttributeError, which the
            # conversion handler does not catch -- so it escapes
            # `_run_joint_metrics`, where no outer guard exists, and aborts
            # the whole validation. One metric returning the wrong shape
            # would then discard every marginal result the trial had
            # already computed, which is exactly the blast radius D8's
            # guard exists to contain.
            failed_joint[name] = (
                f"returned {type(result).__name__}, expected a mapping"
            )
            logger.warning(
                "Joint metric %r returned %s instead of a mapping and is "
                "invalidated for this trial.",
                name,
                type(result).__name__,
            )
            continue
        try:
            converted = {key: float(value) for key, value in result.items()}
        except (TypeError, ValueError) as exc:
            failed_joint[name] = (
                f"returned a non-numeric value on condition {cond_id}: {exc}"
            )
            logger.warning(
                "Joint metric %r returned a non-numeric value on condition "
                "%d and is invalidated for this trial: %s",
                name,
                cond_id,
                exc,
            )
            emitted_keys.setdefault(name, set()).update(result)
            continue
        nan_keys = [
            key for key, value in converted.items() if math.isnan(value)
        ]
        if nan_keys:
            failed_joint[name] = (
                f"returned NaN for {sorted(nan_keys)} on condition {cond_id}"
            )
            logger.warning(
                "Joint metric %r returned NaN for %s on condition %d and is "
                "invalidated for this trial.",
                name,
                sorted(nan_keys),
                cond_id,
            )
            emitted_keys.setdefault(name, set()).update(result)
            continue
        row.update(converted)
        # Recorded so that a LATER failure can drop what this condition
        # already contributed. The registry's declared outputs are not
        # enough on their own: a metric passed through `joint_metrics=`
        # carries a caller-chosen mapping key that need not be registered
        # at all, and `output_keys_for` then falls back to that key itself
        # -- which is not what the callable emits, so nothing would be
        # dropped and a partial mean would reach the objective.
        emitted_keys.setdefault(name, set()).update(result)
    return row


#: Reserved key under which the validation run's own counts are pinned.
#:
#: Not a metric name, and deliberately not per-metric: these describe the
#: RUN, and every joint metric in it shares them.
VALIDATION_RUN_SETTINGS = "__validation_run__"


def _declared_settings(
    joint_metric_fns: dict[str, Any],
    n_posterior_samples: int,
    n_conditions: int,
) -> dict[str, dict[str, Any]]:
    """Settings to pin: what each metric declares, plus the run's own counts.

    A joint metric's score moves with the validation run as well as with
    its own configuration, and D7 names both. TARP's coverage fractions are
    supported on ``{0, 1/n_draws, ..., 1}``, so resuming at 5 posterior
    draws instead of 500 puts them on a different grid entirely; the
    condition count moves the Monte Carlo error and, for a subsampled
    L-C2ST, changes WHICH conditions run even at an unchanged
    ``max_conditions``. Neither is derivable from the per-metric
    declarations -- a metric reads ``draws.shape[1]`` rather than being told
    -- so the pipeline contributes them.

    Returns an empty mapping when nothing declares settings, so a study
    using no joint metrics still never acquires the attribute.
    """
    declared = joint_metric_settings(joint_metric_fns)
    # Keyed on whether any joint metric RAN, not on whether one declared
    # settings. A custom joint metric may declare nothing and still depend
    # on `draws.shape[1]` or the condition count -- every joint metric can
    # -- so gating on `declared` left those runs pinning nothing at all, and
    # a resume could change either count undetected.
    if not joint_metric_fns:
        return {}
    return {
        **declared,
        VALIDATION_RUN_SETTINGS: {
            "n_posterior_samples": int(n_posterior_samples),
            "n_conditions": int(n_conditions),
        },
    }


def _dropped_joint_keys(
    failed_joint: dict[str, str], emitted_keys: dict[str, set[str]],
) -> set[str]:
    """Summary keys belonging to a joint metric invalidated for this trial.

    Both sources are consulted: what the metric was observed to emit before
    it failed, and what a registered name declares. The observed set is
    empty when the metric failed on the very first condition, and the
    declared set is only meaningful for a registered name, so neither covers
    the other.
    """
    dropped: set[str] = set()
    for name in failed_joint:
        dropped.update(emitted_keys.get(name, ()))
        dropped.update(output_keys_for(name))
    return dropped


def _joint_condition_frame(
    joint_condition_rows: list[dict[str, float]],
    failed_joint: dict[str, str],
    emitted_keys: dict[str, set[str]],
) -> pd.DataFrame:
    """Per-condition joint values as a frame, one row per condition.

    The values the loop already computed are what a caller needs to see
    which conditions a joint metric actually scores on: an unconditional
    reduction blends a condition where the metric detects a miscalibration
    with one where it cannot, and the contribution of each is not
    recoverable from the reduced number. Modrak et al. (2025) describe that
    cancellation for the marginal case, and it is no weaker jointly, because
    a joint metric's VALIDITY can differ by condition -- Lemos et al. (2023,
    Sec. 4.3) give the concrete failure, an estimator returning the prior
    that TARP scores as perfectly covered.

    Keys of a metric in *failed_joint* are dropped here as they are from the
    summary. The frame would otherwise invite the caller to reduce the
    surviving conditions themselves and recover exactly the flattering
    partial value that whole-trial invalidation exists to prevent;
    ``ValidationResult.failed_joint_metrics`` names what was dropped and
    why.

    Returns
    -------
    pandas.DataFrame
        Columns ``id_cond`` plus each surviving joint key, or an empty frame
        when no joint metric ran.
    """
    if not joint_condition_rows:
        return pd.DataFrame()
    dropped = _dropped_joint_keys(failed_joint, emitted_keys)
    return pd.DataFrame(
        [
            {
                "id_cond": cond_id,
                **{k: v for k, v in row.items() if k not in dropped},
            }
            for cond_id, row in enumerate(joint_condition_rows)
        ]
    )


def _aggregate_joint_rows(
    joint_condition_rows: list[dict[str, float]],
    failed_joint: dict[str, str],
    emitted_keys: dict[str, set[str]],
    aggregate: Aggregate = "mean",
) -> dict[str, float]:
    """Reduce each joint key across conditions, dropping invalidated metrics.

    Keys belonging to a metric in *failed_joint* are removed outright rather
    than averaged over its surviving conditions: a partially computed joint
    metric must reach the objective as ABSENT, so the objective substitutes
    its registered worst case exactly once. Averaging the successes instead
    would report a finite, flattering number that never triggers the
    penalty; substituting the worst case per condition would report a blend
    of real and penalty values, which reads as a mediocre model rather than
    a broken measurement.
    """
    dropped = _dropped_joint_keys(failed_joint, emitted_keys)

    keys: list[str] = []
    for row in joint_condition_rows:
        for key in row:
            if key not in dropped and key not in keys:
                keys.append(key)

    summary: dict[str, float] = {}
    for key in keys:
        vals = [row[key] for row in joint_condition_rows if key in row]
        if vals:
            reduction, explicit = resolve_reduction(aggregate, key)
            summary[key] = reduce_metric(vals, key, reduction, explicit=explicit)
    return summary


def run_validation_pipeline(
    approximator: Any,
    validation_data: ValidationDataset,
    n_posterior_samples: int = 1000,
    metrics: Sequence[str] | None = None,
    joint_metrics: Mapping[str, JointMetricFn] | None = None,
    max_samples_per_call: int | None = DEFAULT_MAX_SAMPLES_PER_CALL,
    aggregate: Aggregate = "mean",
) -> ValidationResult:
    """Run metric evaluation on a fixed dataset reused across trials.

    Parameters
    ----------
    approximator
        Trained BayesFlow approximator with a ``.sample()`` method.
    validation_data
        Pre-generated :class:`ValidationDataset`.
    n_posterior_samples
        Number of posterior draws per simulation.
    metrics
        List of metric names to compute (resolved via the registry).
        Defaults to :data:`~bayesflow_hpo.validation.registry.DEFAULT_METRICS`.
        Names registered with
        :func:`~bayesflow_hpo.validation.registry.register_joint_metric` are
        routed to the joint dispatch instead of the per-parameter one.
    joint_metrics
        Additional joint metrics as ``{name: fn}``, merged over the ones
        resolved from *metrics* -- and merged **unconditionally**, so a
        metric given here runs whether or not its name appears in
        *metrics*. That is deliberate and load-bearing:
        `make_lc2st_validate_fn` adds L-C2ST this way while passing a
        *metrics* list that omits it. The cost is that an override for a
        metric nothing optimizes is still paid for on every condition, so
        pass only what you want computed. For a metric whose *configuration* belongs
        to one study rather than to the process -- L-C2ST's fold count,
        classifier and seed, say -- passing it here keeps that configuration
        out of the global registry, where it would silently apply to every
        other study in the same interpreter.
    max_samples_per_call
        Cap on posterior draws requested from the approximator per
        ``sample()`` call; the condition batch is sampled in slices that
        respect it.  See
        :func:`~bayesflow_hpo.validation.inference.make_bayesflow_infer_fn`.
        ``None`` samples each condition in a single call.

    aggregate
        Reduction: ``"mean"`` (default), ``"worst"``, ``"geometric"``, or
        a mapping from metric output name to reduction (aliases accepted).
        A scalar reduces conditions within each parameter, then averages
        parameter summaries. Explicit mapping entries reduce the full
        parameter-by-condition grid with equal cell weights; omitted keys
        retain the default mean behavior. Joint metrics reduce conditions
        only. Worst respects the registered metric direction. Geometric
        requires strictly positive values and raises ``ValueError`` for
        zero or negative values; NaNs are omitted for every reduction.
        See :func:`~bayesflow_hpo.validation.metrics.reduce_metric`.

    Returns
    -------
    ValidationResult
        Structured result with per-condition and summary tables.
    """
    aggregate = normalize_aggregate(aggregate)
    if metrics is None:
        metrics = list(DEFAULT_METRICS)
    metric_fns = resolve_metrics(list(metrics))
    # The override names are passed IN rather than merged after: a name
    # the caller supplies must not be resolved from the registry at all,
    # or a placeholder like `tarp_error` raises before its replacement is
    # ever consulted.
    joint_metric_fns = resolve_joint_metrics(
        list(metrics), overridden=(joint_metrics or {}).keys()
    )
    if joint_metrics:
        # An override names a JOINT metric. Merging an unchecked key runs a
        # metric nobody asked for: `joint_metrics={"nrmse": fn}` would leave
        # `nrmse` computing marginally AND dispatch the override jointly
        # under the same name, and a typo'd key would silently add a metric
        # that appears in the summary from nowhere. Checked here rather than
        # trusted, since `canonical_metric_name` passes unknown names
        # through unchanged by design.
        unknown = [
            name for name in joint_metrics if not is_joint_metric(name)
        ]
        if unknown:
            raise JointMetricConfigurationError(
                f"joint_metrics keys {sorted(unknown)} are not registered "
                "joint metrics. A key overrides a registered joint metric's "
                "callable, so it must name one; register it first with "
                "`register_joint_metric`, or correct the spelling. "
                f"Registered joint metrics: {sorted(_JOINT)}."
            )
        joint_metric_fns = {**joint_metric_fns, **joint_metrics}

    available_keys = (
        set(validation_data.simulations[0].keys())
        if validation_data.simulations
        else None
    )
    infer_fn = make_bayesflow_infer_fn(
        approximator=approximator,
        param_keys=validation_data.param_keys,
        data_keys=validation_data.data_keys,
        available_keys=available_keys,
        max_samples_per_call=max_samples_per_call,
    )

    timing: dict[str, float] = {"inference": 0.0, "metrics": 0.0}
    n_params = len(validation_data.param_keys)
    multi_param = n_params > 1

    # Joint metrics aggregate across conditions on their own path: their
    # values are NOT written into the per-parameter rows. See the summary
    # assembly below for why routing them through `per_parameter` would
    # delete them in the multi-parameter case.
    joint_condition_rows: list[dict[str, float]] = []
    # Joint metrics that raised on at least one condition. Per D8: a metric
    # that failed anywhere is invalidated for the WHOLE trial rather than
    # averaged over the conditions that happened to succeed, so that its
    # score cannot depend on WHICH condition failed and a model cannot
    # benefit from failing on the conditions it finds hardest.
    failed_joint: dict[str, str] = {}
    # Keys each joint metric was seen to emit, so a failure on a later
    # condition can withdraw what an earlier one contributed.
    emitted_keys: dict[str, set[str]] = {}

    # Per-parameter condition rows: {param_key: [row_dicts]}
    param_condition_rows: dict[str, list[dict[str, Any]]] = {}
    if multi_param:
        for pk in validation_data.param_keys:
            param_condition_rows[pk] = []
    else:
        param_condition_rows[validation_data.param_keys[0]] = []

    for cond_id, sim_batch in enumerate(validation_data.simulations):
        # --- Inference ---
        t0 = time.perf_counter()
        draws = infer_fn(sim_batch, n_posterior_samples)
        timing["inference"] += time.perf_counter() - t0

        # --- Joint metrics, before the per-parameter branch ---
        # Placed here on purpose: the single-parameter branch below rebinds
        # `draws` to a 2-D array, so a dispatch after it would hand a joint
        # metric the wrong rank in exactly the studies most likely to use
        # one. Placement alone is not sufficient, though -- see
        # `_joint_draws`.
        t1 = time.perf_counter()
        if joint_metric_fns:
            joint_row = _run_joint_metrics(
                joint_metric_fns,
                draws=draws,
                sim_batch=sim_batch,
                validation_data=validation_data,
                approximator=approximator,
                cond_id=cond_id,
                n_conditions=len(validation_data.simulations),
                failed_joint=failed_joint,
                emitted_keys=emitted_keys,
            )
            joint_condition_rows.append(joint_row)
        timing["metrics"] += time.perf_counter() - t1

        # --- Metrics per parameter ---
        t1 = time.perf_counter()
        if multi_param:
            if draws.ndim != 3:
                raise ValueError(
                    "Expected posterior draws with shape (n_sims, n_samples, n_params) "
                    "for multi-parameter inference."
                )
            for param_idx, param_key in enumerate(validation_data.param_keys):
                true_values = np.asarray(sim_batch[param_key]).reshape(-1)
                param_draws = np.asarray(draws[:, :, param_idx])
                row = compute_condition_metrics(
                    param_draws, true_values, cond_id, metric_fns,
                )
                param_condition_rows[param_key].append(row)
        else:
            param_key = validation_data.param_keys[0]
            true_values = np.asarray(sim_batch[param_key]).reshape(-1)
            if draws.ndim == 3 and draws.shape[-1] == 1:
                draws = np.squeeze(draws, axis=-1)
            row = compute_condition_metrics(draws, true_values, cond_id, metric_fns)
            param_condition_rows[param_key].append(row)

        timing["metrics"] += time.perf_counter() - t1
        cleanup_trial()

    # --- Assemble result ---
    n_conditions = len(validation_data.simulations)

    if multi_param:
        per_parameter: dict[str, ValidationResult] = {}
        all_condition_rows: list[dict[str, Any]] = []

        for param_key, cond_rows in param_condition_rows.items():
            param_summary = aggregate_condition_rows(cond_rows, aggregate)
            param_cond_df = pd.DataFrame(cond_rows)
            per_parameter[param_key] = ValidationResult(
                condition_metrics=param_cond_df,
                summary=param_summary,
                n_conditions=n_conditions,
                n_posterior_samples=n_posterior_samples,
                metric_names=list(metrics),
            )
            for row in cond_rows:
                tagged = dict(row, param_key=param_key)
                all_condition_rows.append(tagged)

        condition_df = pd.DataFrame(all_condition_rows)
        # Overall summary: average across per-parameter summaries
        per_parameter_mean_summary: dict[str, float] = {}
        for key in per_parameter[validation_data.param_keys[0]].summary:
            vals = [pr.summary.get(key, float("nan")) for pr in per_parameter.values()]
            per_parameter_mean_summary[key] = float(np.nanmean(vals))

        if isinstance(aggregate, Mapping):
            # Explicit settings retain the parameter axis until the final
            # reduction; otherwise an average can hide a failing parameter.
            grid_summary = aggregate_condition_rows(all_condition_rows, aggregate)
            for key in aggregate:
                if key in grid_summary:
                    per_parameter_mean_summary[key] = grid_summary[key]

        # Joint keys are merged in AFTER that loop, not routed through it.
        # The loop takes its key set from the first parameter's summary, so a
        # joint key -- which has no per-parameter value by construction --
        # would be absent from the overall summary entirely. The objective
        # would then find no value, substitute the metric's worst case on
        # every trial, and the study would silently optimize a constant.
        overall_summary = {
            **per_parameter_mean_summary,
            **_aggregate_joint_rows(
                joint_condition_rows, failed_joint, emitted_keys, aggregate
            ),
        }

        return ValidationResult(
            condition_metrics=condition_df,
            joint_condition_metrics=_joint_condition_frame(
                joint_condition_rows, failed_joint, emitted_keys
            ),
            summary=overall_summary,
            per_parameter=per_parameter,
            timing=timing,
            n_conditions=n_conditions,
            n_posterior_samples=n_posterior_samples,
            metric_names=list(metrics),
            failed_joint_metrics=dict(failed_joint),
            joint_metric_settings=_declared_settings(
                joint_metric_fns, n_posterior_samples, n_conditions
            ),
        )

    # Single-parameter case
    param_key = validation_data.param_keys[0]
    cond_rows = param_condition_rows[param_key]
    condition_df = pd.DataFrame(cond_rows)
    summary = {
        **aggregate_condition_rows(cond_rows, aggregate),
        **_aggregate_joint_rows(
            joint_condition_rows, failed_joint, emitted_keys, aggregate
        ),
    }

    return ValidationResult(
        condition_metrics=condition_df,
        joint_condition_metrics=_joint_condition_frame(
            joint_condition_rows, failed_joint, emitted_keys
        ),
        summary=summary,
        timing=timing,
        n_conditions=n_conditions,
        n_posterior_samples=n_posterior_samples,
        metric_names=list(metrics),
        failed_joint_metrics=dict(failed_joint),
        joint_metric_settings=_declared_settings(
            joint_metric_fns, n_posterior_samples, n_conditions
        ),
    )
