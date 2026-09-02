"""Generic Optuna objective for BayesFlow HPO.

This module implements the core trial loop: sample → build → compile →
train → validate → return objective values.  Each ``__call__`` invocation
maps one Optuna trial to a minimize-all tuple of metric and cost scores.

Key design decisions:

- **Pre-training budget rejection**: Trials exceeding ``max_param_count``
  or ``max_memory_mb`` are rejected *before* training to save GPU time.
  These trials still return penalty values so Optuna records them, but
  they are flagged via ``rejected_reason`` and excluded from the trained
  trial count.

- **Two-phase param count check**: First a heuristic estimate (fast, no
  GPU), then an exact count after lazy Keras weight initialization.

- **Three hooks**: ``build_approximator_fn``, ``train_fn``, and
  ``validate_fn`` let callers replace the build, train, and validate
  steps while reusing the full trial lifecycle.
"""

from __future__ import annotations

import logging
import math
import numbers
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import bayesflow as bf
import optuna

from bayesflow_hpo.builders.workflow import (
    _compile_for_compat,
    _make_cosine_decay_optimizer,
    _make_inverse_sqrt_optimizer,
    build_continuous_approximator,
)
from bayesflow_hpo.objectives import (
    FAILED_TRIAL_CAL_ERROR,
    FAILED_TRIAL_COST,
    MAX_PARAM_COUNT,
    _direction_for,
    compute_inference_time_per_dataset,
    extract_multi_objective_values,
    get_param_count,
    mean_objective_score,
    normalize_param_count,
    worst_objective_value,
    worst_raw_value,
)
from bayesflow_hpo.optimization.callbacks import (
    MovingAverageEarlyStopping,
    OptunaReportCallback,
)
from bayesflow_hpo.optimization.checkpoint_pool import CheckpointPool
from bayesflow_hpo.optimization.cleanup import cleanup_trial
from bayesflow_hpo.optimization.constraints import (
    MetricConstraintSpec,
    estimate_peak_memory_mb,
)
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.types import BuildApproximatorFn, TrainFn, ValidateFn
from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.registry import (
    canonical_metric_name,
    validate_objective_metric_kinds,
)

logger = logging.getLogger(__name__)


def default_train_fn(
    approximator: Any,
    simulator: bf.simulators.Simulator,
    hparams: dict[str, Any],
    callbacks: list[Any],
) -> None:
    """Train via ``approximator.fit(simulator=..., ...)``.

    This is the default used by ``optimize()`` when ``train_fn`` is ``None``.
    Reads ``epochs``, ``num_batches``, and ``batch_size`` from
    ``hparams`` (injected by the objective before calling).

    Parameters
    ----------
    approximator
        Compiled BayesFlow approximator.
    simulator
        BayesFlow simulator for online training.
    hparams
        Hyperparameters dict (must contain ``epochs``, ``num_batches``,
        and optionally ``batch_size``).
    callbacks
        Keras callbacks (early stopping, Optuna reporter, etc.).
    """
    approximator.fit(
        simulator=simulator,
        epochs=int(hparams["epochs"]),
        batch_size=int(hparams.get("batch_size", 256)),
        num_batches=int(hparams["num_batches"]),
        callbacks=callbacks,
    )


def default_validate_fn(
    approximator: Any,
    validation_data: ValidationDataset,
    n_posterior_samples: int,
    objective_metrics: list[str] | None = None,
) -> dict[str, float]:
    """Run the built-in validation pipeline and return metric dict.

    This is the default used by ``optimize()`` when ``validate_fn`` is ``None``.
    Wraps ``run_validation_pipeline()`` and returns its summary as a flat dict.

    Parameters
    ----------
    approximator
        Trained BayesFlow approximator with a ``.sample()`` method.
    validation_data
        Pre-generated validation dataset.
    n_posterior_samples
        Number of posterior draws per simulation.
    objective_metrics
        Metrics this run optimizes.  Every registered name here is computed in
        addition to :data:`DEFAULT_METRICS`.  Leaving it ``None`` computes
        ``DEFAULT_METRICS`` alone, which is why callers must thread it
        through: ``log_gamma``, ``sbc_ks`` and ``sbc_chi2`` are registered but
        *not* default, so pre-flight reported them as missing keys and
        rejected the run before training started -- the headline metric could
        not be optimized through the public workflow at all.

    Returns
    -------
    dict[str, float]
        Metric name → value mapping (e.g. ``{"calibration_error": 0.05}``).
    """
    from bayesflow_hpo.validation.pipeline import run_validation_pipeline

    result = run_validation_pipeline(
        approximator=approximator,
        validation_data=validation_data,
        n_posterior_samples=n_posterior_samples,
        metrics=_pipeline_metrics(objective_metrics or []),
    )
    return dict(result.summary)


def _validate_metric_keys(
    raw: dict[str, float],
    objective_metrics: list[str],
    penalty_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """Validate and sanitize metric dict from a custom validate_fn.

    - Missing keys → replaced with penalty value + warning.
    - NaN/Inf values → replaced with penalty value + warning.

    Parameters
    ----------
    raw
        Raw metric dict from the validation function.
    objective_metrics
        Metric keys expected in *raw*.
    penalty_values
        Per-metric penalty values.  When provided, the penalty for a
        given key is looked up here; keys not present fall back to
        ``FAILED_TRIAL_CAL_ERROR``.

    Returns a cleaned copy of the dict.
    """
    # `objective_metrics` was canonicalized at the public boundary, but a
    # custom hook returns whatever spelling its author used -- and the
    # documented contract is that it returns the keys the caller *asked* for.
    # Comparing literally would penalize every trial of a hook that honoured
    # that contract with an alias. Canonicalize both sides so they meet.
    cleaned = {canonical_metric_name(k): v for k, v in raw.items()}
    for key in objective_metrics:
        # The penalty is inserted BEFORE direction conversion, so it must be a
        # raw-space value. A flat 1.0 is not: for `log_gamma` it converts to
        # -1.0, which beats a genuinely reported log_gamma of 0.5 -- a trial
        # that failed to report the metric would outrank one that reported a
        # good value. `worst_raw_value` is the raw-space worst case.
        penalty = (
            penalty_values.get(key, worst_raw_value(key))
            if penalty_values is not None
            else worst_raw_value(key)
        )
        if key not in cleaned:
            logger.warning(
                "validate_fn output missing metric %r — using penalty value", key
            )
            cleaned[key] = penalty
        elif not math.isfinite(cleaned[key]):
            logger.warning(
                "validate_fn returned non-finite value for %r — using penalty", key
            )
            cleaned[key] = penalty
    return cleaned


def _normalize_warmup_spec(
    name: str,
    value: int | Sequence[int] | None,
    *,
    default: int | None = None,
) -> int | tuple[int, ...] | None:
    """Validate a fixed or categorical warmup configuration."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, a sequence of ints, or None.")
    if isinstance(value, (int, numbers.Integral)):
        return int(value)
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an int, a sequence of ints, or None.")
    try:
        choices = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an int, a sequence of ints, or None."
        ) from exc
    if not choices or any(
        not isinstance(choice, numbers.Integral) or isinstance(choice, bool)
        for choice in choices
    ):
        raise TypeError(f"{name} choices must be a non-empty sequence of ints.")
    choices = tuple(int(choice) for choice in choices)
    if len(set(choices)) != len(choices):
        raise ValueError(f"{name} choices must be unique.")
    return choices


def _sample_warmup_spec(
    trial: optuna.Trial,
    name: str,
    spec: int | tuple[int, ...],
) -> int:
    """Resolve a fixed value or sample an explicitly configured choice."""
    if isinstance(spec, tuple):
        return int(trial.suggest_categorical(name, list(spec)))
    return spec


def _normalize_warmup_fraction_spec(
    value: float | Sequence[float] | None,
    *,
    default: float | None = None,
) -> float | tuple[float, ...] | None:
    """Validate a fixed or categorical warmup fraction in [0, 0.1]."""
    if value is None:
        return default
    values: tuple[float, ...]
    if isinstance(value, bool):
        raise TypeError(
            "lr_warmup_fraction must be a number, a sequence of numbers, "
            "or None."
        )
    if isinstance(value, (int, float)):
        values = (float(value),)
        scalar = True
    elif isinstance(value, (str, bytes)):
        raise TypeError(
            "lr_warmup_fraction must be a number, a sequence of numbers, "
            "or None."
        )
    else:
        try:
            values = tuple(float(choice) for choice in value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "lr_warmup_fraction choices must be a non-empty sequence "
                "of numbers."
            ) from exc
        scalar = False
    if not values:
        raise TypeError(
            "lr_warmup_fraction choices must be a non-empty sequence of numbers."
        )
    if any(not math.isfinite(choice) or not 0.0 <= choice <= 0.1 for choice in values):
        raise ValueError("lr_warmup_fraction must be between 0.0 and 0.1.")
    if len(set(values)) != len(values):
        raise ValueError("lr_warmup_fraction choices must be unique.")
    return values[0] if scalar else values


def _sample_warmup_fraction(
    trial: optuna.Trial,
    spec: float | tuple[float, ...],
) -> float:
    """Resolve a fixed fraction or sample explicitly configured choices."""
    if isinstance(spec, tuple):
        return float(trial.suggest_categorical("lr_warmup_fraction", list(spec)))
    return spec


@dataclass
class ObjectiveConfig:
    """Configuration for one objective function instance.

    Parameters
    ----------
    simulator, adapter
        BayesFlow simulator and adapter.
    search_space
        Composite search space defining the tunable dimensions.
    validation_data
        Pre-generated :class:`ValidationDataset` (required).
        Use :func:`~bayesflow_hpo.validation.data.generate_validation_dataset`
        to create one.
    training_mode
        ``"fixed_budget"`` (default) uses cosine decay and runs to the
        trial's full budget. ``"open_ended"`` uses inverse-square-root decay
        with warmup and stops on the configured validation objective.
    epochs
        Training epochs per trial. In ``open_ended`` mode this is a generous
        safety cap rather than a target (default 200).
    num_batches
        Online simulation batches per epoch (default 50).
    early_stopping_patience
        Validation checks without improvement before stopping in
        ``open_ended`` mode. ``None`` selects 5 checks. It must remain
        ``None`` in ``fixed_budget`` mode.
    early_stopping_window
        Moving-average window size (default 7).
    early_stopping_monitor
        Validation stopping objective. ``"objective_mean"`` (default) combines all
        objective metrics; a metric name monitors only that metric.
    lr_warmup_epochs
        Linear warmup measured using the trial's actual ``num_batches``.
        ``None`` selects 0 epochs in ``fixed_budget`` mode and 1 epoch in
        ``open_ended`` mode. A sequence enables opt-in categorical HPO.
    lr_warmup_steps
        Exact warmup-step override. A sequence enables opt-in categorical
        HPO. When provided, this takes precedence over ``lr_warmup_epochs``.
    lr_warmup_fraction
        Fraction of the fixed training budget used for linear warmup. ``None``
        selects 0.05 in ``fixed_budget`` mode. Values must be between 0 and
        0.1; a sequence enables opt-in categorical HPO. Exact steps or epochs
        take precedence. Fractions are not valid for ``open_ended`` mode.
    max_param_count
        Trials with actual param count above this are rejected
        before training (default 1 000 000).
    max_memory_mb
        Optional peak-memory budget in MB (disabled by default).
    n_posterior_samples
        Posterior draws for final validation (default 500).
    n_intermediate_posterior_samples
        Posterior draws for mid-training pruning validation
        (default 250).
    intermediate_validation_interval
        Run a lightweight validation every *n* epochs for pruning
        (default 10).
    intermediate_validation_warmup
        Skip the first *n* epochs before intermediate validation
        (default 10).
    pruning_strategy
        Multi-objective pruning strategy.  One of ``"dominance"``
        (default), ``"mo-sha"``, ``"primary"``, or ``"none"``.
        For ``"primary"``, pass a tuple ``("primary", metric_name)``.
        See :mod:`~bayesflow_hpo.optimization.pruning_strategies`.
    pruning_n_startup_trials
        Minimum completed trials before multi-objective pruning
        activates.  ``None`` (default) → auto-detected from the
        sampler's ``n_startup_trials`` attribute (25 for TPE, fallback
        10).  Explicit ``int`` overrides auto-detection.
    objective_metrics
        List of metric keys to optimize simultaneously.
        Default ``["calibration_error", "nrmse"]``.
    objective_mode
        ``"pareto"`` (default) — each metric becomes its own Optuna
        objective; returns ``len(objective_metrics) + 1`` values.
        ``"mean"`` — arithmetic mean of the listed metrics forms a
        single scalar; returns 2 values ``(mean, cost_score)``.
    cost_metric
        Which cost objective to use as the last Optuna objective.
        ``"inference_time"`` (default) or ``"param_count"``.
    report_frequency
        How often (in epochs) the ``OptunaReportCallback`` stores
        ``epoch_{N}_loss`` user attributes on each trial (default 10).
    checkpoint_pool
        Optional :class:`CheckpointPool` to persist the best trial
        weights.  When ``None`` a default pool of size 5 is created.
    build_approximator_fn
        Optional custom build function ``(hparams) -> Approximator``.
        Must return an **uncompiled** approximator.
    train_fn
        Optional custom training function
        ``(approximator, simulator, hparams, callbacks) -> None``.
    validate_fn
        Optional custom validation function
        ``(approximator, validation_data, n_posterior_samples) ->
        dict[str, float]``.

        The returned dict must contain all keys listed in
        ``objective_metrics``.  Missing or non-finite values are
        replaced with a penalty and a warning is logged.  Extra keys
        are silently ignored.

        **Timing caveat:** wall-clock time of this function is used as
        the trial's inference time, lumping inference and metric
        computation together (unlike the default path which isolates
        pure inference timing).

        **Intermediate pruning:** also called during training at the
        configured interval with reduced ``n_posterior_samples`` for
        median-based multi-objective pruning.
    """

    simulator: bf.simulators.Simulator
    adapter: bf.adapters.Adapter
    search_space: CompositeSearchSpace
    validation_data: ValidationDataset
    training_mode: Literal["fixed_budget", "open_ended"] = "fixed_budget"
    epochs: int = 200
    num_batches: int = 50
    early_stopping_patience: int | None = None
    early_stopping_window: int = 7
    early_stopping_monitor: str = "objective_mean"
    lr_warmup_epochs: int | Sequence[int] | None = None
    lr_warmup_steps: int | Sequence[int] | None = None
    lr_warmup_fraction: float | Sequence[float] | None = None
    max_param_count: int = MAX_PARAM_COUNT
    max_memory_mb: float | None = None
    metric_constraints_hard: list[MetricConstraintSpec] | None = None
    #: Soft constraints are enforced by Optuna's constraints function, not by
    #: this objective, but their metrics still have to be COMPUTED here or the
    #: constraint reads a missing user attribute as zero violation and never
    #: fires. Recorded so the pipeline metric list can include them.
    metric_constraints_soft: list[MetricConstraintSpec] | None = None
    n_posterior_samples: int = 500
    n_intermediate_posterior_samples: int = 250
    intermediate_validation_interval: int = 10
    intermediate_validation_warmup: int = 10
    pruning_strategy: str | tuple[str, str] = "dominance"
    pruning_n_startup_trials: int | None = None
    objective_metrics: list[str] = field(
        default_factory=lambda: ["calibration_error", "nrmse"]
    )
    objective_mode: str = "pareto"
    cost_metric: str = "inference_time"
    checkpoint_pool: CheckpointPool | None = None
    report_frequency: int = 10
    build_approximator_fn: BuildApproximatorFn | None = None
    train_fn: TrainFn | None = None
    validate_fn: ValidateFn | None = None

    def __post_init__(self) -> None:
        validate_objective_metric_kinds(self.objective_metrics)
        # Canonicalize aliases HERE, once, so every downstream consumer agrees
        # on the key. They did not: `list_metrics()` returns canonical names,
        # so `cal_error` was excluded from the pipeline's metric list; the
        # pipeline emitted `calibration_error`; and `_validate_metric_keys`
        # then looked for `cal_error`, found it missing, and inserted the
        # penalty. Every trial using a supported alias tied at the penalty.
        self.objective_metrics = [
            canonical_metric_name(m) for m in self.objective_metrics
        ]
        # `early_stopping_monitor` names one of those metrics, so it has to be
        # canonicalized with them. Otherwise canonicalizing only the list turns
        # a previously-valid pairing -- objective_metrics=["cal_error"] with
        # early_stopping_monitor="cal_error" -- into a ValueError, because the
        # membership check below compares the still-aliased monitor against
        # the now-canonical list.
        if self.early_stopping_monitor != "objective_mean":
            self.early_stopping_monitor = canonical_metric_name(
                self.early_stopping_monitor
            )
        # Every OTHER field naming a metric has to be canonicalized in the same
        # place, or it reads a key nothing writes. These are not hypothetical:
        # `("primary", "cal_error")` made PeriodicValidationCallback index a
        # dict keyed `calibration_error`, raising KeyError at each intermediate
        # validation and converting the trial into a training-failure penalty;
        # a constraint spec naming an alias silently never matched, so a hard
        # constraint skipped it and a soft one read zero violation. See
        # `tests/test_optimization/test_metric_name_inventory.py`, which fails
        # if a future field joins this set without joining this block.
        if isinstance(self.pruning_strategy, tuple):
            self.pruning_strategy = (
                self.pruning_strategy[0],
                canonical_metric_name(self.pruning_strategy[1]),
            )
        for _attr in ("metric_constraints_hard", "metric_constraints_soft"):
            _specs = getattr(self, _attr)
            if _specs is not None:
                setattr(self, _attr, [
                    (canonical_metric_name(name), threshold, direction)
                    for name, threshold, direction in _specs
                ])
        if self.training_mode not in ("fixed_budget", "open_ended"):
            raise ValueError(
                f"Unknown training_mode: {self.training_mode!r}. "
                "Expected 'fixed_budget' or 'open_ended'."
            )
        if (
            self.training_mode == "fixed_budget"
            and self.early_stopping_patience is not None
        ):
            raise ValueError(
                "early_stopping_patience cannot be set in fixed_budget mode; "
                "finite-horizon cosine annealing must run to its horizon."
            )
        if (
            self.training_mode == "open_ended"
            and self.early_stopping_patience is None
        ):
            self.early_stopping_patience = 5
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be >= 1 or None.")
        if self.early_stopping_window < 1:
            raise ValueError("early_stopping_window must be >= 1.")
        if (
            self.early_stopping_monitor != "objective_mean"
            and self.early_stopping_monitor not in self.objective_metrics
        ):
            raise ValueError(
                "early_stopping_monitor must be 'objective_mean' or one of "
                f"objective_metrics, got {self.early_stopping_monitor!r}."
            )
        requested_warmup_fraction = self.lr_warmup_fraction
        default_warmup_epochs = None if self.training_mode == "fixed_budget" else 1
        self.lr_warmup_epochs = _normalize_warmup_spec(
            "lr_warmup_epochs",
            self.lr_warmup_epochs,
            default=default_warmup_epochs,
        )
        self.lr_warmup_steps = _normalize_warmup_spec(
            "lr_warmup_steps",
            self.lr_warmup_steps,
        )
        if self.training_mode == "open_ended" and requested_warmup_fraction is not None:
            raise ValueError(
                "lr_warmup_fraction is only valid in 'fixed_budget' mode; "
                "use lr_warmup_epochs or lr_warmup_steps for open-ended training."
            )
        self.lr_warmup_fraction = _normalize_warmup_fraction_spec(
            self.lr_warmup_fraction,
            default=0.05 if self.training_mode == "fixed_budget" else None,
        )
        minimum_warmup = 0 if self.training_mode == "fixed_budget" else 1
        precedence = (
            ("lr_warmup_steps", self.lr_warmup_steps),
            ("lr_warmup_epochs", self.lr_warmup_epochs),
            ("lr_warmup_fraction", self.lr_warmup_fraction),
        )
        for name, spec in precedence:
            if spec is None:
                continue
            values = spec if isinstance(spec, tuple) else (spec,)
            if any(value < minimum_warmup for value in values):
                raise ValueError(
                    f"{name} must be >= {minimum_warmup} in "
                    f"{self.training_mode!r} mode."
                )
            break
        if self.training_mode == "fixed_budget":
            total_steps = self.epochs * self.num_batches
            if isinstance(self.lr_warmup_steps, int):
                static_warmup_steps = self.lr_warmup_steps
            elif isinstance(self.lr_warmup_epochs, int):
                static_warmup_steps = self.lr_warmup_epochs * self.num_batches
            else:
                static_warmup_steps = None
            if static_warmup_steps is not None and static_warmup_steps >= total_steps:
                raise ValueError(
                    "Static warmup must be shorter than the fixed training budget, "
                    f"got {static_warmup_steps} >= {total_steps} optimizer steps."
                )
        if not isinstance(self.validation_data, ValidationDataset):
            raise TypeError(
                f"validation_data must be a ValidationDataset, "
                f"got {type(self.validation_data).__name__}. "
                f"Use generate_validation_dataset() to create one."
            )
        if self.objective_mode not in ("mean", "pareto"):
            raise ValueError(
                f"Unknown objective_mode: {self.objective_mode!r}. "
                f"Expected 'mean' or 'pareto'."
            )
        if self.cost_metric not in ("inference_time", "param_count"):
            raise ValueError(
                f"Unknown cost_metric: {self.cost_metric!r}. "
                f"Expected 'inference_time' or 'param_count'."
            )
        if self.metric_constraints_hard is not None:
            for metric, _, direction in self.metric_constraints_hard:
                if direction not in ("above", "below"):
                    raise ValueError(
                        "Invalid hard metric constraint direction "
                        f"{direction!r} for metric {metric!r}; "
                        "expected 'above' or 'below'."
                    )
        if self.report_frequency < 1:
            raise ValueError(
                f"report_frequency must be >= 1, got {self.report_frequency}."
            )
        # Validate pruning_strategy.
        _valid = {"none", "dominance", "mo-sha", "primary"}
        if isinstance(self.pruning_strategy, tuple):
            if (
                len(self.pruning_strategy) != 2
                or self.pruning_strategy[0] != "primary"
            ):
                raise ValueError(
                    f"Tuple pruning_strategy must be "
                    f"('primary', metric_name), "
                    f"got {self.pruning_strategy!r}"
                )
        elif self.pruning_strategy not in _valid:
            raise ValueError(
                f"Unknown pruning_strategy: {self.pruning_strategy!r}. "
                f"Expected one of {sorted(_valid)}."
            )
        # Validate pruning_n_startup_trials.
        if (
            self.pruning_n_startup_trials is not None
            and self.pruning_n_startup_trials < 0
        ):
            raise ValueError(
                f"pruning_n_startup_trials must be >= 0, "
                f"got {self.pruning_n_startup_trials}."
            )


def _extract_best_training_loss(callbacks: list[Any]) -> float | None:
    """Extract the best moving-average loss from training callbacks.

    Looks for a :class:`MovingAverageEarlyStopping` callback and returns
    its ``best_ma_loss``.  Returns ``None`` if no such callback exists
    or the value is non-finite.
    """
    for cb in callbacks:
        if isinstance(cb, MovingAverageEarlyStopping):
            val = cb.best_ma_loss
            if math.isfinite(val):
                return val
    return None


def _accepts_training_loss_proxy(metric: str) -> bool:
    """Is a clamped [0, 1] lower-is-better loss on scale for this metric?

    When final validation raises, :func:`_training_loss_fallback` substitutes
    the best clamped training loss for each objective. That substitution is a
    [0, 1] lower-is-better number, so it is only meaningful for a metric on
    the same scale and in the same direction. Applied to a higher-is-better
    metric it does not merely approximate badly, it inverts the ranking: a
    trial whose validation failed would score better than one that reported a
    real value.

    Parameters
    ----------
    metric
        Objective metric name, canonical or custom.

    Returns
    -------
    bool
        ``True`` only when the metric has a recorded direction that is
        lower-is-better with a unit worst case. A metric with no recorded
        direction returns ``False``: absence establishes neither direction nor
        scale, and guessing in the permissive direction is what lets a failed
        trial win.

    References
    ----------
    The asymmetry this guards against is documented for ``log_gamma`` in
    ``docs/references.md`` (Modrák et al., 2025): ``log_gamma < 0`` rejects
    rank uniformity, so larger is better and a small substituted loss reads as
    a *good* result.
    """
    direction = _direction_for(metric)
    if direction is None:
        # No recorded direction or scale. A custom metric could be
        # higher-is-better or unbounded, and substituting a small loss would
        # claim a good result for a trial whose validation failed.
        return False
    return not direction.higher_is_better and direction.worst_objective == 1.0


def _constraint_metric_names(config: ObjectiveConfig) -> list[str]:
    """Metric names referenced by hard or soft constraints, in order."""
    names: list[str] = []
    for specs in (
        config.metric_constraints_hard,
        config.metric_constraints_soft,
    ):
        for name, _threshold, _direction in specs or ():
            if name not in names:
                names.append(name)
    return names


def _pipeline_metrics(
    objective_metrics: list[str],
    constraint_metrics: Sequence[str] = (),
) -> list[str]:
    """DEFAULT_METRICS plus registered objectives and constraints, in order.

    Constraints are included for the same reason objectives are, and the
    consequence of omitting them is quieter: a constraint naming a registered
    non-default metric -- ``metric_constraints_hard=[("sbc_ks", 0.2, "above")]``
    alongside the default objectives -- was never computed, and neither path
    complains. The hard path skips a missing key outright, and the soft
    callback reads a missing user attribute as zero violation, so the
    constraint is configured, inactive, and silent.

    Unregistered names are left out because the pipeline resolves names
    through the registry and would raise; they behave as before, falling to a
    penalty.
    """
    from bayesflow_hpo.validation.registry import (
        DEFAULT_METRICS,
        producer_for_key,
    )

    names = list(DEFAULT_METRICS)
    for metric in list(objective_metrics) + list(constraint_metrics):
        # A constraint names an output KEY, which for a multi-output
        # diagnostic is not the metric that produces it: constraining
        # `left_coverage_90` needs `coverage_left` in the pipeline. Filtering
        # on registered names alone dropped the producer, so nothing computed
        # the key -- and neither constraint path complains, because the hard
        # path skips a missing key and the soft path reads it as zero
        # violation.
        producer = producer_for_key(metric)
        if producer is not None and producer not in names:
            names.append(producer)
    return names


def _training_loss_fallback(
    best_training_loss: float | None,
    objective_metrics: list[str],
    objective_mode: str,
    param_count: int,
    max_param_count: int,
    cost_metric: str,
    penalty: tuple[float, ...],
) -> tuple[float, ...]:
    """Build objective values from training loss when validation fails.

    Uses the best moving-average training loss (clamped to [0, 1]) as a
    proxy for each metric objective, paired with a cost score.  When
    ``cost_metric`` is ``"param_count"``, the real normalized param count
    is used; when ``"inference_time"``, the penalty cost is used since
    no inference was performed.  Falls back to full penalty values if
    the training loss is unavailable.

    Parameters
    ----------
    best_training_loss
        Best moving-average training loss, or ``None`` if unavailable.
    objective_metrics
        Metric keys being optimized.
    objective_mode
        ``"pareto"`` or ``"mean"``.
    param_count
        Actual parameter count from the built model.
    max_param_count
        Budget cap for param-count normalization.
    cost_metric
        ``"inference_time"`` or ``"param_count"``.
    penalty
        Full penalty tuple to return if training loss is unavailable.
    """
    if best_training_loss is None:
        return penalty

    # Clamp to [0, 1] — training loss is not directly comparable to
    # calibration metrics, but a lower loss is a reasonable proxy for
    # better performance.  Clamping ensures the value fits the same
    # scale as the metric objectives.
    clamped_loss = max(0.0, min(1.0, best_training_loss))

    if cost_metric == "param_count":
        cost_score = normalize_param_count(param_count, max_count=max_param_count)
    else:
        # No inference was performed, so use the penalty cost value.
        cost_score = FAILED_TRIAL_COST

    # The clamped loss is a [0, 1] lower-is-better proxy, so it is only on
    # scale for a metric that is itself [0, 1] and lower-is-better. Applying
    # it to `log_gamma` is not merely imprecise, it inverts the ranking: a
    # trial whose validation FAILED scores 0.1, while a valid trial reporting
    # log_gamma = -50 scores 50, so the failure dominates the real result.
    # Metrics off that scale get their worst objective instead.
    per_metric = [
        clamped_loss
        if _accepts_training_loss_proxy(metric)
        else worst_objective_value(metric)
        for metric in objective_metrics
    ]
    if objective_mode == "pareto":
        return tuple(per_metric) + (cost_score,)
    # "mean" mode collapses the metrics, so the proxy must too.
    mean_val = math.fsum(per_metric) / len(per_metric) if per_metric else clamped_loss
    return (mean_val, cost_score)


def _log_trial_summary(
    trial: optuna.Trial,
    values: tuple[float, ...],
    param_count: int,
    training_time: float,
    objective_metrics: list[str] | None = None,
    cost_metric_name: str | None = None,
) -> None:
    """Log a concise one-line summary after a trial completes.

    Parameters
    ----------
    trial
        The Optuna trial that just completed.
    values
        Objective values returned by the trial.
    param_count
        Actual trainable parameter count.
    training_time
        Wall-clock training time in seconds.
    objective_metrics
        Metric keys being optimized (e.g. ``["calibration_error", "nrmse"]``).
        Each metric's value is read from ``trial.user_attrs``.
    cost_metric_name
        Display name for the cost metric (last value in *values*).
    """
    params_label = (
        f"{param_count / 1e6:.2f}M"
        if param_count >= 1e6
        else f"{param_count / 1e3:.1f}K"
        if param_count >= 1e3
        else str(param_count)
    )
    parts = [
        f"Trial #{trial.number} done ({training_time:.0f}s)",
        f"params: {params_label}",
    ]
    # Show each objective metric from trial user attrs.
    if objective_metrics:
        for key in objective_metrics:
            val = trial.user_attrs.get(key)
            if val is not None:
                parts.append(f"{key}: {val:.4f}")
    # Show the cost metric (last objective value).
    if cost_metric_name and values:
        cost_val = values[-1]
        if cost_metric_name == "inference_time_s":
            parts.append(f"{cost_metric_name}: {cost_val:.2f}s")
        else:
            parts.append(f"{cost_metric_name}: {cost_val:.4f}")
    logger.info(" | ".join(parts))


class GenericObjective:
    """Optuna objective returning a minimize-all tuple of metric and cost scores.

    Each call samples hyperparameters, builds the model, trains it,
    validates, and returns an objective tuple.  Failed, pruned, or
    budget-rejected trials return penalty values.

    The trial lifecycle:

    1. Sample hparams from search_space
    2. Supply fallback training config in hparams
    3. Budget pre-check (memory estimate)
    4. BUILD approximator (custom or default)
    5. COMPILE with the training mode's Adam learning-rate schedule
    6. Exact param count check
    7. TRAIN (custom or default)
    8. VALIDATE (custom or default)
    9. Cost scoring
    10. Checkpoint pool
    11. Logging
    """

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        if config.checkpoint_pool is None:
            self._checkpoint_pool = CheckpointPool()
        else:
            self._checkpoint_pool = config.checkpoint_pool

    @property
    def checkpoint_pool(self) -> CheckpointPool:
        """The checkpoint pool used by this objective."""
        return self._checkpoint_pool

    @property
    def _metric_label(self) -> str:
        """Human-readable label for the first objective value in logs."""
        cfg = self.config
        if cfg.objective_mode == "pareto":
            return cfg.objective_metrics[0]
        return f"mean({'+'.join(cfg.objective_metrics)})"

    @property
    def _cost_metric_name(self) -> str:
        """Display name for the cost metric in logs."""
        if self.config.cost_metric == "inference_time":
            return "inference_time_s"
        return "param_count_norm"

    @property
    def n_objectives(self) -> int:
        """Number of objective values returned per trial."""
        if self.config.objective_mode == "pareto":
            return len(self.config.objective_metrics) + 1  # metrics + cost
        return 2  # mean + cost

    def _penalty(self) -> tuple[float, ...]:
        """Return penalty values matching the expected objective shape.

        These are returned directly as Optuna objective values, so they are in
        MINIMIZE space and must be per-metric: a flat ``FAILED_TRIAL_CAL_ERROR``
        of 1.0 means ``log_gamma = -1``, an ordinary value that would not deter
        the sampler from a region where trials keep failing.
        """
        n = self.n_objectives
        metrics = self.config.objective_metrics
        if self.config.objective_mode == "pareto":
            worst = [worst_objective_value(m) for m in metrics]
            return tuple(worst) + (FAILED_TRIAL_COST,)
        # "mean" mode collapses the metrics into one value, so the penalty is
        # the mean of their individual worst cases.
        if n == 2:
            worst = [worst_objective_value(m) for m in metrics]
            mean_worst = (
                math.fsum(worst) / len(worst)
                if worst
                else FAILED_TRIAL_CAL_ERROR
            )
            return (mean_worst, FAILED_TRIAL_COST)
        return tuple([FAILED_TRIAL_CAL_ERROR] * (n - 1)) + (FAILED_TRIAL_COST,)

    def _metric_penalty_map(self) -> dict[str, float]:
        """Per-metric RAW-space penalties for :func:`_validate_metric_keys`.

        Deliberately *not* derived from :meth:`_penalty`, which is in minimize
        space. Feeding a minimize-space value into the pre-conversion
        substitution double-converts it -- the bug that let a missing
        ``log_gamma`` outrank a reported one.
        """
        return {m: worst_raw_value(m) for m in self.config.objective_metrics}

    def _reject_compile(
        self, trial: optuna.Trial, exc: Exception,
    ) -> tuple[float, ...]:
        """Log a compile failure, mark the trial rejected, and return penalty."""
        logger.warning(
            "Trial #%d: compile failed: %s", trial.number, exc,
        )
        trial.set_user_attr("rejected_reason", "compile_failed")
        trial.set_user_attr("compile_error", str(exc))
        cleanup_trial()
        return self._penalty()

    def _check_hard_constraints(
        self,
        metrics_summary: dict[str, float],
        trial: optuna.Trial,
    ) -> tuple[float, ...] | None:
        """Check hard metric constraints and return penalty on first violation."""
        constraints = self.config.metric_constraints_hard
        if constraints is None:
            return None

        for metric, threshold, direction in constraints:
            value = metrics_summary.get(metric)
            if value is None:
                logger.warning(
                    "Trial #%d: hard metric constraint skipped; missing metric %r",
                    trial.number,
                    metric,
                )
                continue

            violated = False
            if direction == "above":
                violated = value > threshold
            elif direction == "below":
                violated = value < threshold
            else:
                logger.warning(
                    "Trial #%d: hard metric constraint skipped; invalid direction %r",
                    trial.number,
                    direction,
                )
                continue

            if violated:
                trial.set_user_attr("rejected_reason", "metric_constraint")
                logger.info(
                    "Trial #%d rejected by hard metric constraint: %s=%.6f (%s %.6f)",
                    trial.number,
                    metric,
                    value,
                    direction,
                    threshold,
                )
                return self._penalty()

        return None

    def __call__(self, trial: optuna.Trial) -> tuple[float, ...]:
        """Execute one HPO trial: sample → build → compile → train → validate.

        Returns
        -------
        tuple[float, ...]
            Objective values (all minimize-is-better).  Shape depends on
            ``objective_mode``: 2 values for ``"mean"``, N+1 for ``"pareto"``.
            Failed or budget-rejected trials return penalty values.
        """
        config = self.config

        # --- Step 1: Sample hparams ---
        params = config.search_space.sample(trial)

        # --- Step 2: Supply fallback training config ---
        # Search-space values (including DerivedDimension results) take
        # precedence so the optimizer schedule and train_fn share one source.
        params.setdefault("epochs", config.epochs)
        params.setdefault("num_batches", config.num_batches)
        epochs = int(params["epochs"])
        num_batches = int(params["num_batches"])
        if epochs < 1 or num_batches < 1:
            trial.set_user_attr("rejected_reason", "invalid_training_budget")
            logger.info(
                "Trial #%d rejected: epochs=%d and num_batches=%d must be >= 1.",
                trial.number,
                epochs,
                num_batches,
            )
            return self._penalty()
        trial.set_user_attr("training_mode", config.training_mode)
        trial.set_user_attr("epochs", epochs)
        trial.set_user_attr("num_batches", num_batches)

        if "lr_warmup_steps" in params:
            warmup_steps = int(params["lr_warmup_steps"])
        elif config.lr_warmup_steps is not None:
            warmup_steps = _sample_warmup_spec(
                trial,
                "lr_warmup_steps",
                cast(int | tuple[int, ...], config.lr_warmup_steps),
            )
        elif "lr_warmup_epochs" in params:
            warmup_epochs = int(params["lr_warmup_epochs"])
            warmup_steps = warmup_epochs * num_batches
        elif config.lr_warmup_epochs is not None:
            warmup_epochs = _sample_warmup_spec(
                trial,
                "lr_warmup_epochs",
                cast(int | tuple[int, ...], config.lr_warmup_epochs),
            )
            warmup_steps = warmup_epochs * num_batches
        else:
            if "lr_warmup_fraction" in params:
                warmup_fraction = float(params["lr_warmup_fraction"])
                normalized_fraction = _normalize_warmup_fraction_spec(warmup_fraction)
                if not isinstance(normalized_fraction, float):
                    raise TypeError("A sampled warmup fraction must be scalar.")
                warmup_fraction = float(normalized_fraction)
            else:
                if config.lr_warmup_fraction is None:
                    raise TypeError("Fixed-budget training requires a warmup fraction.")
                warmup_fraction = _sample_warmup_fraction(
                    trial,
                    cast(float | tuple[float, ...], config.lr_warmup_fraction),
                )
            warmup_steps = round(warmup_fraction * epochs * num_batches)
        minimum_warmup = 0 if config.training_mode == "fixed_budget" else 1
        if warmup_steps < minimum_warmup:
            trial.set_user_attr("rejected_reason", "invalid_warmup")
            logger.info(
                "Trial #%d rejected: warmup_steps=%d must be >= %d in %r mode.",
                trial.number,
                warmup_steps,
                minimum_warmup,
                config.training_mode,
            )
            return self._penalty()
        trial.set_user_attr("lr_warmup_steps", warmup_steps)
        trial.set_user_attr("lr_warmup_epochs", warmup_steps / num_batches)
        trial.set_user_attr(
            "lr_warmup_fraction",
            warmup_steps / (epochs * num_batches),
        )

        # --- Step 3: Budget pre-check (memory) ---
        estimated_memory = estimate_peak_memory_mb(params)
        trial.set_user_attr("estimated_peak_memory_mb", float(estimated_memory))
        if (
            config.max_memory_mb is not None
            and estimated_memory > config.max_memory_mb
        ):
            trial.set_user_attr("rejected_reason", "memory_budget")
            logger.info(
                "Trial #%d rejected: estimated %.0f MB > budget %.0f MB",
                trial.number, estimated_memory, config.max_memory_mb,
            )
            return self._penalty()

        # --- Step 4: BUILD approximator ---
        try:
            if config.build_approximator_fn is not None:
                approximator = config.build_approximator_fn(params)
            else:
                approximator = build_continuous_approximator(
                    params, config.adapter, config.search_space,
                )
        except Exception as exc:
            logger.warning(
                "Trial #%d: build failed: %s", trial.number, exc,
            )
            trial.set_user_attr("rejected_reason", "build_failed")
            trial.set_user_attr("build_error", str(exc))
            cleanup_trial()
            return self._penalty()

        # --- Step 5: COMPILE with the mode's coherent LR schedule ---
        if config.train_fn is None and "initial_lr" not in params:
            logger.warning(
                "Trial #%d: 'initial_lr' not in hparams, defaulting to 1e-3. "
                "Add 'initial_lr' to your search space or provide a custom "
                "train_fn.",
                trial.number,
            )
        initial_lr = float(params.get("initial_lr", 1e-3))
        trial.set_user_attr("peak_learning_rate", initial_lr)
        try:
            if config.training_mode == "fixed_budget":
                optimizer = _make_cosine_decay_optimizer(
                    initial_lr,
                    num_batches * epochs,
                    warmup_steps,
                )
            else:
                optimizer = _make_inverse_sqrt_optimizer(
                    initial_lr,
                    warmup_steps,
                )
        except Exception as exc:
            return self._reject_compile(trial, exc)
        try:
            _compile_for_compat(approximator, optimizer)
        except TypeError:
            pass  # _compile_for_compat raises TypeError on signature mismatch
        except Exception as exc:
            return self._reject_compile(trial, exc)

        # --- Step 6: Exact param count check ---
        try:
            dummy = config.simulator.sample((2,))
            adapted = config.adapter(dummy)
            if hasattr(approximator, "build_from_data"):
                approximator.build_from_data(adapted)
            else:
                approximator.compute_loss(adapted)
            param_count_actual = get_param_count(approximator)
            trial.set_user_attr("param_count", int(param_count_actual))
            if param_count_actual > config.max_param_count:
                trial.set_user_attr("rejected_reason", "param_budget")
                logger.info(
                    "Trial #%d rejected: %d params > budget %d",
                    trial.number, param_count_actual, config.max_param_count,
                )
                cleanup_trial()
                return self._penalty()
        except MemoryError:
            cleanup_trial()
            raise
        except Exception as exc:
            logger.warning(
                "Trial #%d: param count probe failed, rejecting trial: %s",
                trial.number, exc,
            )
            trial.set_user_attr("rejected_reason", "param_probe_failed")
            trial.set_user_attr("param_probe_error", str(exc))
            cleanup_trial()
            return self._penalty()

        # --- Callbacks ---
        callbacks: list[Any] = [
            MovingAverageEarlyStopping(
                monitor="loss",
                window=config.early_stopping_window,
                patience=None,
                restore_best_weights=False,
            ),
            OptunaReportCallback(
                trial, monitor="loss",
                report_frequency=config.report_frequency,
            ),
        ]

        # Resolve pruning strategy name for the "none" check.
        _strategy = (
            config.pruning_strategy[0]
            if isinstance(config.pruning_strategy, tuple)
            else config.pruning_strategy
        )
        if _strategy != "none" or config.training_mode == "open_ended":
            from bayesflow_hpo.optimization.validation_callback import (
                PeriodicValidationCallback,
            )

            callbacks.append(
                PeriodicValidationCallback(
                    trial=trial,
                    approximator=approximator,
                    validation_data=config.validation_data,
                    interval=config.intermediate_validation_interval,
                    warmup=config.intermediate_validation_warmup,
                    n_posterior_samples=config.n_intermediate_posterior_samples,
                    n_startup_trials=config.pruning_n_startup_trials,
                    validate_fn=config.validate_fn,
                    pruning_strategy=config.pruning_strategy,
                    objective_metrics=config.objective_metrics,
                    early_stopping_patience=config.early_stopping_patience,
                    early_stopping_window=config.early_stopping_window,
                    early_stopping_monitor=config.early_stopping_monitor,
                )
            )

        # --- Step 7: TRAIN ---
        t_train_start = time.perf_counter()
        try:
            if config.train_fn is not None:
                config.train_fn(approximator, config.simulator, params, callbacks)
            else:
                default_train_fn(approximator, config.simulator, params, callbacks)
        except optuna.TrialPruned:
            cleanup_trial()
            raise
        except Exception as exc:
            logger.warning("Trial %d failed during training: %s", trial.number, exc)
            trial.set_user_attr("training_error", str(exc))
            cleanup_trial()
            return self._penalty()
        training_time = time.perf_counter() - t_train_start
        trial.set_user_attr("training_time_s", round(training_time, 2))

        # Extract best training loss for potential validation fallback.
        best_training_loss = _extract_best_training_loss(callbacks)

        # --- Step 8: VALIDATE ---
        inference_time = 0.0
        n_conditions = 0
        try:
            n_conditions = len(config.validation_data.simulations)

            if config.validate_fn is not None:
                # Custom validation hook — wall-clock time includes
                # metric computation; pure inference timing unavailable.
                t_val_start = time.perf_counter()
                raw = config.validate_fn(
                    approximator,
                    config.validation_data,
                    config.n_posterior_samples,
                )
                inference_time = time.perf_counter() - t_val_start
                metrics_summary = _validate_metric_keys(
                    raw, config.objective_metrics,
                    penalty_values=self._metric_penalty_map(),
                )
            else:
                # Default path — call pipeline directly to get pure
                # inference timing from result.timing["inference"].
                from bayesflow_hpo.validation.pipeline import (
                    run_validation_pipeline,
                )

                result = run_validation_pipeline(
                    approximator=approximator,
                    validation_data=config.validation_data,
                    n_posterior_samples=config.n_posterior_samples,
                    # UNION, not restriction. Without the objectives the
                    # pipeline computes only DEFAULT_METRICS, so a configured
                    # objective missing from that list falls through to a
                    # penalty. But restricting it to the objectives is worse:
                    # constraints may reference metrics that cannot be
                    # objectives at all -- `coverage_90` comes from
                    # `coverage`, which is registered diagnostic-only -- and a
                    # hard constraint silently skips a missing key while a
                    # soft one reads it as zero violation. Computing both sets
                    # keeps constraints working and the objectives present.
                    metrics=_pipeline_metrics(
                        config.objective_metrics,
                        _constraint_metric_names(config),
                    ),
                )
                inference_time = result.timing.get("inference", 0.0)
                metrics_summary = _validate_metric_keys(
                    dict(result.summary), config.objective_metrics,
                    penalty_values=self._metric_penalty_map(),
                )

            inference_time_s = compute_inference_time_per_dataset(
                inference_time, n_conditions,
            )
            trial.set_user_attr(
                "inference_time_s", round(inference_time_s, 4),
            )
            for key, val in metrics_summary.items():
                trial.set_user_attr(key, round(float(val), 6))

            # --- Step 8b: Hard metric constraints ---
            if config.metric_constraints_hard is not None:
                penalty = self._check_hard_constraints(metrics_summary, trial)
                if penalty is not None:
                    cleanup_trial()
                    return penalty

            # Wrap for extract_multi_objective_values compatibility.
            metrics = {"summary": metrics_summary}

        except optuna.TrialPruned:
            cleanup_trial()
            raise
        except Exception as exc:
            logger.warning(
                "Trial %d failed during final validation: %s",
                trial.number, exc,
            )
            trial.set_user_attr("validation_error", str(exc))
            values = _training_loss_fallback(
                best_training_loss,
                config.objective_metrics,
                config.objective_mode,
                param_count_actual,
                config.max_param_count,
                config.cost_metric,
                self._penalty(),
            )
            trial.set_user_attr(
                "validation_fallback",
                "training_loss" if best_training_loss is not None else "penalty",
            )
            _log_trial_summary(
                trial, values, param_count_actual,
                training_time,
                objective_metrics=config.objective_metrics,
                cost_metric_name=self._cost_metric_name,
            )
            cleanup_trial()
            return values

        # --- Step 9: Cost score ---
        try:
            param_count = get_param_count(approximator)
        except (TypeError, ValueError) as exc:
            logger.warning("Trial %d: could not count params: %s", trial.number, exc)
            param_count = -1
        trial.set_user_attr("param_count", param_count)

        if config.cost_metric == "inference_time":
            cost_score = compute_inference_time_per_dataset(
                inference_time, n_conditions,
            )
        else:
            cost_score = normalize_param_count(
                param_count,
                max_count=config.max_param_count,
            )

        values = extract_multi_objective_values(
            metrics,
            cost_score,
            objective_metrics=config.objective_metrics,
            objective_mode=config.objective_mode,
        )

        # --- Step 10: Checkpoint pool ---
        self._checkpoint_pool.maybe_save(
            trial_number=trial.number,
            objective_value=mean_objective_score(values),
            approximator=approximator,
        )

        # --- Step 11: Per-trial summary log ---
        _log_trial_summary(
            trial, values, param_count, training_time,
            objective_metrics=config.objective_metrics,
            cost_metric_name=self._cost_metric_name,
        )

        cleanup_trial()
        return values
