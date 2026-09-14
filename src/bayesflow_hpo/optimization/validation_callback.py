"""Periodic validation callback for mid-training pruning.

Runs a lightweight validation every *interval* epochs and uses a
pluggable pruning strategy for multi-objective studies.

For single-objective studies, the standard ``trial.report()`` /
``trial.should_prune()`` API is used with the study's pruner.

For multi-objective studies (the default in bayesflow_hpo), Optuna
does not support ``trial.report()`` (Issue #3450, open since April
2022).  Instead, one of three custom pruning strategies is applied:

- ``"dominance"`` — per-objective normalized median check (AND rule).
  Simplified adaptation of the dominance-based selection idea in
  MO-ASHA's Algorithm 1 selector (Schmucker et al., 2021); the median
  rule itself is ours, not the paper's.
- ``"mo-sha"`` — non-dominated sorting at each step, bottom-fraction
  pruning per MO-ASHA Algorithm 2 (Schmucker et al., 2021).
- ``"primary"`` — single-metric median pruning on a user-chosen
  objective (equivalent to Optuna's ``MedianPruner``; Optuna API
  reference, not Akiba et al., 2019 -- see
  :mod:`~bayesflow_hpo.optimization.pruning_strategies`).

Strategy implementations live in
:mod:`bayesflow_hpo.optimization.pruning_strategies`.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import optuna
from keras.callbacks import Callback

from bayesflow_hpo.objectives import (
    RawScore,
    _metric_to_minimize,
    canonical_summary,
)
from bayesflow_hpo.optimization.pruning_strategies import (
    should_prune_dominance,
    should_prune_mo_sha,
    should_prune_primary,
)
from bayesflow_hpo.types import ValidateFn
from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.registry import (
    CanonicalMetricName,
    canonical_metric_name,
    is_joint_metric,
)

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = {"none", "dominance", "mo-sha", "primary"}

#: Fallback used when ``ObjectiveConfig.pruning_n_startup_trials`` is left
#: unresolved. ``optimize()`` normally auto-detects it from the sampler, but
#: building an objective directly bypasses that, and the pruning strategies
#: compare this against an int.
DEFAULT_PRUNING_N_STARTUP_TRIALS = 5


class PeriodicValidationCallback(Callback):
    """Run validation every *interval* epochs and report to Optuna.

    For single-objective studies the first metric in ``objective_metrics``
    is reported via ``trial.report()`` and pruning uses the study's
    pruner.  For multi-objective studies (where ``trial.report()`` is
    unsupported), per-metric user attributes are stored and the
    configured ``pruning_strategy`` decides whether to prune.

    Parameters
    ----------
    trial
        Current Optuna trial.
    approximator
        Trained approximator with a ``.sample()`` method (updated
        in-place during training).
    validation_data
        Pre-generated
        :class:`~bayesflow_hpo.validation.data.ValidationDataset`.
    interval
        Run validation every *interval* epochs.  Default 10.
    warmup
        Skip the first *warmup* epochs before running validation.
        Default 10.
    n_posterior_samples
        Number of posterior draws for intermediate validation.
        Default 250.
    n_startup_trials
        Minimum completed trials before multi-objective pruning
        activates.  ``None`` (the default) resolves to
        :data:`DEFAULT_PRUNING_N_STARTUP_TRIALS`, because
        ``ObjectiveConfig.pruning_n_startup_trials`` is auto-detected by
        ``optimize()`` and stays ``None`` when an objective is built
        directly.
    validate_fn
        Optional custom validation function with signature
        ``(approximator, validation_data, n_posterior_samples) ->
        dict[str, float]``.  When provided, replaces the default
        ``run_validation_pipeline`` for intermediate pruning.  The
        returned dict must contain all keys in ``objective_metrics``.
    pruning_strategy
        Multi-objective pruning strategy.  One of ``"dominance"``
        (default), ``"mo-sha"``, ``"primary"``, or ``"none"``.
        For ``"primary"``, pass a tuple ``("primary", metric_name)``
        to specify which metric to prune on (defaults to
        ``objective_metrics[0]``).
    objective_metrics
        Metric keys to compute during intermediate validation.
        Defaults to ``["calibration_error", "nrmse"]``.
    early_stopping_patience
        Validation checks without improvement before stopping. ``None``
        disables early stopping. This is intended for horizon-free schedules;
        finite-budget schedules should run to their horizon.
    early_stopping_window
        Number of validation scores in the moving average.
    early_stopping_monitor
        Validation objective used for stopping. ``"objective_mean"`` (default) averages
        all objective metrics after converting them to minimize-is-better
        values. A metric name selects that metric alone.
    joint_metrics
        Configured joint metrics forwarded to the validation pipeline, for
        names that cannot run at a registry default. Only consulted when
        the metric is in the intermediate set at all.
    include_joint_metrics
        Whether joint metrics in *objective_metrics* are computed at each
        intermediate validation. ``False`` (default) excludes them.

        Joint metrics are expensive enough to change what this callback is
        for: L-C2ST measured ~54 s per condition at 500 simulations with 15
        parameters, so a 20-condition grid spends ~18 minutes per interval
        deciding whether to prune a trial. A pruning decision that costs
        more than the training it might save is not a pruning decision.
        TARP, by contrast, is ~79 ms per condition -- three to four orders
        of magnitude cheaper -- so opting in is reasonable for some joint
        metrics and not others. See ``docs/plans/plan-joint-metric-path.md``
        D9 for the measurements.

        Excluding them changes what the intermediate signal MEANS, which is
        why the exclusion is explicit rather than implied: with
        ``early_stopping_monitor="objective_mean"`` the mid-training average
        is taken over the remaining metrics only, and is therefore not on
        the same scale as the final objective mean.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        approximator: Any,
        validation_data: ValidationDataset,
        interval: int = 10,
        warmup: int = 10,
        n_posterior_samples: int = 250,
        n_startup_trials: int | None = None,
        validate_fn: ValidateFn | None = None,
        pruning_strategy: str | tuple[str, str] = "dominance",
        objective_metrics: list[str] | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_window: int = 1,
        early_stopping_monitor: str = "objective_mean",
        include_joint_metrics: bool = False,
        joint_metrics: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.trial = trial
        self.approximator = approximator
        self.validation_data = validation_data
        self.interval = interval
        self.warmup = warmup
        self.n_posterior_samples = n_posterior_samples
        # `optimize()` auto-detects this from the sampler, but building an
        # objective directly leaves it None, and every pruning strategy
        # compares it against an int.
        self.n_startup_trials = (
            DEFAULT_PRUNING_N_STARTUP_TRIALS
            if n_startup_trials is None
            else n_startup_trials
        )
        self.validate_fn = validate_fn
        self._step = 0  # monotonic step counter for Optuna
        self._consecutive_failures = 0
        self._is_multi_objective = len(trial.study.directions) > 1
        # Every strategy in `pruning_strategies` compares several objectives,
        # so a one-direction study falls through to Optuna's own pruner and
        # the requested strategy never runs. Reachable since `cost_metric`
        # became optional: mean mode, or pareto over a single metric, now
        # yields one direction where the count was previously always >= 2.
        if not self._is_multi_objective and pruning_strategy != "dominance":
            logger.warning(
                "pruning_strategy=%r is ignored: this study has a single "
                "objective direction, and every multi-objective strategy "
                "needs at least two. Optuna's own pruner (set via "
                "create_study(pruner=...)) decides pruning instead.",
                pruning_strategy,
            )
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_window = early_stopping_window
        # Canonicalized for the same reason `ObjectiveConfig.__post_init__`
        # does it: this class is public and constructible directly, the
        # validation summary is keyed by canonical name, and an alias here
        # therefore missed every lookup. `objective_mean` is a sentinel, not a
        # metric, so it is passed through untouched.
        self.early_stopping_monitor = (
            early_stopping_monitor
            if early_stopping_monitor == "objective_mean"
            else canonical_metric_name(early_stopping_monitor)
        )
        self._early_stopping_values: list[float] = []
        self._early_stopping_wait = 0
        self.best_validation_score = np.inf
        self.best_weights: Any = None

        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1 or None.")
        if early_stopping_window < 1:
            raise ValueError("early_stopping_window must be >= 1.")

        # Parse pruning strategy.
        if isinstance(pruning_strategy, tuple):
            if (
                len(pruning_strategy) != 2
                or pruning_strategy[0] != "primary"
            ):
                raise ValueError(
                    f"Tuple pruning_strategy must be "
                    f"('primary', metric_name), got {pruning_strategy!r}"
                )
            self._strategy_name = "primary"
            self._primary_metric: CanonicalMetricName | None = (
                canonical_metric_name(pruning_strategy[1])
            )
        else:
            if pruning_strategy not in _VALID_STRATEGIES:
                raise ValueError(
                    f"Unknown pruning_strategy: {pruning_strategy!r}. "
                    f"Expected one of {sorted(_VALID_STRATEGIES)}."
                )
            self._strategy_name = pruning_strategy
            self._primary_metric = None

        # Resolve objective_metrics with backward-compatible default.
        # Canonicalized BEFORE the membership check below, so that a caller
        # pairing an alias in both fields -- objective_metrics=["cal_error"]
        # with early_stopping_monitor="cal_error" -- still validates.
        self.objective_metrics: list[CanonicalMetricName] = [
            canonical_metric_name(m)
            for m in (
                objective_metrics
                if objective_metrics is not None
                else ["calibration_error", "nrmse"]
            )
        ]
        if (
            self.early_stopping_monitor != "objective_mean"
            and self.early_stopping_monitor not in self.objective_metrics
        ):
            raise ValueError(
                "early_stopping_monitor must be 'objective_mean' or one of "
                "objective_metrics, got "
                f"{self.early_stopping_monitor!r}."
            )

        # The metrics this callback actually computes per interval, as an
        # EXPLICIT set rather than one implied by `objective_metrics`.
        #
        # Simply omitting a joint metric from the pipeline call would not
        # work: `_run_lightweight_validation` requires every entry of
        # `objective_metrics` to be present and returns None when any is
        # missing, and `on_epoch_end` then bails before both pruning AND
        # `_update_early_stopping`. For objective_metrics=["nrmse",
        # "lc2st"], excluding `lc2st` to save time would silently disable
        # marginal pruning and validation early stopping as well -- a cost
        # optimization that turns off stopping is a regression, not a
        # saving. Naming the set is what makes an absent joint key expected
        # rather than a fault.
        # Default primary metric to first objective metric. MUST precede
        # the excluded-metric guards below: `pruning_strategy="primary"`
        # (the bare string) leaves `_primary_metric` None until here, so a
        # guard running first would see None, wave it through, and let the
        # default land on an excluded joint metric -- surfacing later as an
        # uncaught KeyError from `_evaluate_pruning` mid-training rather
        # than the clear refusal the guard exists to give.
        if self._strategy_name == "primary" and self._primary_metric is None:
            self._primary_metric = self.objective_metrics[0]

        self.joint_metrics = joint_metrics
        self.include_joint_metrics = include_joint_metrics
        self.intermediate_metrics: list[CanonicalMetricName] = [
            m
            for m in self.objective_metrics
            if include_joint_metrics or not is_joint_metric(m)
        ]
        excluded = [
            m for m in self.objective_metrics
            if m not in self.intermediate_metrics
        ]

        if not self.intermediate_metrics:
            # A joint-only study. Every interval would compute nothing, so
            # there is no intermediate signal to prune or stop on at all.
            # Rejected up front rather than degraded into a study that
            # silently cannot stop early: the caller either opts in and pays
            # the cost knowingly, or drops the callback.
            raise ValueError(
                "Every objective metric is a joint metric "
                f"({[str(m) for m in self.objective_metrics]}), so "
                "intermediate validation would compute nothing and this "
                "callback could neither prune nor stop early. Pass "
                "include_joint_metrics=True to pay their cost at every "
                "interval, add a cheap marginal objective, or do not use "
                "PeriodicValidationCallback for this study."
            )

        if excluded:
            if self.early_stopping_monitor in excluded:
                raise ValueError(
                    f"early_stopping_monitor={self.early_stopping_monitor!r} "
                    "is a joint metric excluded from intermediate "
                    "validation, so nothing would ever be monitored. Pass "
                    "include_joint_metrics=True, or monitor a metric that "
                    "is computed at each interval: "
                    f"{[str(m) for m in self.intermediate_metrics]}."
                )
            if self._primary_metric in excluded:
                raise ValueError(
                    f"pruning_strategy=('primary', {self._primary_metric!r}) "
                    "names a joint metric excluded from intermediate "
                    "validation, so no pruning decision could ever be made. "
                    "Pass include_joint_metrics=True, or choose a primary "
                    "metric that is computed at each interval: "
                    f"{[str(m) for m in self.intermediate_metrics]}."
                )
            if self.early_stopping_monitor == "objective_mean":
                # Not an error, but it silently changes what is monitored:
                # the members of the mean differ between mid-training and
                # the final objective, so the two are not comparable.
                logger.info(
                    "Joint metric(s) %s are excluded from intermediate "
                    "validation, so 'objective_mean' averages %s here while "
                    "the final objective averages all of %s. Pass "
                    "include_joint_metrics=True to include them.",
                    [str(m) for m in excluded],
                    [str(m) for m in self.intermediate_metrics],
                    [str(m) for m in self.objective_metrics],
                )


    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        """Run validation and check for pruning at scheduled intervals.

        Skips epochs before ``warmup`` and non-interval epochs.
        After 3 consecutive validation failures, logs a warning
        (but does not prune — the trial continues without pruning).
        """
        if epoch < self.warmup:
            return
        if (epoch - self.warmup) % self.interval != 0:
            return

        raw_scores = self._run_lightweight_validation()
        if raw_scores is None:
            self._consecutive_failures += 1
            if self._consecutive_failures == 3:
                logger.warning(
                    "Trial %d: %d consecutive intermediate validation "
                    "failures — pruning may be ineffective.",
                    self.trial.number,
                    self._consecutive_failures,
                )
            return
        self._consecutive_failures = 0

        self._step += 1

        self._update_early_stopping(raw_scores)
        scores: dict[str, float] = {
            metric: _metric_to_minimize(
                metric, RawScore(float(raw_scores[metric]))
            )
            for metric in self.intermediate_metrics
        }

        if self._is_multi_objective:
            # Store per-metric user attrs for strategy functions.
            for metric, val in scores.items():
                self.trial.set_user_attr(
                    f"val_{metric}_step_{self._step}",
                    round(float(val), 6),
                )

            should_prune = self._evaluate_pruning(scores)
            if should_prune:
                raise optuna.TrialPruned()
        else:
            # Single-objective: report the study's actual objective to
            # Optuna's own pruner. That is the MEAN of the converted scores,
            # not `objective_metrics[0]`: a one-direction study over several
            # metrics is mean mode, whose objective is exactly this average
            # (`objectives.extract_multi_objective_values`). Reporting the
            # first metric instead pruned on a quantity the study does not
            # optimize. With a single metric the mean is that metric, so one
            # expression is right in both cases.
            primary_val = float(np.mean(list(scores.values())))
            # Reported even when pruning is off: the intermediate values are
            # telemetry in their own right, and reporting alone prunes
            # nothing.
            self.trial.report(primary_val, step=self._step)
            # `_evaluate_pruning` returns False for "none", so the
            # multi-objective branch has always honoured it. This branch
            # consulted Optuna's pruner unconditionally, so a study that
            # asked for no pruning still got the default MedianPruner's
            # verdict -- and `optimize()` installs this callback whenever
            # early stopping is on, including with `pruning_strategy="none"`.
            if self._strategy_name != "none" and self.trial.should_prune():
                raise optuna.TrialPruned()

    def _update_early_stopping(
        self, raw_scores: dict[str, float]
    ) -> None:
        """Stop on a moving average of the configured validation objective.

        Takes RAW pipeline values and converts them here. The parameter used
        to be called `scores`, which reads as the already-converted dict of
        the same name in the caller -- the two spaces are indistinguishable by
        inspection, so the name was the only thing distinguishing them.
        """
        if self.early_stopping_patience is None:
            return

        if self.early_stopping_monitor == "objective_mean":
            value = float(
                np.mean(
                    [
                        _metric_to_minimize(
                            metric, RawScore(float(raw_scores[metric]))
                        )
                        for metric in self.intermediate_metrics
                    ]
                )
            )
        else:
            # Not the `objective_mean` sentinel, so `__init__` put a
            # canonical name here; the attribute is a plain `str` because it
            # holds either.
            monitor = cast(CanonicalMetricName, self.early_stopping_monitor)
            value = _metric_to_minimize(
                monitor, RawScore(float(raw_scores[monitor]))
            )
        self._early_stopping_values.append(value)
        if len(self._early_stopping_values) > self.early_stopping_window:
            self._early_stopping_values.pop(0)
        moving_average = float(np.mean(self._early_stopping_values))

        if moving_average < self.best_validation_score:
            self.best_validation_score = moving_average
            self._early_stopping_wait = 0
            self.best_weights = self.approximator.get_weights()
            return

        self._early_stopping_wait += 1
        if self._early_stopping_wait >= self.early_stopping_patience:
            self.approximator.stop_training = True
            if self.best_weights is not None:
                self.approximator.set_weights(self.best_weights)

    def on_train_end(self, logs: Any = None) -> None:
        """Restore the best validation weights when training reaches its cap."""
        if self.early_stopping_patience is not None and self.best_weights is not None:
            self.approximator.set_weights(self.best_weights)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_pruning(self, scores: dict[str, float]) -> bool:
        """Dispatch to the configured pruning strategy."""
        if self._strategy_name == "none":
            return False
        if self._strategy_name == "dominance":
            return should_prune_dominance(
                self.trial, scores, self._step, self.n_startup_trials
            )
        if self._strategy_name == "mo-sha":
            return should_prune_mo_sha(
                self.trial, scores, self._step, self.n_startup_trials
            )
        if self._strategy_name == "primary":
            if self._primary_metric is None:  # pragma: no cover - set together
                raise RuntimeError(
                    'pruning_strategy "primary" without a metric name.'
                )
            primary_score = scores[self._primary_metric]
            return should_prune_primary(
                self.trial,
                float(primary_score),
                self._primary_metric,
                self._step,
                self.n_startup_trials,
            )
        return False  # pragma: no cover

    def _run_lightweight_validation(self) -> dict[str, float] | None:
        """Compute objective_metrics via validation pipeline."""
        try:
            if self.validate_fn is not None:

                raw_result = self.validate_fn(
                    self.approximator,
                    self.validation_data,
                    self.n_posterior_samples,
                )
                # A hook returns the spelling its caller asked for, while
                # `objective_metrics` was canonicalized at the API boundary.
                # This is the third call site of that mismatch, after
                # pre-flight and final validation: here every scheduled
                # validation read as "missing", disabling pruning and -- in
                # open_ended mode -- validation early stopping and best-weight
                # restoration, behind nothing louder than a warning log.
                # Collision-aware, for the reason given in
                # `_validate_metric_keys`: a comprehension here made pruning
                # read whichever spelling the hook happened to emit last.
                result_dict = canonical_summary(raw_result)
                # Validate that all objective_metrics are present.
                missing = [
                    k for k in self.intermediate_metrics
                    if k not in result_dict
                ]
                if missing:
                    logger.warning(
                        "validate_fn output missing metrics %s — "
                        "skipping pruning this step.",
                        missing,
                    )
                    return None
                out: dict[str, float] = {
                    k: float(result_dict[k])
                    for k in self.intermediate_metrics
                }
                return out
            else:
                from bayesflow_hpo.validation.pipeline import (
                    run_validation_pipeline,
                )

                result = run_validation_pipeline(
                    approximator=self.approximator,
                    validation_data=self.validation_data,
                    n_posterior_samples=self.n_posterior_samples,
                    metrics=self.intermediate_metrics,
                    joint_metrics=self.joint_metrics,
                )
                extracted: dict[str, float] = {
                    k: float(result.summary[k])
                    for k in self.intermediate_metrics
                    if k in result.summary
                }
                # `intermediate_metrics`, NOT `objective_metrics`. This
                # is the branch taken when no `validate_fn` is supplied --
                # the default -- and requiring every objective key here
                # defeats the whole exclusion: the joint key is absent by
                # design, so this returned None on every interval and
                # `on_epoch_end` bailed before pruning AND
                # `_update_early_stopping`. Exactly the regression the
                # explicit set exists to prevent, reintroduced one branch
                # over from where it was fixed.
                missing = [
                    k for k in self.intermediate_metrics
                    if k not in extracted
                ]
                if missing:
                    logger.warning(
                        "run_validation_pipeline output missing "
                        "metrics %s — skipping pruning this step.",
                        missing,
                    )
                    return None
                return extracted
        except optuna.TrialPruned:
            raise
        except Exception:
            logger.warning(
                "Intermediate validation failed (trial %d)",
                self.trial.number,
                exc_info=True,
            )
            return None
