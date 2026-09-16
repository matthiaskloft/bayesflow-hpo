"""Pre-flight validation for the HPO pipeline.

``check_pipeline()`` runs a minimal dry-run of the full build → compile
→ train → validate lifecycle to catch interface errors before GPU hours
are wasted.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Callable
from typing import Any

from bayesflow_hpo.builders.workflow import (
    _compile_for_compat,
    _make_cosine_decay_optimizer,
    build_continuous_approximator,
)
from bayesflow_hpo.objectives import _direction_for, canonical_summary
from bayesflow_hpo.optimization.objective import default_train_fn, default_validate_fn
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.types import BuildApproximatorFn, TrainFn, ValidateFn
from bayesflow_hpo.validation.data import generate_validation_dataset
from bayesflow_hpo.validation.registry import (
    canonical_metric_name,
    is_joint_metric,
    validate_objective_metric_kinds,
)

logger = logging.getLogger(__name__)


def _declares_infinity(key: str, value: float) -> bool:
    """Whether *key*'s registered direction declares *value* as its worst case.

    Only an infinity that **matches the registered** ``worst_raw`` exactly is
    declared. ``log_gamma`` registers ``worst_raw=-math.inf``, so ``-inf`` is
    declared and ``+inf`` is not: the metric is ``log(gamma / null_quantile)``
    with ``gamma`` a probability, so it is unbounded *below* and a ``+inf``
    would be an arithmetic fault rather than a bad model.

    ``False`` for an unregistered metric: nothing declares that an infinity is
    meaningful for it, so the pre-flight keeps refusing one. Resolution goes
    through :func:`~bayesflow_hpo.objectives._direction_for` rather than
    :data:`~bayesflow_hpo.objectives.METRIC_DIRECTIONS` directly, so a metric
    whose direction was removed via the legacy ``HIGHER_IS_BETTER`` set is
    treated as unregistered here too.

    *key* is canonicalized here rather than assumed canonical. Callers in this
    module have already canonicalized, and the operation is idempotent, so this
    costs nothing -- but it keeps the raw/canonical distinction that
    :data:`~bayesflow_hpo.validation.registry.CanonicalMetricName` exists to
    enforce from
    depending on where the helper is called from.

    References
    ----------
    The gamma discrepancy — the probability, under uniform ranks, of the most
    extreme point of the observed rank ECDF — is Säilynoja, T., Bürkner, P.-C.,
    & Vehtari, A. (2022). Graphical test for discrete uniformity and its
    applications in goodness-of-fit evaluation and multiple sample comparison.
    *Statistics and Computing, 32*(2). https://doi.org/10.1007/s11222-022-10090-6

    Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N., Faltejsková, K.,
    Gelman, A., & Vehtari, A. (2025). Simulation-based calibration checking for
    Bayesian computation: The choice of test quantities shapes sensitivity.
    *Bayesian Analysis, 20*(2), 461-488. https://doi.org/10.1214/23-BA1404
    adopt it in Section 4.1 and define the quantity BayesFlow's
    ``calibration_log_gamma`` reports, ``log(gamma / gamma_bar)`` with
    ``gamma_bar`` the 5th percentile of the null distribution. A rank
    distribution extreme enough to drive ``gamma`` to ``0.0`` yields ``-inf``.
    """
    direction = _direction_for(canonical_metric_name(key))
    if direction is None:
        return False
    return value == direction.worst_raw


class PipelineError(Exception):
    """Raised when ``check_pipeline()`` detects an interface mismatch.

    Common causes:

    - Adapter missing ``Rename``/``Concatenate`` transforms targeting
      ``inference_variables`` or ``summary_variables``.
    - Custom ``build_approximator_fn`` signature does not accept exactly
      1 positional argument.
    - Builder returns an object without ``fit`` / ``sample`` methods.
    - ``validate_fn`` output missing required metric keys.
    """


class _MockTrial:
    """Lightweight Optuna trial stub for sampling dummy hparams."""

    def suggest_int(
        self, name: str, low: int, high: int,
        step: int | None = None, log: bool = False,
    ) -> int:
        return low

    def suggest_float(
        self, name: str, low: float, high: float,
        log: bool = False,
    ) -> float:
        return low

    def suggest_categorical(self, name: str, choices: list) -> Any:
        return choices[0]


class _TrackingDict(dict):
    """Dict wrapper that records which keys are accessed.

    Tracks ``__getitem__``, ``get``, ``__contains__``, ``pop``,
    ``items()``, and ``values()`` so that unused-key detection works
    regardless of how the builder accesses parameters.

    Note: ``__iter__`` is intentionally **not** overridden because
    ``dict(tracking_dict)`` calls ``__iter__`` internally, which would
    falsely mark all keys as accessed.

    The price of that choice is that a consumer which copies the dict
    (``dict(hparams)``, ``{**hparams}``) reads the copy, not this object,
    and its accesses are invisible here. Unused-key reporting must
    therefore stay advisory rather than prescriptive.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed_keys: set[str] = set()

    def __getitem__(self, key):
        self.accessed_keys.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed_keys.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        # Only track if the key actually exists — defensive checks like
        # ``if "optional" in hparams`` should not suppress unused warnings.
        if super().__contains__(key):
            self.accessed_keys.add(key)
        return super().__contains__(key)

    def pop(self, key, *args):
        self.accessed_keys.add(key)
        return super().pop(key, *args)

    def items(self):
        self.accessed_keys.update(self.keys())
        return super().items()

    def values(self):
        self.accessed_keys.update(self.keys())
        return super().values()


def _check_hook_arity(fn: Callable[..., Any], expected: int, name: str) -> None:
    """Raise ``PipelineError`` if *fn* doesn't accept *expected* positional args."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return  # Can't inspect (e.g. built-in) — skip check.

    # Count parameters that can accept a positional argument.
    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    if has_var_positional:
        logger.debug("%s uses *args — skipping arity check for %r", name, fn)
        return

    positional_params = [
        p for p in sig.parameters.values() if p.kind in positional_kinds
    ]
    n_required = sum(
        1 for p in positional_params if p.default is inspect.Parameter.empty
    )
    n_positional = len(positional_params)
    if not (n_required <= expected <= n_positional):
        raise PipelineError(
            f"{name} must accept exactly {expected} positional argument(s) "
            f"(hooks are called positionally), but its signature requires "
            f"{n_required} and allows at most {n_positional}: {sig}"
        )


def check_pipeline(
    simulator: Any,
    adapter: Any,
    search_space: CompositeSearchSpace,
    build_approximator_fn: BuildApproximatorFn | None = None,
    train_fn: TrainFn | None = None,
    validate_fn: ValidateFn | None = None,
    objective_metrics: list[str] | None = None,
    sims_per_condition: int = 5,
    n_posterior_samples: int = 2,
    validation_conditions: dict[str, list[Any]] | None = None,
    epochs: int = 1,
    num_batches: int = 1,
) -> None:
    """Dry-run the full pipeline to catch interface errors early.

    Uses intentionally minimal defaults (1 epoch, 1 batch, 5 sims) for
    speed.  This validates interfaces and shapes but is **not** a full
    fidelity check — a config that passes here can still OOM or diverge
    under ``optimize()``'s larger budget.

    Steps:

    1. Sample dummy hparams from ``search_space`` (using a mock trial).
    2. Call ``build_approximator_fn`` (or default) — verify result has
       ``fit``, ``compute_loss``/``build_from_data``, and ``sample``
       methods (duck-typed).
    3. Generate a tiny validation dataset (``sims_per_condition=5``).
    4. Compile and run one training step (1 epoch, 1 batch).
    5. Call ``validate_fn`` (or default) — verify it returns
       ``dict[str, float]`` whose keys include all ``objective_metrics``.
    6. Warn about sampled hparam keys not consumed by the builder.

    Parameters
    ----------
    simulator
        BayesFlow simulator.
    adapter
        BayesFlow adapter.
    search_space
        Composite search space.
    build_approximator_fn
        Optional custom builder.
    train_fn
        Optional custom training function.
    validate_fn
        Optional custom validation function.
    objective_metrics
        Metric keys the objective expects. Default
        ``["calibration_error", "nrmse"]``.
    sims_per_condition
        Simulations per condition for tiny validation dataset.
    n_posterior_samples
        Posterior draws for validation dry run.
    validation_conditions
        Optional condition grid for validation data generation.
    epochs
        Training epochs for dry run (default 1).
    num_batches
        Batches per epoch for dry run (default 1).

    Raises
    ------
    PipelineError
        With a clear message identifying which component failed and why.
    ValueError
        If ``objective_metrics`` contains a registered diagnostic metric.
    """
    if objective_metrics is None:
        objective_metrics = ["calibration_error", "nrmse"]
    # Public entry point, so canonicalize here too: `optimize()` already hands
    # us canonical names, but a direct caller may not.
    objective_metrics = [canonical_metric_name(m) for m in objective_metrics]
    validate_objective_metric_kinds(objective_metrics)

    # --- Step 0: Validate hook signatures ---
    if build_approximator_fn is not None:
        _check_hook_arity(build_approximator_fn, 1, "build_approximator_fn")
    if train_fn is not None:
        _check_hook_arity(train_fn, 4, "train_fn")
    if validate_fn is not None:
        _check_hook_arity(validate_fn, 3, "validate_fn")

    # --- Step 1: Sample dummy hparams ---
    try:
        raw_hparams = search_space.sample(_MockTrial())
    except Exception as exc:
        raise PipelineError(
            f"Failed to sample hparams from search_space: {exc}"
        ) from exc

    hparams = _TrackingDict(raw_hparams)
    hparams["epochs"] = epochs
    hparams["num_batches"] = num_batches

    # --- Step 2: Build approximator ---
    try:
        if build_approximator_fn is not None:
            approximator = build_approximator_fn(hparams)
        else:
            approximator = build_continuous_approximator(hparams, adapter, search_space)
    except Exception as exc:
        raise PipelineError(f"Build step failed: {exc}") from exc

    if not hasattr(approximator, "fit"):
        raise PipelineError(
            f"Builder returned {type(approximator).__name__} which has no "
            f"'fit' method. The approximator must support .fit()."
        )

    # When using default train/validate, the objective's param-probe
    # calls compute_loss or build_from_data, and validation calls sample.
    if build_approximator_fn is not None:
        if (
            not hasattr(approximator, "compute_loss")
            and not hasattr(approximator, "build_from_data")
        ):
            raise PipelineError(
                f"Builder returned {type(approximator).__name__} which "
                f"has neither 'compute_loss' nor 'build_from_data'. "
                f"The objective uses these for parameter counting."
            )
        if validate_fn is None and not hasattr(approximator, "sample"):
            raise PipelineError(
                f"Builder returned {type(approximator).__name__} which "
                f"has no 'sample' method. The default validation "
                f"pipeline requires .sample(). Provide a custom "
                f"validate_fn if your approximator uses a different "
                f"inference API."
            )

    # --- Step 3: Generate tiny validation dataset ---
    from bayesflow_hpo.api import infer_keys_from_adapter

    adapter_keys = infer_keys_from_adapter(adapter)
    param_keys = adapter_keys.get("param_keys")
    data_keys = adapter_keys.get("data_keys")

    if param_keys is None or data_keys is None:
        raise PipelineError(
            "Could not infer param_keys and/or data_keys from the "
            "adapter. Ensure the adapter has Rename/Concatenate "
            "transforms targeting 'inference_variables' and "
            "'summary_variables'."
        )

    try:
        validation_data = generate_validation_dataset(
            simulator=simulator,
            param_keys=param_keys,
            data_keys=data_keys,
            condition_grid=validation_conditions,
            sims_per_condition=sims_per_condition,
        )
    except Exception as exc:
        raise PipelineError(f"Validation dataset generation failed: {exc}") from exc

    # --- Step 4: Compile ---
    if train_fn is None and "initial_lr" not in raw_hparams:
        raise PipelineError(
            "Search space does not sample 'initial_lr', which is required "
            "for the default Adam compile step. Either add "
            "'initial_lr' to your search space (e.g. via TrainingSpace) or "
            "provide a custom train_fn that compiles with its own optimizer."
        )
    initial_lr = float(hparams.get("initial_lr", 1e-3))
    decay_steps = num_batches * epochs
    try:
        optimizer = _make_cosine_decay_optimizer(initial_lr, decay_steps)
        _compile_for_compat(approximator, optimizer)
    except TypeError:
        pass  # _compile_for_compat handles TypeError internally
    except Exception as exc:
        raise PipelineError(
            f"Compile step failed: {exc}"
        ) from exc

    # --- Step 5: Train one step ---
    actual_train_fn = train_fn if train_fn is not None else default_train_fn
    # A tracking copy, not a plain one: a custom train_fn is a documented
    # consumer of the same hparams dict (``batch_size`` is the common case),
    # and its reads have to count toward the unused-key report in Step 7.
    # Copying still isolates the hook from the builder's dict.
    train_hparams = _TrackingDict(dict(hparams))
    try:
        actual_train_fn(approximator, simulator, train_hparams, [])
    except Exception as exc:
        raise PipelineError(f"Training step failed: {exc}") from exc

    # --- Step 6: Validate ---
    # JOINT metrics are excluded from pre-flight, and this is a deliberate
    # limit rather than an oversight.
    #
    # Pre-flight generates its own tiny batch -- five simulations per
    # condition by default -- to check interfaces cheaply. A joint metric's
    # configuration is sized for the PRODUCTION batch, so running the
    # caller's real callable against that batch fails for reasons that say
    # nothing about the configuration: `make_lc2st_joint_metric(n_folds=10)`
    # is rejected because five simulations cannot fill ten folds, and a
    # per-condition reference array shaped for 500 rows does not match five
    # -- and truncating it would pair references with different, newly
    # generated observations, which is a silently wrong check rather than a
    # failed one. A pre-flight that always rejects a valid configuration is
    # worse than one that does not examine it.
    #
    # The cost is that a joint metric's own interface errors surface at the
    # first trial instead of before the study. The metric is still resolved
    # there, `resolve_joint_metrics` still refuses an unconfigured
    # placeholder, and `JointMetricConfigurationError` is re-raised rather
    # than penalized -- so the study still stops on the first trial with the
    # real message, one trial later than it might have.
    marginal_metrics = [
        name for name in objective_metrics if not is_joint_metric(name)
    ]
    # The restriction applies to the BUILT-IN validator only. A custom
    # `validate_fn` computes whatever it likes on whatever batch it is
    # given, so nothing about the tiny pre-flight batch excuses it from
    # producing the objective keys it was configured for -- and narrowing
    # the requirement for it would be worse than not checking: with
    # `objective_metrics=["tarp_error"]` alone, `marginal_metrics` is empty,
    # so pre-flight would verify nothing at all and a hook silently omitting
    # the key would take a penalty on every trial.
    required_metrics = (
        objective_metrics if validate_fn is not None else marginal_metrics
    )
    try:
        if validate_fn is not None:
            # A custom hook keeps the documented 3-argument contract.
            result = validate_fn(
                approximator, validation_data, n_posterior_samples
            )
        else:
            # The built-in validator has to be told which metrics this run
            # optimizes, or it computes DEFAULT_METRICS only and the missing
            # key check below rejects every non-default objective.
            result = default_validate_fn(
                approximator,
                validation_data,
                n_posterior_samples,
                objective_metrics=marginal_metrics,
            )
    except Exception as exc:
        raise PipelineError(f"Validation step failed: {exc}") from exc

    if not isinstance(result, dict):
        raise PipelineError(
            f"validate_fn must return dict[str, float], got {type(result).__name__}"
        )

    # A custom hook returns the spelling its caller asked for, which may be an
    # alias; `objective_metrics` is already canonical. Meet in canonical space
    # so pre-flight does not reject a hook honouring the documented contract.
    #
    # `canonical_summary`, not a comprehension: a comprehension is
    # last-write-wins, so a hook emitting both spellings of one metric -- one
    # finite, one not -- passed pre-flight or was rejected by it depending on
    # nothing but insertion order. This was the fourth such boundary; the other
    # three were fixed together and this one was missed.
    result = canonical_summary(result)

    missing_keys = set(required_metrics) - set(result.keys())
    if missing_keys:
        raise PipelineError(
            f"validate_fn output is missing required metric keys: "
            f"{sorted(missing_keys)}. Got keys: {sorted(result.keys())}"
        )

    for key in required_metrics:
        val = result[key]
        if not isinstance(val, (int, float)) or math.isnan(val):
            raise PipelineError(
                f"validate_fn returned non-finite value for {key!r}: {val}"
            )
        # An infinity is allowed only where it matches the metric's registered
        # ``worst_raw`` exactly. ``log_gamma`` registers ``-math.inf``, on the
        # reasoning that the metric is unbounded below and no finite constant
        # is defensibly its worst -- so refusing -inf here rejected the very
        # value the objective is built to represent. ``+inf`` for that same
        # metric stays refused: it is unbounded below, not above.
        #
        # It rejected it exactly where the pre-flight is meant to help. This
        # runs at ``n_posterior_samples=2`` on a barely-trained model, where
        # the SBC ranks are maximally non-uniform, ``gamma_discrepancy``
        # returns 0.0 and ``log(0)`` is -inf: not a broken hook, but the
        # correct value of a correct metric on a deliberately degenerate
        # input. And because the underflow needs enough ranks for the binomial
        # tails to reach zero, it surfaced only on realistically sized
        # validation sets -- pre-flight passed on toy configurations and
        # failed on real ones.
        #
        # NaN stays refused above: no metric declares it, and it is the
        # signature of an arithmetic mistake rather than of a bad model.
        if math.isinf(val) and not _declares_infinity(key, val):
            raise PipelineError(
                f"validate_fn returned non-finite value for {key!r}: {val}"
            )

    # --- Step 7: Warn about unused hparams ---
    if build_approximator_fn is not None:
        sampled_keys = set(raw_hparams.keys())
        # Union across every hook handed a tracking dict. Without the train
        # hook's reads, a key consumed only by a custom train_fn (e.g.
        # ``batch_size``) is reported as dead, and acting on that report
        # would delete a live search dimension.
        accessed = hparams.accessed_keys | train_hparams.accessed_keys
        unused = sampled_keys - accessed
        if unused:
            logger.warning(
                "Search space sampled keys that were never read by "
                "build_approximator_fn or train_fn: %s. Check whether they "
                "are still needed — a hook that copies the hparams dict "
                "(dict(hparams), {**hparams}) reads the copy, so its reads "
                "cannot be seen here and the key may well be in use.",
                sorted(unused),
            )
