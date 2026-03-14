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
from bayesflow_hpo.optimization.objective import default_train_fn, default_validate_fn
from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace
from bayesflow_hpo.types import BuildApproximatorFn, TrainFn, ValidateFn
from bayesflow_hpo.validation.data import generate_validation_dataset

logger = logging.getLogger(__name__)


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

    Tracks ``__getitem__``, ``get``, ``__contains__``, and ``pop``
    so that unused-key detection works regardless of how the builder
    accesses parameters.

    Note: ``__iter__``, ``items()``, and ``values()`` are intentionally
    **not** overridden because ``dict(tracking_dict)`` calls ``__iter__``
    internally, which would falsely mark all keys as accessed.
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
        return  # *args — can't determine arity statically.

    n_positional = sum(
        1 for p in sig.parameters.values() if p.kind in positional_kinds
    )
    if n_positional != expected:
        raise PipelineError(
            f"{name} must accept exactly {expected} positional argument(s) "
            f"(hooks are called positionally), but its signature has "
            f"{n_positional}: {sig}"
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
    batches_per_epoch: int = 1,
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
    batches_per_epoch
        Batches per epoch for dry run (default 1).

    Raises
    ------
    PipelineError
        With a clear message identifying which component failed and why.
    """
    if objective_metrics is None:
        objective_metrics = ["calibration_error", "nrmse"]

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
    hparams["batches_per_epoch"] = batches_per_epoch

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
            "for the default compile step (Adam + CosineDecay). Either add "
            "'initial_lr' to your search space (e.g. via TrainingSpace) or "
            "provide a custom train_fn that compiles with its own optimizer."
        )
    initial_lr = float(hparams.get("initial_lr", 1e-3))
    decay_steps = batches_per_epoch * epochs
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
    try:
        actual_train_fn(approximator, simulator, dict(hparams), [])
    except Exception as exc:
        raise PipelineError(f"Training step failed: {exc}") from exc

    # --- Step 6: Validate ---
    actual_validate_fn = validate_fn if validate_fn is not None else default_validate_fn
    try:
        result = actual_validate_fn(approximator, validation_data, n_posterior_samples)
    except Exception as exc:
        raise PipelineError(f"Validation step failed: {exc}") from exc

    if not isinstance(result, dict):
        raise PipelineError(
            f"validate_fn must return dict[str, float], got {type(result).__name__}"
        )

    missing_keys = set(objective_metrics) - set(result.keys())
    if missing_keys:
        raise PipelineError(
            f"validate_fn output is missing required metric keys: "
            f"{sorted(missing_keys)}. Got keys: {sorted(result.keys())}"
        )

    for key in objective_metrics:
        val = result[key]
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise PipelineError(
                f"validate_fn returned non-finite value for {key!r}: {val}"
            )

    # --- Step 7: Warn about unused hparams ---
    if build_approximator_fn is not None:
        sampled_keys = set(raw_hparams.keys())
        unused = sampled_keys - hparams.accessed_keys
        if unused:
            logger.warning(
                "Search space sampled keys that were never read by "
                "build_approximator_fn: %s. Consider removing them "
                "from the search space to avoid wasting Optuna budget.",
                sorted(unused),
            )
