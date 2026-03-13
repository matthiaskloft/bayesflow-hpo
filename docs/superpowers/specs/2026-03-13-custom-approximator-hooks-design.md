# Custom Approximator Hooks for `optimize()`

**Date:** 2026-03-13
**Status:** Draft
**Addresses:** TODO item 8 — `optimize()` assumes `BasicWorkflow`

## Problem

`optimize()` hard-codes `build_workflow() -> BasicWorkflow`. Packages with custom
approximators (e.g. `bayesflow-irt`'s `EquivariantIRTApproximator`) cannot use the
high-level API and must reimplement the entire trial lifecycle — budget rejection,
early stopping, checkpoint management, cost scoring, and metric logging.

## Solution

Three user-facing hooks let callers replace the build, train, and validate steps
while reusing the full trial lifecycle. Public wrappers expose the default behavior
so users can read, copy, and swap out one piece at a time. A pre-flight
`check_pipeline()` catches interface mismatches before GPU hours are wasted.

## New `optimize()` Signature

```python
optimize(
    # Required
    simulator: bf.simulators.Simulator,
    adapter: bf.adapters.Adapter,
    search_space: CompositeSearchSpace,

    # Custom approximator hooks (all optional)
    build_approximator_fn: BuildApproximatorFn | None = None,
    train_fn: TrainFn | None = None,
    validate_fn: ValidateFn | None = None,

    # Validation data (always built internally for timing)
    validation_conditions: dict[str, list[Any]] | None = None,
    sims_per_condition: int = 200,
    n_posterior_samples: int = 500,

    # Objectives
    objective_metrics: list[str] = ["calibration_error", "nrmse"],
    objective_mode: str = "pareto",
    cost_metric: str = "inference_time",

    # Training
    epochs: int = 200,
    batches_per_epoch: int = 50,
    early_stopping_patience: int = 5,
    early_stopping_window: int = 7,

    # Budget
    max_param_count: int = 1_000_000,
    max_memory_mb: float | None = None,

    # Study
    n_trials: int = 50,
    max_total_trials: int | None = None,
    study_name: str = "bayesflow_hpo",
    storage: str | None = DEFAULT_STORAGE,
    resume: bool = False,

    # Optional
    directions: list[str] | None = None,
    warm_start_from: Any | None = None,
    warm_start_top_k: int = 25,
    checkpoint_pool: CheckpointPool | None = None,
    show_progress_bar: bool = True,
) -> optuna.Study
```

### Removed parameters (breaking)

| Parameter | Reason |
|-----------|--------|
| `validation_data` | Always built internally to track `sim_time_per_sim` |
| `param_keys` | Inferred from adapter |
| `data_keys` | Inferred from adapter |
| `inference_conditions` | Inferred from adapter |
| `objective_metric` | Replaced by `objective_metrics` list (single metric = list of one) |

### Changed parameters (breaking)

| Parameter | Old | New |
|-----------|-----|-----|
| `train_fn` | `(workflow, params, callbacks) -> None` | `(approximator, simulator, hparams, callbacks) -> None` |
| `objective_metrics` | default `None` | default `["calibration_error", "nrmse"]` |
| `objective_mode` | default `"mean"` | default `"pareto"` |
| `search_space` | optional (hidden default) | required |

### Added parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `build_approximator_fn` | `(hparams: dict) -> Approximator` | Replaces internal `BasicWorkflow` construction |
| `validate_fn` | `(approximator, validation_data, n_samples) -> dict[str, float]` | Replaces `run_validation_pipeline()` |
| `n_posterior_samples` | `int` | Posterior draws for validation (default 500) |

## Type Aliases

```python
BuildApproximatorFn = Callable[[dict[str, Any]], bf.approximators.Approximator]

TrainFn = Callable[
    [bf.approximators.Approximator, bf.simulators.Simulator, dict[str, Any], list],
    None,
]

ValidateFn = Callable[
    [bf.approximators.Approximator, ValidationDataset, int],
    dict[str, float],
]
```

## Public Default Wrappers

Exposed as importable functions so users can see what happens by default, and
replace one piece at a time.

### `build_continuous_approximator`

```python
def build_continuous_approximator(
    hparams: dict[str, Any],
    adapter: bf.adapters.Adapter,
    search_space: CompositeSearchSpace,
) -> bf.approximators.ContinuousApproximator:
    """Build and compile a ContinuousApproximator from search-space hparams.

    This is the default used by ``optimize()`` when ``build_approximator_fn``
    is ``None``.  It:

    1. Constructs inference and summary networks from the search space.
    2. Wraps them in a ``ContinuousApproximator``.
    3. Compiles with Adam + CosineDecay (LR from ``hparams["initial_lr"]``).

    Note: this function has a broader signature than ``BuildApproximatorFn``
    because it needs the adapter and search space.  Inside ``optimize()``
    these are captured internally.  It cannot be passed directly as
    ``build_approximator_fn`` — use a partial or lambda if you want to
    call it explicitly::

        build_approximator_fn=lambda hp: hpo.build_continuous_approximator(
            hp, adapter, search_space
        )
    """
```

### `default_train_fn`

```python
def default_train_fn(
    approximator: bf.approximators.Approximator,
    simulator: bf.simulators.Simulator,
    hparams: dict[str, Any],
    callbacks: list,
) -> None:
    """Train via ``approximator.fit(simulator=..., ...)``.

    This is the default used by ``optimize()`` when ``train_fn`` is ``None``.
    Reads ``epochs``, ``batches_per_epoch``, and ``batch_size`` from
    ``hparams`` (injected by the objective before calling).
    """
```

### `default_validate_fn`

```python
def default_validate_fn(
    approximator: bf.approximators.Approximator,
    validation_data: ValidationDataset,
    n_posterior_samples: int,
) -> dict[str, float]:
    """Run the built-in validation pipeline and return metric dict.

    This is the default used by ``optimize()`` when ``validate_fn`` is ``None``.
    Wraps ``run_validation_pipeline()`` and returns its summary as a flat dict.
    """
```

## Training Config in `hparams`

The objective injects training configuration into the `hparams` dict before
passing it to `train_fn`:

```python
hparams["epochs"] = config.epochs
hparams["batches_per_epoch"] = config.batches_per_epoch
```

This keeps `TrainFn` a clean 4-arg signature. Custom `train_fn` implementations
can read these or ignore them. The injected keys use reserved names that search
spaces must not sample (enforced by `check_pipeline()`).

## Compile Ownership

**The objective always compiles the approximator** (step 4 in the lifecycle).
This applies regardless of whether `build_approximator_fn` is provided.

- **Default path:** `build_continuous_approximator` returns an uncompiled
  approximator. The objective compiles it.
- **Custom path:** `build_approximator_fn` must return an **uncompiled**
  approximator. The objective compiles it with Adam + CosineDecay using
  `hparams["initial_lr"]`.

If a custom approximator needs a non-standard optimizer, the user should
provide a `train_fn` that re-compiles before training. The objective's
compile (step 5) runs first, but calling `compile()` again in `train_fn`
replaces the optimizer — Keras uses the last `compile()` call.

**Required hparam:** `initial_lr` must be present in `hparams` for the
compile step. The default `TrainingSpace` always samples it. If using a
custom search space without `TrainingSpace`, include `initial_lr` in
the search space or provide a `train_fn` that compiles with its own
optimizer (the objective's compile will be overridden). `check_pipeline()`
validates that `initial_lr` is present in the sampled hparams.

## Checkpoint Pool

`CheckpointPool.maybe_save()` currently accepts a `workflow` argument.
This changes to accept an `approximator`:

```python
# Before
self._checkpoint_pool.maybe_save(trial_number=..., objective_value=..., workflow=workflow)

# After
self._checkpoint_pool.maybe_save(trial_number=..., objective_value=..., approximator=approximator)
```

Both `BasicWorkflow` and `Approximator` are `keras.Model` subclasses, so
the underlying `save_weights()` / `load_weights()` calls work identically.
This is an internal API change (not user-facing).

## Validation Data

Validation data is **always** built internally by `optimize()` via
`generate_validation_dataset()`. This ensures `sim_time_per_sim` is tracked
for the inference-time cost metric.

When `validation_conditions` is `None` and no conditions are inferred from
the adapter, a single unconditional batch is generated with
`sims_per_condition` simulations. The no-validation fallback (training loss
only) is removed — every study gets proper validation.

## Search Space and Custom Builders

When `build_approximator_fn` is provided, the user controls which `hparams`
their builder uses. The `search_space` still defines what Optuna samples.

**Responsibility:** The user must ensure their `search_space` only contains
dimensions their `build_approximator_fn` consumes. If the search space
includes a `summary_space` but the custom builder ignores summary hparams,
Optuna wastes budget exploring unused dimensions.

`check_pipeline()` warns about this: it compares the set of sampled hparam
keys against the keys actually read during a dry-run build (tracked via a
dict wrapper), and logs a warning for any unread keys.

## Validation of `validate_fn` Output

At trial time (step 7), the objective validates the returned dict:

- All keys in `objective_metrics` must be present. Missing keys cause the
  trial to return penalty values (not crash), with a warning logged.
- All values must be finite floats. `NaN`/`Inf` values are replaced with
  penalty values and a warning is logged.

`check_pipeline()` also checks this during the dry run and raises
`PipelineError` if keys are missing — catching the problem before any
real trials run.

## Pre-flight Validation: `check_pipeline()`

```python
def check_pipeline(
    simulator: bf.simulators.Simulator,
    adapter: bf.adapters.Adapter,
    search_space: CompositeSearchSpace,
    build_approximator_fn: BuildApproximatorFn | None = None,
    train_fn: TrainFn | None = None,
    validate_fn: ValidateFn | None = None,
    objective_metrics: list[str] = ["calibration_error", "nrmse"],
    sims_per_condition: int = 5,
    n_posterior_samples: int = 2,
    validation_conditions: dict[str, list[Any]] | None = None,
    epochs: int = 1,
    batches_per_epoch: int = 1,
) -> None:
    """Dry-run the full pipeline to catch interface errors early.

    Steps:
    1. Sample dummy hparams from search_space (using a mock Optuna trial).
    2. Call build_approximator_fn (or default) — verify result has ``fit``
       and ``compute_loss`` methods (duck-typed, no isinstance check).
    3. Generate a tiny validation dataset (sims_per_condition=5).
    4. Compile and run one training step (1 epoch, 1 batch) — verify
       simulator/adapter/approximator are compatible.
    5. Call validate_fn (or default) — verify it returns ``dict[str, float]``
       whose keys include all entries in ``objective_metrics``.
    6. Warn about any sampled hparam keys not consumed by the builder.

    Raises
    ------
    PipelineError
        With a clear message identifying which component failed and why.
    """
```

Called automatically at the start of `optimize()`. Users can also call it
manually to debug before launching a long run. Uses `epochs=1,
batches_per_epoch=1` by default for minimal cost.

## Trial Lifecycle (inside `GenericObjective.__call__`)

```
1.  sample hparams from search_space                     (unchanged)
2.  inject training config into hparams                  (NEW)
      hparams["epochs"] = config.epochs
      hparams["batches_per_epoch"] = config.batches_per_epoch
3.  budget pre-check (memory estimate from hparams)      (unchanged)
      Uses heuristic based on hparams (hidden_dim, depth, etc.)
      — no model needed. Skipped when max_memory_mb is None.
4.  BUILD approximator
      if build_approximator_fn:
          approximator = build_approximator_fn(hparams)
      else:
          approximator = build_continuous_approximator(hparams, adapter, search_space)
5.  COMPILE with Adam + CosineDecay                      (moved from build_workflow)
6.  exact param count check                              (unchanged, uses approximator)
7.  TRAIN
      if train_fn:
          train_fn(approximator, simulator, hparams, callbacks)
      else:
          default_train_fn(approximator, simulator, hparams, callbacks)
8.  VALIDATE
      if validate_fn:
          raw = validate_fn(approximator, validation_data, n_posterior_samples)
          metrics = _validate_metric_keys(raw, objective_metrics)
      else:
          metrics = default_validate_fn(approximator, validation_data, n_posterior_samples)
9.  cost scoring                                         (unchanged)
10. checkpoint pool                                      (updated: approximator, not workflow)
11. logging                                              (unchanged)
```

Steps 1-3 and 6, 9-11 are fully reused regardless of which hooks are provided.

## Usage Examples

### Quickstart (default path, educational)

```python
import bayesflow as bf
import bayesflow_hpo as hpo

simulator = bf.simulators.make_simulator([prior_fn, likelihood_fn])
adapter = (
    bf.Adapter()
    .as_set(["x"])
    .rename("theta", "inference_variables")
    .concatenate(["x"], into="summary_variables", axis=-1)
)

# All three hooks shown explicitly for education — passing None
# (or omitting them) gives the same behavior.
study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    search_space=hpo.CompositeSearchSpace(
        inference_space=hpo.FlowMatchingSpace(),
        summary_space=hpo.DeepSetSpace(),
        training_space=hpo.TrainingSpace(),
    ),
    sims_per_condition=100,
    n_trials=10,
    epochs=30,
    batches_per_epoch=30,
    max_param_count=500_000,
)
```

The defaults are:
- `build_approximator_fn=None` → uses `build_continuous_approximator` internally
- `train_fn=None` → uses `default_train_fn` (calls `approximator.fit`)
- `validate_fn=None` → uses `default_validate_fn` (calls `run_validation_pipeline`)

Users can import and read these functions to understand what each step does,
then swap out one at a time.

### Custom approximator (IRT-style)

```python
from bayesflow_irt import EquivariantIRTApproximator, EquivariantIRTSummary

# Only tune inference network + training — IRT builds its own summary
search_space = hpo.CompositeSearchSpace(
    inference_space=hpo.CouplingFlowSpace(depth=(2, 8), hidden_dim=(64, 256)),
    training_space=hpo.TrainingSpace(),
    # no summary_space — build_irt_approximator handles it
)

def build_irt_approximator(hparams):
    return EquivariantIRTApproximator(
        inference_network=bf.networks.CouplingFlow(
            depth=hparams["depth"],
            subnet_kwargs={"hidden_dim": hparams["hidden_dim"]},
        ),
        summary_network=EquivariantIRTSummary(
            embed_dim=hparams.get("embed_dim", 64),
        ),
        adapter=adapter,
    )

def validate_irt(approximator, validation_data, n_samples):
    loss = approximator.evaluate(validation_data)
    return {"calibration_error": loss}

# Optional: catch interface errors before a long run
hpo.check_pipeline(
    simulator=simulator,
    adapter=adapter,
    search_space=search_space,
    build_approximator_fn=build_irt_approximator,
    validate_fn=validate_irt,
    objective_metrics=["calibration_error"],
)

study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    search_space=search_space,
    build_approximator_fn=build_irt_approximator,
    validate_fn=validate_irt,
    validation_conditions={"n_items": [10, 20], "n_persons": [50, 100]},
    sims_per_condition=200,
    objective_metrics=["calibration_error"],
    n_trials=30,
)
```

## Files to Change

| File | Change |
|------|--------|
| `api.py` | New `optimize()` signature, remove old params, add `check_pipeline()`, type aliases |
| `optimization/objective.py` | Add hooks to `ObjectiveConfig`, refactor `__call__` lifecycle, extract `default_train_fn` and `default_validate_fn` as public functions |
| `builders/workflow.py` | Extract optimizer creation into helper; add `build_continuous_approximator()` |
| `builders/__init__.py` | Export `build_continuous_approximator` |
| `optimization/checkpoint_pool.py` | Rename `workflow` param to `approximator` in `maybe_save()` |
| `__init__.py` | Export new public API: wrappers, `check_pipeline`, type aliases |
| `examples/quickstart.ipynb` | Update to new signature |
| `tests/` | Tests for custom hooks, `check_pipeline`, and default path |

## Cost Metric

The `cost_metric` parameter controls the last Optuna objective dimension,
which is always appended after the accuracy metrics from `objective_metrics`.

- `"inference_time"` (default) — ratio of inference wall time to simulation
  wall time, measured during validation. Lower = cheaper model.
- `"param_count"` — normalized parameter count (log-linear scaling from 0
  to 1). Lower = smaller model.

The total number of Optuna objectives is:
- `objective_mode="pareto"`: `len(objective_metrics) + 1` (each metric + cost)
- `objective_mode="mean"`: `2` (mean of metrics + cost)

All directions default to `"minimize"`.

## Non-goals

- Custom optimizer hooks (users compile in `build_approximator_fn` or `train_fn`)
- Custom search space protocol changes (out of scope)
- Resolving TODO item 9 (flat posterior assumption in `run_validation_pipeline`) — `validate_fn` is the escape hatch
