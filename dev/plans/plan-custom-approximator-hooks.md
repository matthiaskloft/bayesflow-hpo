# Implementation Plan: Custom Approximator Hooks

**Spec:** `docs/superpowers/specs/2026-03-13-custom-approximator-hooks-design.md`
**Status:** DONE (all 7 phases implemented, 2026-03-14)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Type aliases & public default wrappers | DONE | `types.py`, `build_continuous_approximator()`, `default_train_fn()`, `default_validate_fn()` |
| Phase 2: `check_pipeline()` pre-flight validation | DONE | `pipeline.py` with `PipelineError`, `_MockTrial`, `_TrackingDict`, arity checks |
| Phase 3: Refactor `GenericObjective.__call__()` lifecycle | DONE | 11-step lifecycle, `_validate_metric_keys()`, hook integration |
| Phase 4: Update `CheckpointPool` | DONE | `workflow` → `approximator` rename |
| Phase 5: Update `optimize()` signature & orchestration | DONE | `search_space` required, old params removed, hooks wired, `check_pipeline()` called |
| Phase 6: Update tests | DONE | `test_pipeline.py`, updated `test_api.py`, `test_objective.py` |
| Phase 7: Cleanup & final exports | DONE | `build_workflow()`/`WorkflowBuildConfig` deleted, all new symbols exported |

## Overview

Refactor `optimize()` and the trial lifecycle to support three user-facing hooks
(`build_approximator_fn`, `train_fn`, `validate_fn`), enabling packages with
custom approximators (e.g. bayesflow-irt) to use the high-level HPO API without
reimplementing the trial lifecycle. Add `check_pipeline()` pre-flight validation
and extract default behaviors as public importable functions.

---

## Phase 1: Type Aliases & Public Default Wrappers

**Goal:** Define the hook contracts and extract default behavior into standalone,
importable functions before rewiring anything.

### Step 1.1: Add type aliases module

**File:** `src/bayesflow_hpo/types.py` (new)

Create the three type aliases:

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

These are referenced by `optimize()`, `ObjectiveConfig`, and `check_pipeline()`.

### Step 1.2: Extract `build_continuous_approximator()`

**File:** `src/bayesflow_hpo/builders/workflow.py`

Extract from the current `build_workflow()` the logic that:
1. Builds inference network via `search_space.inference_space.build(params)`
2. Builds summary network via `search_space.summary_space.build(params)` (if present)
3. Creates a `ContinuousApproximator` (not `BasicWorkflow`) with those networks and the adapter

The function has a broader signature than `BuildApproximatorFn` because it needs
`adapter` and `search_space`:

```python
def build_continuous_approximator(
    hparams: dict[str, Any],
    adapter: bf.adapters.Adapter,
    search_space: CompositeSearchSpace,
) -> bf.approximators.ContinuousApproximator:
```

**Key detail:** Returns an **uncompiled** approximator. The objective handles
compilation (step 5 in the lifecycle).

**Breaking change:** Delete `build_workflow()` and `WorkflowBuildConfig` entirely.
The objective will call `build_continuous_approximator()` directly in Phase 3.
Move the CosineDecay optimizer creation into a small helper
`_make_cosine_decay_optimizer(initial_lr, decay_steps)` in the same file, since
the objective needs it for the compile step.

### Step 1.3: Extract `default_train_fn()`

**File:** `src/bayesflow_hpo/optimization/objective.py`

Rename the existing `_default_train_fn()` (currently a module-level helper) to a
public function. Change the signature from
`(workflow, params, callbacks)` to `(approximator, simulator, hparams, callbacks)`:

```python
def default_train_fn(
    approximator: bf.approximators.Approximator,
    simulator: bf.simulators.Simulator,
    hparams: dict[str, Any],
    callbacks: list,
) -> None:
```

Implementation: calls `approximator.fit(simulator=simulator, epochs=hparams["epochs"],
batch_size=hparams["batch_size"], batches_per_epoch=hparams["batches_per_epoch"],
callbacks=callbacks)`.

### Step 1.4: Extract `default_validate_fn()`

**File:** `src/bayesflow_hpo/optimization/objective.py`

```python
def default_validate_fn(
    approximator: bf.approximators.Approximator,
    validation_data: ValidationDataset,
    n_posterior_samples: int,
) -> dict[str, float]:
```

Implementation: wraps `run_validation_pipeline()` and returns `result.summary`.

### Step 1.5: Export new public symbols

**File:** `src/bayesflow_hpo/builders/__init__.py`

Add `build_continuous_approximator` to exports.

**File:** `src/bayesflow_hpo/__init__.py`

Add exports:
- `BuildApproximatorFn`, `TrainFn`, `ValidateFn` from `types.py`
- `build_continuous_approximator` from `builders`
- `default_train_fn`, `default_validate_fn` from `optimization.objective`
- `check_pipeline` (added in Phase 2)

---

## Phase 2: `check_pipeline()` Pre-flight Validation

**Goal:** Implement `check_pipeline()` so users can catch interface errors before
launching expensive studies.

### Step 2.1: Implement `check_pipeline()`

**File:** `src/bayesflow_hpo/pipeline.py` (new)

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
```

### Step 2.2: Define `PipelineError`

**File:** `src/bayesflow_hpo/pipeline.py`

Custom exception with a clear message identifying which component failed and why.

### Step 2.3: Implementation steps within `check_pipeline()`

1. **Sample dummy hparams** from `search_space` using a mock Optuna trial
   (reuse or adapt `FakeTrial` from test conftest into a lightweight
   `MockTrial` class within this module).

2. **Build approximator**: Call `build_approximator_fn(hparams)` or
   `build_continuous_approximator(hparams, adapter, search_space)`. Verify
   result has `fit` and `compute_loss` attributes (duck-typed). Wrap in
   try/except → `PipelineError("build failed: ...")`.

3. **Generate tiny validation dataset**: `generate_validation_dataset()` with
   `sims_per_condition=5`.

4. **Inject training config**: `hparams["epochs"] = epochs`,
   `hparams["batches_per_epoch"] = batches_per_epoch`.

5. **Compile + train one step**: Create optimizer with CosineDecay from
   `hparams["initial_lr"]`. Compile approximator. Call `train_fn` or
   `default_train_fn` with 1 epoch, 1 batch.

6. **Validate**: Call `validate_fn` or `default_validate_fn`. Check returned
   dict contains all `objective_metrics` keys. Check all values are finite
   floats. Raise `PipelineError` on missing keys.

7. **Warn about unused hparams**: Track which keys the builder actually read
   (via a dict wrapper that records `__getitem__` calls) and log warnings for
   unread keys.

### Step 2.4: Export `check_pipeline` and `PipelineError`

Add to `__init__.py`.

---

## Phase 3: Refactor `GenericObjective.__call__()` Lifecycle

**Goal:** Rewire the trial lifecycle to use the three hooks, moving from
`BasicWorkflow` to raw `Approximator`.

### Step 3.1: Update `ObjectiveConfig`

**File:** `src/bayesflow_hpo/optimization/objective.py`

Changes to the dataclass:
- **Remove:** `param_keys`, `data_keys`, `inference_conditions`, `objective_metric`,
  `metrics` (the old single-metric fields)
- **Change:** `train_fn` signature from `(BasicWorkflow, dict, list)` to new `TrainFn`
- **Add:** `build_approximator_fn: BuildApproximatorFn | None = None`
- **Add:** `validate_fn: ValidateFn | None = None`
- **Change:** `objective_metrics` default from `None` to `["calibration_error", "nrmse"]`
- **Change:** `objective_mode` default from `"mean"` to `"pareto"`

Keep: `simulator`, `adapter`, `search_space`, `validation_data`, `epochs`,
`batches_per_epoch`, `early_stopping_patience`, `early_stopping_window`,
`max_param_count`, `max_memory_mb`, `n_posterior_samples`, `cost_metric`,
`checkpoint_pool`, and the intermediate validation / pruning fields.

### Step 3.2: Refactor `__call__()` lifecycle

Rewrite the trial lifecycle following the spec's 11-step sequence:

```
1.  sample hparams from search_space
2.  inject training config into hparams
      hparams["epochs"] = config.epochs
      hparams["batches_per_epoch"] = config.batches_per_epoch
3.  budget pre-check (memory estimate from hparams)
4.  BUILD approximator
      if build_approximator_fn:
          approximator = build_approximator_fn(hparams)
      else:
          approximator = build_continuous_approximator(hparams, adapter, search_space)
5.  COMPILE with Adam + CosineDecay (using hparams["initial_lr"])
6.  exact param count check
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
9.  cost scoring
10. checkpoint pool (approximator, not workflow)
11. logging
```

**Key changes from current implementation:**
- Step 2 is new (inject training config into hparams dict)
- Steps 4-5 replace the old build_inference → build_summary → build_workflow flow
- Compile is pulled out of `build_workflow()` into the objective directly
- Step 7 passes `approximator` instead of `workflow`
- Step 8 adds metric key validation for custom `validate_fn` output
- Step 10 passes `approximator` instead of `workflow`

### Step 3.3: Add `_validate_metric_keys()` helper

**File:** `src/bayesflow_hpo/optimization/objective.py`

```python
def _validate_metric_keys(
    raw: dict[str, float],
    objective_metrics: list[str],
) -> dict[str, float]:
```

- Check all `objective_metrics` keys are present; replace missing with penalty value + warning
- Check all values are finite floats; replace NaN/Inf with penalty value + warning
- Return cleaned dict

### Step 3.4: Remove `_default_train_fn()` private helper

It has been replaced by the public `default_train_fn()` from Step 1.3.

### Step 3.5: Update `n_objectives` property

Currently handles both `objective_metric` (single) and `objective_metrics` (multi).
Simplify to only use `objective_metrics` list:
- `objective_mode="pareto"`: `len(objective_metrics) + 1`
- `objective_mode="mean"`: `2`

### Step 3.6: Update `_penalty()` method

Should return the correct number of penalty values based on the simplified
`n_objectives`.

---

## Phase 4: Update `CheckpointPool`

**Goal:** Rename `workflow` parameter to `approximator`.

### Step 4.1: Rename parameter in `maybe_save()`

**File:** `src/bayesflow_hpo/optimization/checkpoint_pool.py`

```python
def maybe_save(
    self,
    trial_number: int,
    objective_value: float,
    approximator: Any,  # was: workflow
) -> bool:
```

The internal logic already extracts `approximator` from workflow, so simplify:
just call `approximator.save_weights()` directly. Remove the
`getattr(workflow, 'approximator', workflow)` indirection.

---

## Phase 5: Update `optimize()` Signature & Orchestration

**Goal:** Implement the new `optimize()` signature from the spec.

### Step 5.1: Update `optimize()` signature

**File:** `src/bayesflow_hpo/api.py`

**Remove parameters:**
- `validation_data` (always built internally)
- `param_keys` (inferred from adapter)
- `data_keys` (inferred from adapter)
- `inference_conditions` (inferred from adapter)
- `objective_metric` (replaced by `objective_metrics` list)
- `metrics` (old alias)

**Add parameters:**
- `build_approximator_fn: BuildApproximatorFn | None = None`
- `validate_fn: ValidateFn | None = None`
- `n_posterior_samples: int = 500`

**Change parameters:**
- `train_fn` signature: old `(BasicWorkflow, dict, list)` → new `TrainFn`
- `objective_metrics` default: `None` → `["calibration_error", "nrmse"]`
- `objective_mode` default: `"mean"` → `"pareto"`
- `search_space`: optional → **required** (no default)

### Step 5.2: Simplify key inference

Since `param_keys`/`data_keys` are no longer user-facing params, call
`infer_keys_from_adapter()` internally and use the results directly. Remove all
the cross-validation logic between explicit keys and validation_data keys.

Keep `infer_keys_from_adapter()` as a public function (it's still useful for
users).

### Step 5.3: Always build validation data internally

Replace the conditional validation_data logic with:
```python
validation_data = generate_validation_dataset(
    simulator=simulator,
    param_keys=inferred_param_keys,
    data_keys=inferred_data_keys,
    condition_grid=validation_conditions,
    sims_per_condition=sims_per_condition,
)
```

This ensures `sim_time_per_sim` is always tracked.

### Step 5.4: Call `check_pipeline()` automatically

At the start of `optimize()`, before creating the study, call:
```python
check_pipeline(
    simulator=simulator,
    adapter=adapter,
    search_space=search_space,
    build_approximator_fn=build_approximator_fn,
    train_fn=train_fn,
    validate_fn=validate_fn,
    objective_metrics=objective_metrics,
    validation_conditions=validation_conditions,
)
```

### Step 5.5: Wire new hooks into `ObjectiveConfig`

Pass `build_approximator_fn`, `train_fn`, and `validate_fn` through to
`ObjectiveConfig`. Remove the old removed fields.

---

## Phase 6: Update Tests

**Goal:** Update existing tests and add new ones for the hook system and
`check_pipeline()`.

### Step 6.1: Update `tests/conftest.py`

- Add a `canonical_search_space()` fixture that returns a minimal
  `CompositeSearchSpace` (so tests that call `optimize()` can pass it as
  required).

### Step 6.2: Update `tests/test_api.py`

- Update all `_patched_optimize()` calls to pass `search_space` (now required).
- Remove tests for removed parameters (`validation_data`, `param_keys`,
  `data_keys`, `inference_conditions`, `objective_metric`).
- Add tests for:
  - `build_approximator_fn` is forwarded to `ObjectiveConfig`
  - `validate_fn` is forwarded to `ObjectiveConfig`
  - `check_pipeline()` is called at start of `optimize()`
  - `n_posterior_samples` forwarding

### Step 6.3: Update `tests/test_optimization/test_objective.py`

- Update `ObjectiveConfig` construction to use new fields.
- Add tests for:
  - Custom `build_approximator_fn` is called instead of default
  - Custom `train_fn` receives `(approximator, simulator, hparams, callbacks)`
  - Custom `validate_fn` output is validated via `_validate_metric_keys()`
  - Missing metric keys in `validate_fn` output → penalty + warning
  - NaN/Inf values in `validate_fn` output → penalty + warning
  - Training config injection (`epochs`, `batches_per_epoch` in hparams)
  - Compile happens in objective (step 5), not in builder

### Step 6.4: Add `tests/test_pipeline.py` (new)

Tests for `check_pipeline()`:
- Successful dry run with all defaults
- `PipelineError` on builder that returns object without `fit`
- `PipelineError` on `validate_fn` that returns wrong keys
- Warning on unused hparam keys
- Works with custom `build_approximator_fn`
- Works with custom `validate_fn`

### Step 6.5: Update `tests/test_optimization/test_checkpoint_pool.py`

- Rename `workflow` param to `approximator` in all test calls

---

## Phase 7: Cleanup & Final Exports

### Step 7.1: Delete `build_workflow()` and `WorkflowBuildConfig`

**File:** `src/bayesflow_hpo/builders/workflow.py`

Remove `build_workflow()`, `WorkflowBuildConfig`, and the `_compile_candidate_for_compat()`
helper entirely. The file now contains only `build_continuous_approximator()` and
`_make_cosine_decay_optimizer()`.

### Step 7.2: Remove dead code from objective

Remove `_default_train_fn()` (replaced by public `default_train_fn()`).
Remove old single-metric extraction path (fully superseded by `objective_metrics` list).

### Step 7.3: Clean up `__init__.py` exports

- Add: `check_pipeline`, `PipelineError`, `BuildApproximatorFn`, `TrainFn`,
  `ValidateFn`, `build_continuous_approximator`, `default_train_fn`,
  `default_validate_fn`
- Remove: `build_workflow`, `WorkflowBuildConfig`, `build_inference_network`,
  `build_summary_network` (absorbed into `build_continuous_approximator`)

### Step 7.4: Run linter and tests

```bash
ruff check src/ tests/
pytest tests/ -v
```

Fix any issues.

---

## Dependency Graph

```
Phase 1 (types + wrappers)
   │
   ├──► Phase 2 (check_pipeline)
   │       │
   │       └──► Phase 5 (optimize() signature) ──► Phase 7 (cleanup)
   │
   └──► Phase 3 (objective refactor) ──► Phase 5
           │
           └──► Phase 4 (checkpoint_pool) ──► Phase 5
                                                  │
                                                  └──► Phase 6 (tests) ──► Phase 7
```

Phases 2, 3, and 4 can proceed in parallel after Phase 1.
Phase 5 depends on 2, 3, and 4.
Phase 6 can start alongside Phase 5 but needs Phase 5 complete for full test runs.
Phase 7 is final cleanup after everything passes.

---

## Breaking Changes Summary

| Change | Migration |
|--------|-----------|
| `build_workflow()` / `WorkflowBuildConfig` deleted | Use `build_continuous_approximator()` |
| `build_inference_network()` / `build_summary_network()` deleted | Absorbed into `build_continuous_approximator()` |
| `search_space` now required | Pass explicitly (no more hidden default) |
| `validation_data` param removed | Use `validation_conditions` + `sims_per_condition` |
| `param_keys`/`data_keys` params removed | Auto-inferred from adapter |
| `inference_conditions` param removed | Auto-inferred from adapter |
| `objective_metric` param removed | Use `objective_metrics=["calibration_error"]` |
| `objective_mode` default `"mean"` → `"pareto"` | Set `objective_mode="mean"` explicitly for old behavior |
| `objective_metrics` default `None` → `["calibration_error", "nrmse"]` | Already supported, just new default |
| `train_fn` signature changed | Update to `(approximator, simulator, hparams, callbacks)` |
| `CheckpointPool.maybe_save(workflow=...)` → `maybe_save(approximator=...)` | Internal API, update call sites |

---

## Risk Assessment

- **Low risk:** Type aliases, exports, `check_pipeline()` — additive, no existing behavior changes.
- **Medium risk:** `ObjectiveConfig` field changes — many tests construct this directly. Need careful test updates.
- **Medium risk:** `optimize()` signature — breaking changes to the public API.
- **Low risk:** `CheckpointPool` rename — internal API, only called from `GenericObjective.__call__()`.
- **Medium risk:** Compile ownership move — currently in `build_workflow()`, moves to objective. Need to ensure CosineDecay setup is identical.
- **Low risk:** Deleting `build_workflow()` / `build_inference_network()` / `build_summary_network()` — clean break, no shims.

---

## Estimated File Changes

| File | Type | Lines (est.) |
|------|------|-------------|
| `src/bayesflow_hpo/types.py` | New | ~25 |
| `src/bayesflow_hpo/pipeline.py` | New | ~150 |
| `src/bayesflow_hpo/builders/workflow.py` | Rewrite | ~80 (delete old, write new) |
| `src/bayesflow_hpo/builders/inference.py` | Delete | — |
| `src/bayesflow_hpo/builders/summary.py` | Delete | — |
| `src/bayesflow_hpo/optimization/objective.py` | Modify | ~200 changed |
| `src/bayesflow_hpo/optimization/checkpoint_pool.py` | Modify | ~15 changed |
| `src/bayesflow_hpo/api.py` | Modify | ~150 changed |
| `src/bayesflow_hpo/builders/__init__.py` | Modify | ~10 changed |
| `src/bayesflow_hpo/__init__.py` | Modify | ~20 changed |
| `tests/test_api.py` | Modify | ~100 changed |
| `tests/test_optimization/test_objective.py` | Modify | ~100 changed |
| `tests/test_optimization/test_checkpoint_pool.py` | Modify | ~15 changed |
| `tests/test_pipeline.py` | New | ~120 |
| `tests/conftest.py` | Modify | ~15 changed |
