# TODOs

Active issues and improvement opportunities for bayesflow-hpo.

Items from the multi-objective pruning quality audit (2026-03-06) are
archived in the [Resolved](#resolved-archive) section at the bottom.

---

## Robustness

### 1. `_compile_for_compat` silently returns on total failure

**File:** `builders/workflow.py:40-67`

**Issue:** If all three compile signatures fail with `TypeError`, the function
returns without raising.  The approximator is left **uncompiled**, which can
cause cryptic failures during training.

**Fix:** Log a warning when no compile signature succeeds so the caller
(and the user) know the model is uncompiled.

### 2. `loguniform_int` can exceed upper bound after rounding

**File:** `utils.py:43`

**Issue:** `int(np.round(np.exp(log_val)))` can exceed `high` when the
exponential lands just above `high - 0.5`.

**Fix:** Clamp the result: `return int(np.clip(np.round(np.exp(log_val)), low, high))`.

### 3. `normalize_param_count` edge-case inconsistency

**File:** `objectives.py:92-99`

**Issue:** `param_count=0` returns `1.0` (worst), but
`param_count=1, min_count=1, max_count=1` returns `0.0` (best).  The boundary
semantics are not uniformly defined.

**Fix:** Document the intended invariant and add a guard for
`max_count <= min_count` that returns `0.5` (neutral) instead of `0.0`.

### 4. `infer_keys_from_adapter` returns `None` silently on missing transforms

**File:** `api.py:63-65`

**Issue:** When the adapter has no `transforms` attribute, the function returns
all-`None` with no log message.  Users get a `TypeError` later from `optimize()`
without knowing why.

**Fix:** Add a `logger.debug(...)` when `transforms` is `None` so the inference
path is at least visible in debug logs.

---

## Usability

### 5. `optimize()` does not accept explicit `param_keys`/`data_keys`

**File:** `api.py:96-362`

**Issue:** Keys are always inferred from the adapter.  Users with non-standard
adapters (e.g. custom transforms) have no fallback.

**Fix:** Add optional `param_keys` / `data_keys` parameters that override
inference when provided.

### 6. `check_pipeline` uses very different defaults from `optimize()`

**File:** `pipeline.py:124-128`

**Issue:** The dry-run defaults (`sims_per_condition=5`, `batches_per_epoch=1`,
`epochs=1`) are intentionally small for speed, but a config that passes
`check_pipeline` can still fail under `optimize()`'s larger defaults (e.g.
memory-wise).  The asymmetry is not documented.

**Fix:** Add a note to the `check_pipeline` docstring explaining that it uses
intentionally minimal defaults and is not a full fidelity check.

### 7. Missing `py.typed` marker

**File:** `src/bayesflow_hpo/` (package root)

**Issue:** CLAUDE.md declares "Full type hints (mypy-compatible)" but the PEP 561
`py.typed` marker file is missing.  mypy and pyright will not treat the package
as typed.

**Fix:** Create an empty `src/bayesflow_hpo/py.typed` and add it to
`pyproject.toml` package data.

---

## Code Quality

### 8. `optimize()` function is ~270 lines with mixed responsibilities

**File:** `api.py:96-362`

**Issue:** Key inference, validation data generation, pipeline checking,
objective creation, study creation, and optimization are all in one function.

**Fix:** Extract helpers like `_setup_validation_data()` and
`_build_objective()` to improve readability and testability.  No change
to the public API.

### 9. Redundant builder registration loop in `registration.py`

**File:** `registration.py:55-58, 90-92`

**Issue:** The alias loop manually calls `register_inference_builder` for each
alias, duplicating logic that could be a shared helper.

**Fix:** Extract a `_register_with_aliases(registry_fn, name, builder, aliases)`
helper.

### 10. `_TrackingDict` does not track `items()` / `values()` / `__iter__`

**File:** `pipeline.py:51-84`

**Issue:** A builder that uses `for k, v in hparams.items()` will not mark keys
as accessed, so the unused-key warning fires even when keys were consumed.
This is a known trade-off documented in the class (calling `dict()` on the
tracking dict would mark everything), but it can produce false-positive
warnings.

**Fix:** Document the limitation in `check_pipeline`'s docstring so users know
to ignore warnings when their builder iterates the dict.

---

## Type Safety

### 11. `TrainFn` callback list is unparameterized

**File:** `types.py:23`

**Issue:** `list` without a type parameter (`list[Any]` or
`list[keras.callbacks.Callback]`).

**Fix:** Change to `list[Any]` for consistency with the rest of the module.

### 12. `_check_hook_arity` parameter `fn` is typed as `Any`

**File:** `pipeline.py:87`

**Issue:** `fn: Any` defeats static type checking on the helper.

**Fix:** Type as `Callable[..., Any]`.

---

## Documentation

### 13. `builders/adapter.py` deprecation notice lacks version and migration guide

**File:** `builders/adapter.py`

**Issue:** The module docstring says "removed" but gives no version number or
migration path.

**Fix:** Add "Deprecated since v0.2.0" and a one-line pointer to
`bayesflow.Adapter`.

### 14. `utils.py` `rng` parameter doesn't document `None` fallback behavior

**File:** `utils.py:31-32, 66-67`

**Issue:** The docstring for `rng` says "Optional NumPy random generator" but
does not mention that `None` uses the global `np.random` module, which is
non-deterministic across runs.

**Fix:** Add "When ``None``, uses the global ``np.random`` module (non-deterministic)."

### 15. `PipelineError` has a one-line docstring

**File:** `pipeline.py:28-29`

**Issue:** No guidance on common causes or how to debug.

**Fix:** Expand with a brief "Common causes" list (missing adapter transforms,
incompatible hook signatures, etc.).

### 16. CLAUDE.md architecture tree does not mention `py.typed` or `__init__` exports

**File:** `CLAUDE.md:35-79`

**Issue:** The tree is accurate for modules but does not mention the
`__init__.py` re-export strategy or which symbols are intentionally private.

**Fix:** Add a brief "Public API" note to the Key Patterns section.

---

## Testing Gaps

### 17. No test for `warm_start_study`

**File:** `optimization/study.py:169-218`

**Issue:** Warm-start logic (ranking, trial copying, edge cases) has no unit
tests.

### 18. No test for `_training_loss_fallback`

**File:** `optimization/objective.py:281-334`

**Issue:** The validation-failure fallback path is critical but not directly
tested.

### 19. No test for `load_validation_dataset` round-trip

**File:** `validation/data.py:214-244`

**Issue:** `save_validation_dataset` → `load_validation_dataset` round-trip
is not tested.  Serialization bugs would go unnoticed.

### 20. No test for `make_condition_grid` edge cases

**File:** `validation/data.py:149-182`

**Issue:** `logspace` and mixed-mode grids are not tested.

---

## Resolved Archive

<details>
<summary>Issues from the multi-objective pruning quality audit (2026-03-06) — all resolved</summary>

### ~~Broad `except Exception` in `_run_lightweight_validation` (pre-existing)~~ — RESOLVED

**File:** `optimization/validation_callback.py:186-215`

Now logs at `WARNING` level with `exc_info=True`, re-raises `TrialPruned`, and
tracks consecutive failures with a warning after 3.

### ~~Final validation not wrapped in try-except (pre-existing)~~ — RESOLVED

**File:** `optimization/objective.py` (step 8)

Falls back to training-loss-based objective values instead of penalty values.

### ~~`get_param_count` returns -1 on error~~ — RESOLVED

**File:** `objectives.py:43-61`

Now raises `ValueError` / `TypeError` instead of returning `-1`.

### ~~`api.py` delete-study catches only `KeyError`~~ — RESOLVED

**File:** `api.py:320-329`

Now catches generic `Exception` with `exc_info=True`.

### ~~`optimize_until` warning message doesn't mention pruning~~ — RESOLVED

**File:** `optimization/study.py:370-421`

Warning now includes failure breakdown with pruned count and guidance.

### ~~`OptunaReportCallback` stores per-epoch user attrs on every trial~~ — RESOLVED

**File:** `optimization/callbacks.py`

`report_frequency` is now configurable from `optimize()`.

### ~~`MedianPruner` docstring in `create_study` is misleading~~ — RESOLVED

**File:** `optimization/study.py`

Docstring now says "Single-objective only."

### ~~`optimize()` assumes `BasicWorkflow`~~ — RESOLVED

Resolved by custom approximator hooks (2026-03-14).

### ~~`run_validation_pipeline` assumes flat posterior shape~~ — RESOLVED

Resolved by the `validate_fn` hook.

</details>
