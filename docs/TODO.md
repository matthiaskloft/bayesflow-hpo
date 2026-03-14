# TODOs

Active issues and improvement opportunities for bayesflow-hpo.

Items from the multi-objective pruning quality audit (2026-03-06) are
archived in the [Resolved](#resolved-archive) section at the bottom.

---

## Robustness

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

---

## Validation & Results

### 18. `inference.py` silently skips missing data keys

**File:** `validation/inference.py:32`

**Issue:** `{k: sim_data[k] for k in data_keys if k in sim_data}` silently
drops keys not present in the simulation batch.  When a user passes wrong
`data_keys`, `approximator.sample()` fails with a cryptic BayesFlow error
instead of a clear "missing key" message.

**Fix:** Validate all `data_keys` exist in `sim_data` before calling `sample()`.

### 20. `sbc_tests.py` returns NaN silently when scikit-learn is missing

**File:** `validation/sbc_tests.py:66-70`

**Issue:** C2ST returns `{"sbc_c2st_accuracy": NaN}` with no warning when
`sklearn` is not installed.  Users may not realize the metric was skipped.

**Fix:** Add a `logger.warning(...)` when sklearn is unavailable.

### 21. `sbc_tests.py` uses deprecated `np.random.RandomState`

**File:** `validation/sbc_tests.py:76`

**Issue:** Uses `np.random.RandomState(random_state)` while the rest of the
codebase uses `np.random.default_rng()`.

**Fix:** Replace with `np.random.default_rng(random_state)` for consistency.

### 22. `get_pareto_trials` / `summarize_study` no bounds check on `select_by`

**File:** `results/extraction.py:229, 251`

**Issue:** `select_by` is a user-facing int index into `trial.values` with no
bounds checking.  Out-of-range values cause an `IndexError` with no context.

**Fix:** Validate `0 <= select_by < len(study.directions)` at function entry.

---

## Search Spaces

### 28. `IntDimension` allows `log=True` + `step` simultaneously

**File:** `search_spaces/base.py:49`

**Issue:** Optuna's `trial.suggest_int()` raises `ValueError` when both
`log=True` and `step` (other than 1) are set.  No current space triggers this,
but the data model permits it, so a user customizing dimensions could hit a
runtime error.

**Fix:** Add validation in `BaseSearchSpace.sample()` or `IntDimension.__post_init__`.

### 29. `FusionTransformerSpace` missing `mlp_width` and `bidirectional` dimensions

**File:** `search_spaces/summary/fusion_transformer.py`

**Issue:** `SetTransformerSpace` and `TimeSeriesTransformerSpace` expose
`mlp_width` (optional) and `TimeSeriesNetworkSpace` exposes `bidirectional`,
but `FusionTransformerSpace` has neither.  Inconsistent across transformer-based
summary spaces.

**Fix:** Add `ft_mlp_width` and `ft_bidirectional` as optional dimensions.

---

## Testing Gaps

### 23. No test for `warm_start_study`

**File:** `optimization/study.py:169-218`

**Issue:** Warm-start logic (ranking, trial copying, edge cases) has no unit
tests.

### 24. No test for `_training_loss_fallback`

**File:** `optimization/objective.py:281-334`

**Issue:** The validation-failure fallback path is critical but not directly
tested.

### 25. No test for `load_validation_dataset` round-trip

**File:** `validation/data.py:214-244`

**Issue:** `save_validation_dataset` → `load_validation_dataset` round-trip
is not tested.  Serialization bugs would go unnoticed.

### 26. No test for `make_condition_grid` edge cases

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

### ~~ConsistencyModel `build()` casts `s0`, `s1`, `max_time` to `float` instead of `int`~~ — RESOLVED

**File:** `search_spaces/inference/consistency.py:123-130`

Changed `float(...)` to `int(...)` for `max_time`, `s0`, and `s1`,
matching their `IntDimension` declarations and BayesFlow's expected types.

</details>

<details>
<summary>Issues fixed in the package review PR (2026-03-14)</summary>

### ~~1. `_compile_for_compat` silently returns on total failure~~ — RESOLVED

**File:** `builders/workflow.py:40-67`

Now logs a warning when no compile signature succeeds.

### ~~2. `loguniform_int` can exceed upper bound after rounding~~ — RESOLVED

**File:** `utils.py:43`

Clamped result with `np.clip()`.  Also added `alpha > 0` validation.

### ~~6. `check_pipeline` uses very different defaults from `optimize()`~~ — RESOLVED

**File:** `pipeline.py:124-128`

Docstring now explains minimal defaults are intentional.

### ~~7. Missing `py.typed` marker~~ — RESOLVED

Created `src/bayesflow_hpo/py.typed` and added to `pyproject.toml`.

### ~~11. `TrainFn` callback list is unparameterized~~ — RESOLVED

**File:** `types.py:23`

Changed to `list[Any]`.

### ~~12. `_check_hook_arity` parameter `fn` is typed as `Any`~~ — RESOLVED

**File:** `pipeline.py:87`

Changed to `Callable[..., Any]`.

### ~~13. `builders/adapter.py` deprecation notice lacks version~~ — RESOLVED

Added "Deprecated since v0.2.0" with migration pointer.

### ~~14. `utils.py` `rng` parameter doesn't document `None` fallback~~ — RESOLVED

Docstring now describes `None` → global `np.random` behavior.

### ~~15. `PipelineError` has a one-line docstring~~ — RESOLVED

Expanded with common causes and debugging guidance.

### ~~16. CLAUDE.md architecture tree does not mention public API~~ — RESOLVED

Added "Public API" note to Key Patterns section.

### ~~17. `validation/pipeline.py` uses `time.time()`~~ — RESOLVED

Replaced with `time.perf_counter()`.

### ~~19. `make_coverage_metric` float-to-int truncation~~ — RESOLVED

Changed `int(level * 100)` to `round(level * 100)`.

</details>
