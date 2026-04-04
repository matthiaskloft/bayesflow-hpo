# Plan: Metric Constraints & Memory Auto-Detection (Package H)

**Created**: 2026-04-04
**Author**: Claude

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Spec | DONE | 2026-04-04 | |
| Plan | DONE | 2026-04-04 | Reviewed in 2 iterations |
| Phase 1: Metric constraints dataclass + hard rejection | IMPLEMENTED | 2026-04-04 | code + tests + docs |
| Phase 2: Soft constraints (feasibility-guided sampling) | IMPLEMENTED | 2026-04-04 | code + tests + docs |
| Phase 3: GPU memory auto-detection | IMPLEMENTED | 2026-04-04 | code + tests + docs |
| Ship | BLOCKED | 2026-04-04 | `gh auth` token invalid (HTTP 401); branch pushed |

## Spec

### Summary

**Motivation**: Users running HPO studies currently have no way to
automatically reject or deprioritize trials that meet structural budgets
(param count, memory) but produce poor metric quality. A trial with good
architecture but terrible calibration still occupies space on the Pareto
front and wastes sampler budget. Separately, setting `max_memory_mb`
requires users to manually look up their GPU's VRAM — a friction point
that leads most users to skip the memory budget entirely.

**Outcome**: Users can specify per-metric quality thresholds that (a)
guide the sampler away from poor-quality regions (soft constraints) and
(b) reject clearly bad trials after validation (hard constraints). Users
can also pass `max_memory_mb="auto"` to automatically detect available
GPU memory with a configurable safety margin.

### Requirements

- **R1**: Hard metric constraints: after validation, trials violating
  user-specified metric bounds are rejected (set `rejected_reason`,
  return penalty). Rejected trials do NOT count toward `n_trials`.
- **R2**: Soft metric constraints: trials violating user-specified metric
  thresholds are marked infeasible via Optuna's `constraints_func`.
  The sampler learns to avoid those regions but trials are still recorded.
- **R3**: Both layers compose independently — users can specify only hard,
  only soft, or both.
- **R4**: Constraints apply to any metric in `ValidationResult.summary`
  (not limited to `objective_metrics`), enabling thresholds on diagnostic
  metrics like SBC uniformity.
- **R5**: `max_memory_mb="auto"` queries available VRAM via PyTorch CUDA
  APIs, subtracts a safety margin (default 20%), and resolves to a
  concrete float before the study begins.
- **R6**: When CUDA is unavailable, `"auto"` logs a warning and disables
  the memory budget (equivalent to `None`).

### Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Constraint specification | (A) Dict `{metric: threshold}` with direction inferred from study; (B) List of `(metric, bound, "above"/"below")` tuples; (C) `MetricConstraint` dataclass | **(B) Tuples** | Explicit direction avoids ambiguity for non-objective metrics where no study direction exists. Matches `select_best_trial()` priority tuple convention (3-tuple with direction). Lightweight — no new class needed. |
| Direction semantics | (A) `"above"/"below"` = the bad side (reject when value is above/below); (B) `"above"/"below"` = the desired side | **(A) Bad side** | `("calibration_error", 0.5, "above")` reads as "reject if above 0.5". Matches the constraint violation framing. Docstring must clarify: `"above"` = reject when value exceeds threshold, `"below"` = reject when value falls below threshold. |
| Hard rejection counting | (A) Count toward `n_trials`; (B) Don't count (like budget rejection) | **(B) Don't count** | Per the TODO spec: "Rejected-by-metric trials should not count toward `n_trained`". Consistent with budget rejection semantics — the trial didn't produce a usable result. |
| Hard rejection vs `max_total_trials` | (A) Treat like budget rejection (exclude from `_count_non_rejected`); (B) Count toward `_count_non_rejected` (since training ran) | **(B) Count** | Metric-rejected trials cost a full training run (Steps 1–8). Excluding them from `_count_non_rejected` would cause `optimize_until` runaway loops under tight thresholds. Fix: update `_count_non_rejected` to use a whitelist of pre-training rejection reasons (`{"memory_budget", "param_budget", "build_failed", "compile_failed", "param_probe_failed"}`) rather than checking for any `rejected_reason`. |
| Soft constraint wiring | (A) Extend `_budget_constraints_func`; (B) Compose a new function alongside it; (C) Replace with a combined factory | **(C) Factory function** | A single `_make_constraints_func()` that always checks `rejected_reason` (budget) AND optionally checks metric thresholds. Keeps one constraints_func per sampler (Optuna allows only one). The factory receives the soft thresholds at study creation time and returns a closure. |
| Soft constraints + user-provided sampler | (A) Monkey-patch constraints_func; (B) Warn and skip; (C) Document limitation | **(B) Warn and skip** | When `sampler` is a pre-built instance (not a string preset), `_resolve_sampler()` is never called. Log a warning when `metric_constraints_soft` is non-None but the sampler is user-provided. This is also a pre-existing gap for budget constraints. |
| `max_memory_mb` type | (A) `float \| None \| Literal["auto"]`; (B) Separate `auto_memory` bool | **(A) Union type** | Cleaner API — one parameter, three states: disabled (`None`), manual (`float`), auto (`"auto"`). |
| Memory resolution location | (A) Inside `optimize()` before `_build_objective()`; (B) Inside `ObjectiveConfig.__post_init__` | **(A) In `optimize()`** | GPU detection is a side effect (logs, may fail). Keep the dataclass pure. Resolution converts `"auto"` → `float` before the config is constructed. |
| Safety margin API | (A) Hardcoded 20%; (B) Separate parameter `memory_safety_margin` | **(B) Separate parameter** | Low cost to expose, high value for users with known workloads. Default 0.2 (20%). |
| VRAM measurement | (A) Use `total` VRAM; (B) Use `free` (currently available) VRAM | **(B) Use `free`** | `free` from `torch.cuda.mem_get_info()` accounts for memory already consumed by OS, display driver, and other processes. Using `total` would overestimate available memory. The safety margin provides additional headroom on top of `free`. |
| `detect_gpu_memory_mb` visibility | (A) Public export; (B) Private / submodule-only | **(B) Private** | Most users will never call it directly — they use `"auto"`. Keep it accessible via `bayesflow_hpo.optimization.constraints._detect_gpu_memory_mb` for advanced use but exclude from `__init__.py` exports. Consistent with `BaseSearchSpace`, `Dimension`, etc. |
| `metric_constraints_soft` in ObjectiveConfig | (A) Store for introspection; (B) Don't store — sampler-level only | **(B) Don't store** | Soft constraints are sampler-level, not objective-level. No other `ObjectiveConfig` field is storage-only. Keeping it out avoids dual-threading through both `_build_objective()` and `_create_and_run_study()`. Thread only through `optimize()` → `_create_and_run_study()` → `create_study()`. |

### Scope

#### In Scope

- `MetricConstraintSpec` type alias for constraint tuples
- Hard metric constraint check in `GenericObjective.__call__()` (post-Step 8)
- Soft metric constraint check via composed `constraints_func`
- `metric_constraints_hard` parameter on `optimize()` and `ObjectiveConfig`
- `metric_constraints_soft` parameter on `optimize()` (threaded to
  `create_study()`, NOT stored in `ObjectiveConfig`)
- `_detect_gpu_memory_mb()` utility function (private, submodule access)
- `max_memory_mb="auto"` support with `memory_safety_margin` parameter
- Unit tests for all new code paths
- Updated docstrings and `docs/` files

#### Out of Scope

- Per-metric constraint directions inferred from study directions
  (users must specify direction explicitly in the tuple)
- Constraint-aware intermediate pruning (constraints only apply at final
  validation, not during mid-training lightweight validation)
- CPU RAM estimation fallback for `"auto"` mode (just disables budget)
- Multi-GPU memory aggregation (uses device 0 only)
- Fixing pre-existing gap: user-provided sampler instances bypass
  budget constraints (Package H adds a warning for soft constraints
  but does not fix the budget constraint gap — that is out of scope)

### Architecture Overview

```
optimize(metric_constraints_hard=..., metric_constraints_soft=..., max_memory_mb="auto")
    │
    ├── _resolve_memory_budget("auto", margin=0.2)
    │       → _detect_gpu_memory_mb(safety_margin=0.2) → float | None
    │
    ├── _build_objective(metric_constraints_hard=..., max_memory_mb=<resolved float>)
    │       → ObjectiveConfig(metric_constraints_hard=...)
    │       → GenericObjective.__call__():
    │           Step 8: VALIDATE → metrics_summary
    │           Step 8a: Store metrics in trial.user_attrs (existing)
    │           Step 8b (NEW): _check_hard_constraints(metrics_summary)
    │               if violated → set rejected_reason="metric_constraint"
    │                           → return self._penalty()
    │           Step 9: Cost scoring (existing)
    │
    └── _create_and_run_study(metric_constraints_soft=...)
            → create_study(metric_constraints_soft=...)
                → _make_constraints_func(soft_thresholds=...)
                    → _resolve_sampler(soft_thresholds=...)
                        returns closure checking:
                          1. rejected_reason in user_attrs → [1.0, ...]
                          2. each soft threshold vs user_attr metric → [violation_i, ...]
```

The soft constraints function is called by Optuna **after each successful
trial** on a `FrozenTrial` that has full access to `user_attrs` (where
metric values are stored at Step 8a of the objective). The function
returns a list of floats — one for budget check + one per soft metric
threshold. Any value > 0 marks the trial infeasible for
feasibility-guided sampling.

### Constraints

- Backwards compatible: all new parameters have defaults that preserve
  current behavior (`None` / `0.2`)
- `constraints_func` signature is `FrozenTrial -> Sequence[float]` —
  Optuna allows only one per sampler, so budget and metric constraints
  must be composed into a single function
- `torch.cuda` APIs are optional — must handle import failure gracefully
- Soft constraint values are only available after Step 8a (validation).
  For the very first trial, no prior metric values exist in user_attrs,
  so the constraint function must handle missing keys gracefully
  (treat as feasible)
- When `budget_aware=False` in `create_study()` and
  `metric_constraints_soft` is non-None: the factory is still invoked
  but skips the budget check portion (only metric thresholds are checked).
  This allows soft metric constraints without budget-aware sampling.
- Soft constraints are not applied during the QMC warm-up phase
  because `QMCSampler` does not accept `constraints_func`. Trials
  during the QMC phase may violate soft thresholds without penalty —
  this is a pre-existing gap shared with budget constraints.

### Open Questions

- None remaining.

## Implementation Plan

### Phase 1: Metric constraints dataclass + hard rejection

Add the constraint specification type, wire hard metric constraints
through `ObjectiveConfig` and `GenericObjective.__call__()`, and add
the `_detect_gpu_memory_mb()` utility (used in Phase 3).

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/optimization/objective.py` — Add
  `metric_constraints_hard` field to `ObjectiveConfig`. Add
  `_check_hard_constraints()` method to `GenericObjective`. Insert
  Step 8b in `__call__()` after the metrics user_attrs storage loop
  and before `metrics = {"summary": metrics_summary}`.
- `src/bayesflow_hpo/optimization/constraints.py` — Add
  `_detect_gpu_memory_mb(safety_margin)` function and
  `MetricConstraintSpec` type alias.
- `src/bayesflow_hpo/optimization/study.py` — Update
  `_count_non_rejected()` to use a whitelist of pre-training rejection
  reasons instead of checking for any `rejected_reason`. This ensures
  metric-rejected trials (which cost a full training run) count toward
  `max_total_trials`.
- `src/bayesflow_hpo/api.py` — Add `metric_constraints_hard` parameter
  to `optimize()` and `_build_objective()`.
- `tests/test_optimization/test_constraints.py` — Tests for
  `_detect_gpu_memory_mb()` with mocked `torch.cuda`.
- `tests/test_optimization/test_objective.py` — Tests for hard metric
  constraint rejection: constraint met (passes through), constraint
  violated (rejected + penalty + cleanup_trial called), missing metric
  in summary (passes through with warning), multiple constraints with
  partial violation.
- `tests/test_optimization/test_study.py` — Tests for updated
  `_count_non_rejected()`: budget-rejected trials excluded,
  metric-rejected trials counted, normal trials counted.
- `docs/optimization.md` — Document metric constraints usage.
- `docs/api_reference.md` — Add new parameters.

**Steps:**
1. Define `MetricConstraintSpec = tuple[str, float, str]` type alias
   in `constraints.py` with a docstring explaining the
   `(metric_name, threshold, "above"/"below")` convention. Docstring
   must clarify: `"above"` = reject when value exceeds threshold
   (metric must stay below), `"below"` = reject when value falls
   below threshold (metric must stay above).
2. Add `_detect_gpu_memory_mb(safety_margin=0.2) -> float | None` to
   `constraints.py`. Uses `torch.cuda.mem_get_info()` (returns
   `(free, total)` — use `free`). Exact formula:
   `return free_bytes * (1 - safety_margin) / (1024.0 ** 2)`.
   Returns `None` if CUDA unavailable. Handles `ImportError`
   (no torch) and `RuntimeError` (CUDA init failure).
3. Add `metric_constraints_hard: list[MetricConstraintSpec] | None = None`
   field to `ObjectiveConfig`.
4. Add `_check_hard_constraints(self, metrics_summary, trial)` method
   to `GenericObjective`. For each `(metric, threshold, direction)`:
   if metric missing from summary → log warning, skip. If direction
   is `"above"` and value > threshold → violated. If `"below"` and
   value < threshold → violated. On first violation: set
   `rejected_reason="metric_constraint"`, log at INFO level
   (include metric name, value, and threshold), return
   `self._penalty()`. Return `None` if all pass. Do NOT add any
   additional user_attrs for the specific metric — the INFO log
   already records the detail.
5. Insert Step 8b in `__call__()`: after the `for key, val in
   metrics_summary.items(): trial.set_user_attr(...)` loop and before
   `metrics = {"summary": metrics_summary}`. Exact calling pattern:
   ```python
   # --- Step 8b: Hard metric constraints ---
   if config.metric_constraints_hard is not None:
       penalty = self._check_hard_constraints(
           metrics_summary, trial,
       )
       if penalty is not None:
           cleanup_trial()
           return penalty
   ```
6. Update `_count_non_rejected()` in `study.py` to use a whitelist of
   pre-training rejection reasons:
   ```python
   _PRE_TRAINING_REJECTIONS = {
       "memory_budget", "param_budget", "build_failed",
       "compile_failed", "param_probe_failed",
   }

   def _count_non_rejected(study):
       return sum(
           1 for t in study.trials
           if t.user_attrs.get("rejected_reason") not in _PRE_TRAINING_REJECTIONS
       )
   ```
   This ensures metric-rejected trials count toward `max_total_trials`
   (they consumed GPU time) while budget-rejected trials still don't.
7. Thread `metric_constraints_hard` through `optimize()` →
   `_build_objective()` → `ObjectiveConfig`.
8. Write tests and update docs.

**Depends on:** None

### Phase 2: Soft constraints (feasibility-guided sampling)

Wire soft metric constraints through Optuna's `constraints_func`
mechanism. The constraint function is called after each completed trial
and inspects `trial.user_attrs` for stored metric values.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/optimization/study.py` — Replace
  `_budget_constraints_func` with `_make_constraints_func()` factory.
  When `soft_thresholds` is `None` and `budget_aware` is `True`, the
  returned closure is behaviourally identical to the old
  `_budget_constraints_func`. When `soft_thresholds` is provided, the
  closure checks budget (if `budget_aware`) + each threshold. Update
  `_resolve_sampler()` to accept `soft_thresholds` parameter and pass
  the factory result as `constraints_func` to all inner factories that
  support it: `_make_tpe`, `_make_gp`, `_make_botorch`, `_make_nsga2`,
  `_make_nsga3`. Leave `_make_auto` and `_make_random` unchanged (they
  do not accept `constraints_func`). Update `create_study()` to accept
  `metric_constraints_soft` and pass to `_resolve_sampler()`.
- `src/bayesflow_hpo/api.py` — Add `metric_constraints_soft` parameter
  to `optimize()`. Pass through to `_create_and_run_study()` →
  `create_study()`. Do NOT pass to `_build_objective()` (soft
  constraints are sampler-level). Add warning when
  `metric_constraints_soft` is non-None and `sampler` is a
  `BaseSampler` instance (not a string preset).
- `tests/test_optimization/test_study.py` — Update existing tests that
  import `_budget_constraints_func` or assert identity against it.
  Replace identity assertions (`is _budget_constraints_func`) with
  behavioral assertions (call the function with a mock trial and check
  return values). Add new tests for `_make_constraints_func()`: budget
  rejection, soft metric violation, soft metric pass, missing metric
  key, multiple constraints, combined budget + metric,
  `budget_aware=False` with soft thresholds.
- `tests/test_optimization/test_optimize_until.py` — Update tests that
  import `_budget_constraints_func` (lines 6, 39–46, 104) to use the
  factory instead.
- `tests/test_api.py` — Integration test: verify warning is logged when
  `metric_constraints_soft` is non-None with a user-provided sampler.
- `docs/optimization.md` — Document soft constraints.
- `docs/api_reference.md` — Add new parameters.

**Steps:**
1. Create `_make_constraints_func(budget_aware=True, soft_thresholds=None)`
   factory in `study.py`. Returns a `Callable[[FrozenTrial], list[float]]`
   closure. The closure returns a list: index 0 is the budget check
   (1.0 if `rejected_reason` in user_attrs else 0.0 — skipped if
   `budget_aware=False`), subsequent indices are soft metric violations
   (`max(0, value - threshold)` for `"above"`, `max(0, threshold - value)`
   for `"below"`, 0.0 if metric key missing from user_attrs).
2. Update `_resolve_sampler()` signature: add
   `soft_thresholds: list[MetricConstraintSpec] | None = None`. Replace
   `constraints = _budget_constraints_func if budget_aware else None`
   with `constraints = _make_constraints_func(budget_aware, soft_thresholds)
   if budget_aware or soft_thresholds else None`. Pass `constraints` to
   `_make_tpe`, `_make_gp`, `_make_botorch`, `_make_nsga2`, `_make_nsga3`.
   Leave `_make_auto` and `_make_random` unchanged.
3. Update `create_study()` to accept `metric_constraints_soft` and pass
   to `_resolve_sampler()`.
4. Add `metric_constraints_soft` to `optimize()` signature (default
   `None`). Add warning when non-None and `sampler` is a
   `BaseSampler` instance. Pass through `_create_and_run_study()` →
   `create_study()`.
5. Update existing tests in `test_study.py` and `test_optimize_until.py`
   that reference `_budget_constraints_func`. Replace **only** the
   `is _budget_constraints_func` identity assertions with behavioral
   assertions — preserve all other assertions in the same test methods
   (e.g. `_multivariate`, `_n_startup_trials`). Example behavioral
   assertion:
   ```python
   fn = sampler._constraints_func
   mock_trial = FrozenTrial(...)  # no rejected_reason
   assert fn(mock_trial) == [0.0]
   ```
   All `_constraints_func` assertions must go through `_resolve_sampler()`
   directly, NOT through `create_study()`, to avoid `QMCWarmupSampler`
   wrapping.
6. Add explicit test: `_resolve_sampler("tpe", budget_aware=False)` with
   no soft thresholds should give `_constraints_func is None` (preserves
   existing semantics).
7. Write new tests and update docs.

**Depends on:** Phase 1 (for `MetricConstraintSpec` type alias)

### Phase 3: GPU memory auto-detection

Wire `max_memory_mb="auto"` through `optimize()` using
`_detect_gpu_memory_mb()` from Phase 1.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/api.py` — Change `max_memory_mb` type annotation
  from `float | None` to `float | None | str` (accepting `"auto"`).
  Add `memory_safety_margin: float = 0.2` parameter to `optimize()`.
  Add `_resolve_memory_budget(max_memory_mb, safety_margin)` helper:
  if `float` or `None` → return as-is; if `"auto"` → call
  `_detect_gpu_memory_mb(safety_margin)` from constraints module;
  else → raise `ValueError`. Call before `_build_objective()`.
- `src/bayesflow_hpo/optimization/objective.py` — No changes needed
  (`ObjectiveConfig.max_memory_mb` stays `float | None` — resolution
  happens upstream).
- `tests/test_api.py` — Tests for `_resolve_memory_budget()`: "auto"
  with CUDA available (mock), "auto" without CUDA (returns None), float
  passthrough, None passthrough, invalid string raises ValueError.
- `docs/optimization.md` — Document auto memory detection.
- `docs/api_reference.md` — Add `memory_safety_margin` parameter.

**Steps:**
1. Change `max_memory_mb` type in `optimize()` to
   `float | None | str`.
2. Add `memory_safety_margin: float = 0.2` parameter to `optimize()`.
3. Implement `_resolve_memory_budget(max_memory_mb, safety_margin)` in
   `api.py`: if `float` or `None` → return as-is; if `"auto"` → call
   `_detect_gpu_memory_mb(safety_margin)` and log the resolved value
   at INFO; else → raise `ValueError`.
4. Call `_resolve_memory_budget()` in `optimize()` before
   `_build_objective()`. Pass the resolved float to `_build_objective()`.
5. Write tests and update docs.

**Depends on:** Phase 1 (for `_detect_gpu_memory_mb()`)

## Verification & Validation

- **Automated**:
  - Unit tests for `_detect_gpu_memory_mb()` with mocked `torch.cuda`
    (available, unavailable, CUDA init error)
  - Unit tests for `_check_hard_constraints()` (pass, violate, missing
    metric, multiple constraints, "above" and "below" directions)
  - Unit tests for `_make_constraints_func()` (budget only, soft only,
    combined, missing attrs, budget_aware=False with soft thresholds)
  - Unit tests for `_resolve_memory_budget()` (auto, float, None, invalid)
  - Updated existing tests that reference `_budget_constraints_func`
  - Integration test: `optimize()` with hard constraints rejects a trial
    that passes budget but fails metric threshold
  - Warning test: `optimize()` logs warning when soft constraints +
    user-provided sampler
  - CI: `ruff check` + `pytest` on Python 3.11/3.12/3.13
- **Manual**:
  - Run `optimize()` with
    `metric_constraints_hard=[("calibration_error", 0.5, "above")]`
    on Two Moons and verify trials with cal_error > 0.5 are rejected
  - Run with `max_memory_mb="auto"` on a CUDA machine and verify the
    resolved budget appears in logs

## Dependencies

- `torch` (optional, for `_detect_gpu_memory_mb()` — already an optional
  dependency via Keras backend)
- No new external dependencies

## Notes

_Living section — updated during implementation._

### 2026-04-04 — Implementation Notes

- Added `MetricConstraintSpec` and `_detect_gpu_memory_mb()` in
  `optimization/constraints.py`.
- Added hard metric constraint support to `ObjectiveConfig` and
  `GenericObjective` with post-validation rejection path:
  `rejected_reason="metric_constraint"`.
- Updated `_count_non_rejected()` to count metric-rejected trials and
  keep excluding only pre-training rejections via
  `_PRE_TRAINING_REJECTIONS`.
- Replaced budget-only sampler constraint function with composed
  `_make_constraints_func(...)` supporting soft metric constraints.
- Threaded `metric_constraints_soft` through
  `optimize()` → `_create_and_run_study()` → `create_study()`.
- Added warning for soft constraints with user-supplied sampler
  instances (cannot auto-inject constraints func).
- Added `_resolve_memory_budget()` and wired
  `max_memory_mb="auto"` + `memory_safety_margin`.
- Updated tests:
  `test_constraints.py`, `test_objective.py`, `test_study.py`,
  `test_optimize_until.py`, `test_api.py`.
- Updated docs: `docs/optimization.md`, `docs/api_reference.md`.
- Verification:
  - `ruff check` on all touched modules: pass
  - `pytest` targeted suite (5 files): 171 passed
- Shipping blocker:
  - `gh pr create` failed with `HTTP 401: Requires authentication`.
  - Branch is pushed: `feat/metric-constraints-memory-auto`.
  - Next step after re-auth: create PR from that branch into `main`.

## Review Feedback

**Iteration 1** — 10 findings (3 blockers, 4 warnings, 3 suggestions).
All addressed in revision:

- **BLOCKER**: Tests importing `_budget_constraints_func` would break.
  → Added test update steps to Phase 2, listing `test_study.py` and
  `test_optimize_until.py` explicitly.
- **BLOCKER**: User-provided sampler bypasses constraints.
  → Added design decision (warn + skip) and out-of-scope note.
- **BLOCKER**: Step 8.5 insertion point must be after user_attrs storage.
  → Clarified with explicit line numbers in Phase 1, Step 5.
- **WARNING**: Direction naming ambiguity (`"above"/"below"`).
  → Added design decision table row with clarification; added docstring
  requirement in Phase 1, Step 1.
- **WARNING**: `total` vs `free` VRAM.
  → Changed to use `free` from `mem_get_info()`.
- **WARNING**: `metric_constraints_soft` in ObjectiveConfig is redundant.
  → Removed from ObjectiveConfig; thread only through `create_study()`.
- **WARNING**: `budget_aware` + soft constraints interaction.
  → Added to Constraints section; factory accepts both parameters.
- **SUGGESTION**: Dead code (`_resolve_memory_budget`) between phases.
  → Moved to Phase 3.
- **SUGGESTION**: Inner factory functions need explicit listing.
  → Listed all 7 inner functions; noted which support `constraints_func`.
- **SUGGESTION**: Public export of `detect_gpu_memory_mb`.
  → Changed to private `_detect_gpu_memory_mb`.

**Iteration 2** — 10 findings (3 blockers, 5 warnings, 2 suggestions).
All addressed in revision:

- **BLOCKER**: Metric-rejected trials would corrupt `_count_non_rejected`
  causing runaway loops. → Added design decision to whitelist pre-training
  rejections; added `_count_non_rejected` update to Phase 1 Step 6 with
  `_PRE_TRAINING_REJECTIONS` set.
- **BLOCKER**: Calling pattern for `_check_hard_constraints` not specified,
  `cleanup_trial()` missing. → Added explicit code snippet in Phase 1
  Step 5 with `cleanup_trial()` call.
- **BLOCKER**: `rejected_metric` user_attr orphaned (nothing reads it).
  → Removed from Phase 1 Step 4; INFO log is sufficient.
- **WARNING**: `_make_constraints_func` list-length edge cases need tests.
  → Added explicit test case in Phase 2 Step 6.
- **WARNING**: `_constraints_func` tests must avoid `QMCWarmupSampler`
  wrapping. → Added note to Phase 2 Step 5.
- **WARNING**: Soft constraints inactive during QMC warm-up.
  → Added note to Constraints section.
- **WARNING**: Line numbers used as insertion coordinates drift.
  → Replaced with code-pattern anchors throughout.
- **WARNING**: bytes→MB formula not specified.
  → Added exact formula in Phase 1 Step 2.
- **SUGGESTION**: `cleanup_trial()` missing from rejection path.
  → Added to Phase 1 Step 5 code snippet.
- **SUGGESTION**: Behavioral test replacement must preserve other assertions.
  → Added explicit guidance in Phase 2 Step 5.

Plan reviewed in 2 iterations.
