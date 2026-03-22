# Plan: Sampler Presets

**Created**: 2026-03-22
**Author**: Claude Code + Matze

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Spec | DONE | 2026-03-22 | 3 blockers resolved |
| Plan | DONE | 2026-03-22 | 3 phases, batched into 1 PR |
| Phase 1: Sampler presets in create_study() | MERGED | 2026-03-22 | PR #56 |
| Phase 2: Wire sampler through optimize() | MERGED | 2026-03-22 | PR #56 |
| Phase 3: Pruning warmup alignment | MERGED | 2026-03-22 | PR #56 |
| Ship | MERGED | 2026-03-22 | All phases in one PR |

## Spec

_Design decisions and requirements — the "what and why". Produced by
brainstorming._

### Summary

**Motivation**: Users currently must construct Optuna sampler objects
manually to use anything other than the default TPE. This requires
knowing Optuna's API, wiring `constraints_func` for budget-aware
sampling, and choosing sensible defaults for each sampler. The
upcoming benchmark paper needs TPE, GP, and NSGA-II as first-class
options.

**Outcome**: Users can pass `sampler="gp"` to `create_study()` or
`optimize()` and get a correctly configured sampler with budget-aware
constraints, appropriate startup trials, and sensible defaults. Seven
presets cover all major Optuna sampler families.

### Requirements

- R1: `create_study(sampler="tpe")` creates a `TPESampler` with the
  current defaults (`multivariate=True`, `n_startup_trials=25`,
  `seed=42`, `warn_independent_sampling=False`).
- R2: `create_study(sampler="gp")` creates a `GPSampler` with
  `seed=42`, `n_startup_trials=10`.
- R3: `create_study(sampler="botorch")` creates a `BoTorchSampler`
  with `n_startup_trials=10`. Raises `ImportError` with install
  instructions if `optuna-integration[botorch]` is not installed.
- R4: `create_study(sampler="nsga2")` creates an `NSGAIISampler`
  with `population_size=50`, `seed=42`.
- R5: `create_study(sampler="nsga3")` creates an `NSGAIIISampler`
  with `population_size=50`, `seed=42`.
- R6: `create_study(sampler="auto")` creates an `AutoSampler` with
  `seed=42`. `AutoSampler` does not exist in Optuna 4.7.0 (neither in
  `optuna.samplers` nor `optuna_integration`). The preset is included
  for forward compatibility but uses a lazy import from
  `optuna.samplers` and raises `ImportError` with a message stating
  the minimum Optuna version required (once identified).
- R7: `create_study(sampler="random")` creates a `RandomSampler`
  with `seed=42`.
- R8: All presets that support `constraints_func` (TPE, GP, NSGA-II,
  NSGA-III, BoTorch) auto-wire `_budget_constraints_func` when
  `budget_aware=True`.
- R9: Passing a sampler object (`BaseSampler` instance) continues to
  work unchanged — no string resolution, no constraints injection.
- R10: `optimize()` gains a `sampler` parameter that is passed through
  to `_create_and_run_study()` → `create_study()`. Both intermediate
  functions must add the parameter to their signatures.
- R11: Pruning warmup `n_startup_trials` is auto-detected from the
  sampler: `sampler.n_startup_trials` for TPE/GP/BoTorch,
  `sampler.population_size` for NSGA-II/III, 10 for Random, 10 for
  Auto. User override via `pruning_n_startup_trials` on `optimize()`.
- R12: Unknown preset strings raise `ValueError` with the list of
  valid presets (same pattern as `_resolve_pruner`).

### Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Preset scope | Core 4 + Random; All 7; Benchmark 3 | All 7 | Future-proof for the benchmark paper and general use |
| NSGA population size | Fixed 50; Heuristic; Defer | Defer (use 50) | Requires research; Optuna default is reasonable; users can override |
| API surface | create_study() only; Both | Both optimize() and create_study() | Most users call optimize() directly |
| Missing deps | ImportError; Warn+fallback | ImportError | Explicit failure is better than silent substitution |
| Warmup alignment | This package; Separate | This package | Touches the same code path; natural fit |
| constraints_func for Random/QMC | Inject; Skip | Skip | Random/QMC don't accept constraints_func |

### Scope

#### In Scope
- `_resolve_sampler()` function in `optimization/study.py` (mirrors
  `_resolve_pruner()`)
- Type change: `sampler: str | BaseSampler | None` on `create_study()`
- New `sampler` parameter on `optimize()`, passed through to
  `_create_and_run_study()` → `create_study()`
- Smarter `n_startup_trials` auto-detection for pruning warmup
- Tests for all 7 presets, budget-aware wiring, error cases
- Lazy imports for BoTorch and Auto with clear error messages

#### Out of Scope
- NSGA-II/III population size heuristics (deferred to research TODO)
- QMC warm-up (Package A3, separate feature)
- Changing the default sampler from TPE to anything else
- Sampler-specific HP tuning (each sampler's internal scaling is fine)
- Exposing sampler sub-parameters through presets (use object form)

### Architecture Overview

```
optimize()
  └─ sampler: str | BaseSampler | None  (new param, default None)
      └─ _create_and_run_study()
          └─ create_study(sampler=...)
              ├─ str  → _resolve_sampler(name, budget_aware)
              │         └─ returns configured BaseSampler
              ├─ BaseSampler → use as-is (existing behavior)
              └─ None → _resolve_sampler("tpe", budget_aware)
```

`_resolve_sampler()` follows the same pattern as `_resolve_pruner()`:
a dict of preset name → factory lambda. Budget-aware constraints are
injected into the factory for samplers that support `constraints_func`.

For BoTorch and Auto, the factory lambda does a lazy import and raises
`ImportError` with install instructions if the dependency is missing.

Pruning warmup detection in `_create_and_run_study()` is extended:
```python
# Current: getattr(study.sampler, "n_startup_trials", 10)
# New: check n_startup_trials first, then population_size, then 10
n = getattr(sampler, "n_startup_trials", None)
if n is None:
    n = getattr(sampler, "population_size", 10)
```

### Constraints

- Backwards compatible: `sampler=None` must produce the same TPE as
  today (seed=42, multivariate=True, n_startup_trials=25)
- No new required dependencies — BoTorch and Auto are optional
- `pyproject.toml` currently declares `optuna>=3.0.0`. `GPSampler`
  requires >= 3.6, `metric_names` in `create_study` requires >= 4.0.
  Bump to `optuna>=4.0.0` as part of this package.
- `constraints_func` wiring must not break when `budget_aware=False`
  (string presets with `budget_aware=False` create samplers without
  `constraints_func`)

### Open Questions

- Should BoTorch default `n_startup_trials` be 10 or 5? (BoTorch
  is more sample-efficient than TPE, but 10 is Optuna's GP default.)
  → Default to 10 for now, revisit in the research TODO.
- `AutoSampler` does not exist in Optuna 4.7.0. Need to identify
  which Optuna version introduces it (or whether it requires a
  separate package). Include version in the error message.
- `GPSampler` emits an Optuna experimental-API warning when
  `constraints_func` is set. Decide whether to suppress with
  `warnings.filterwarnings` or document as expected.
- `PeriodicValidationCallback` default `n_startup_trials=5` is lower
  than the auto-detect fallback of 10. Consider aligning to 10 or
  documenting the difference.

## Implementation Plan

### Phase 1: Sampler presets in create_study()

**Files to modify:**
- `src/bayesflow_hpo/optimization/study.py`
- `pyproject.toml`

**Steps:**
1. Add `_resolve_sampler(name: str, budget_aware: bool) -> BaseSampler`
   to `study.py`, following the `_resolve_pruner()` pattern. Factory
   dict with 7 entries. BoTorch and Auto use lazy imports with clear
   `ImportError` messages.
2. Refactor `create_study()`: change `sampler: Any | None` to
   `sampler: str | optuna.samplers.BaseSampler | None`. When `str`,
   call `_resolve_sampler(name, budget_aware)`. When `None`, call
   `_resolve_sampler("tpe", budget_aware)`. When `BaseSampler`, use
   as-is (existing behaviour). Remove the inline TPE construction.
3. Bump `optuna>=3.0.0` to `optuna>=4.0.0` in `pyproject.toml`.
4. Update `create_study()` docstring with a preset table (same format
   as the pruner table).

**Tests** (`tests/test_optimization/test_study.py`):
- `TestResolveSampler`: one test per preset verifying type + key attrs
  (seed, n_startup_trials, population_size, multivariate,
  constraints_func).
- `test_budget_aware_false_no_constraints`: verify `budget_aware=False`
  produces sampler without `constraints_func`.
- `test_invalid_string_raises`: `ValueError` with preset list.
- `test_custom_sampler_passed_through`: `BaseSampler` instance.
- `test_default_none_creates_tpe`: `sampler=None` backwards compat.
- `test_botorch_missing_import`: mock the import to raise, verify
  `ImportError` message.
- `test_auto_missing_import`: same pattern.

**Depends on:** None

### Phase 2: Wire sampler through optimize()

**Files to modify:**
- `src/bayesflow_hpo/api.py` (`optimize()`, `_create_and_run_study()`)
- `src/bayesflow_hpo/__init__.py` (no new exports needed)

**Steps:**
1. Add `sampler: str | optuna.samplers.BaseSampler | None = None` to
   `optimize()` signature (after `show_progress_bar`).
2. Pass `sampler` through to `_create_and_run_study()`.
3. Add `sampler` to `_create_and_run_study()` signature and forward
   to `create_study(sampler=sampler)`.
4. Update `optimize()` docstring with sampler parameter docs.

**Tests** (`tests/test_api.py`):
- `test_optimize_passes_sampler_to_create_study`: mock `create_study`
  and verify `sampler=` is forwarded.

**Depends on:** Phase 1

### Phase 3: Pruning warmup alignment

**Files to modify:**
- `src/bayesflow_hpo/api.py` (`_create_and_run_study()`)
- `src/bayesflow_hpo/optimization/study.py` (add
  `_resolve_n_startup_trials()` helper)

**Steps:**
1. Add `_resolve_n_startup_trials(sampler: BaseSampler) -> int` to
   `study.py`: checks `n_startup_trials` first, then
   `population_size`, fallback 10.
2. In `_create_and_run_study()`, replace
   `getattr(study.sampler, "n_startup_trials", 10)` with
   `_resolve_n_startup_trials(study.sampler)`.
3. Update the auto-detect log message to include which attribute was
   used.

**Tests** (`tests/test_optimization/test_study.py`):
- `TestResolveNStartupTrials`: test with TPESampler (25), GPSampler
  (fallback 10), NSGAIISampler (50 via population_size),
  RandomSampler (fallback 10), bare object (fallback 10).

## Verification & Validation

- **Automated**: Unit tests for each preset, budget-aware wiring,
  missing dependency errors, pruning warmup detection, optimize()
  pass-through
- **Manual**: Run the Two Moons example with `sampler="gp"` and
  `sampler="nsga2"` to verify end-to-end

## Dependencies

- `optuna >= 4.0.0` (bump from current `>=3.0.0` in `pyproject.toml`)
- `optuna-integration[botorch]` (optional, for `"botorch"` preset)

## Notes

_Living section — updated during implementation._

## Review Feedback

### Spec Review (2026-03-22, code-architect agent)

14 findings: 3 blockers, 5 warnings, 6 suggestions.

**Blockers (all resolved):**
- F3: `AutoSampler` not found in Optuna 4.7.0 → Updated R6 to note
  forward-compatibility with lazy import; added to Open Questions.
- F6: `pyproject.toml` says `optuna>=3.0.0`, spec said `>=4.0` →
  Updated Constraints to include pyproject.toml bump as a deliverable.
- C1: `_create_and_run_study()` signature change not spelled out →
  Updated R10 to explicitly mention both intermediate function sigs.

**Warnings (acknowledged):**
- F1: `GPSampler.n_startup_trials` is private; warmup falls back to
  10 by coincidence. Correct for GP's default. Noted.
- F4: `BoTorchSampler` import path — use `optuna.integration` which
  provides a clear error message. Noted for implementation.
- F5: `GPSampler` `constraints_func` emits experimental warning →
  Added to Open Questions.
- C5: `budget_aware=False` behaviour for string presets → Updated
  Constraints section to spell this out.
- K3: Callback default `n_startup_trials=5` vs auto-detect 10 →
  Added to Open Questions.
- K4: `pruning_n_startup_trials` override on `optimize()` →
  Already in R11; plan phase should list it explicitly.

**Suggestions (noted for plan phase):**
- C2: Read warmup from resolved sampler, not `study.sampler`.
- C3: Update `create_study()` docstring with preset table.
- C4: `_resolve_sampler` stays internal (not exported in `__init__`).
- S1: Phase 3 warmup detection applies to user-supplied objects too
  (same `getattr` chain, no special-casing needed).
- S2: Type annotation `Any → str|BaseSampler|None` is narrowing,
  backwards-compatible at runtime. Mypy impact minimal.
