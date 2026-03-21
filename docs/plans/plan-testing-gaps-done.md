# Plan: Package F — Testing Gaps

**Created**: 2026-03-21
**Author**: Claude Code

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-21 | |
| Phase 1: warm_start_study edge cases | MERGED | 2026-03-21 | 6 new tests |
| Phase 2: _training_loss_fallback edge cases | MERGED | 2026-03-21 | 4 new tests |
| Phase 3: Validation data edge cases | MERGED | 2026-03-21 | 6 new tests |
| Ship | MERGED | 2026-03-21 | [PR #47](https://github.com/matthiaskloft/bayesflow-hpo/pull/47) |

## Summary

**Motivation**: Package F aims to close remaining test gaps before
refactoring or extending other packages. However, since the TODO was
written, significant test coverage has been added:
- `warm_start_study` already has 5 tests (top-k, ranking keys for 1/2/3
  objectives, no-values sentinel)
- `_training_loss_fallback` already has 8+ tests (pareto/mean modes,
  clamping, None fallback, end-to-end with validation failure)
- `load_validation_dataset` round-trip already has a full save→load test
- `make_condition_grid` already has 4 tests (linspace, logspace, values,
  combined)

**Outcome**: After this plan, each function has edge-case coverage that
protects against regressions during the Package C/D refactors. The
remaining gaps are boundary conditions and error paths not yet exercised.

## Assumptions

- Tests must not require BayesFlow or Keras (pure-Python / NumPy / Optuna
  only), matching existing test conventions.
- The existing `conftest.py` fixtures (`DummySimulator`, `FakeTrial`) are
  sufficient — no new shared fixtures needed.

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Phase granularity | One phase per TODO item vs. one big phase | One phase per TODO item | Each is independently shippable and small; keeps PRs focused |
| Test placement | New files vs. extend existing | Extend existing test files | Tests for each function already live in the right file; adding more there keeps related tests together |

## Scope

### In Scope

- Edge-case tests for `warm_start_study`: empty source, top_k=0,
  top_k > available trials, mixed trial states, user_attr preservation
- Edge-case tests for `_training_loss_fallback`: negative loss (clamp to 0),
  loss exactly at boundaries (0.0 and 1.0), single-metric pareto mode
- Edge-case tests for `load_validation_dataset`: missing file error,
  `sim_time_per_sim` round-trip preservation, multi-key batches
- Edge-case tests for `make_condition_grid`: all three modes combined,
  all-None/empty inputs, single-point grids (n=1)

### Out of Scope

- Integration tests requiring BayesFlow/Keras imports
- Tests for `PeriodicValidationCallback` (belongs to Package E pruning review)
- Tests for `GenericObjective.__call__` end-to-end (already well-covered)
- New test infrastructure or fixtures

## Implementation Plan

### Phase 1: warm_start_study edge cases

**Files to create:** None

**Files to modify:**
- `tests/test_optimization/test_study_warm_start.py`

**Steps:**
1. Add `test_warm_start_empty_source` — source study with no COMPLETE
   trials returns 0 and target stays empty
2. Add `test_warm_start_top_k_zero` — `top_k=0` adds nothing
3. Add `test_warm_start_negative_top_k` — negative `top_k` adds nothing
   (guarded by `max(0, int(top_k))` in implementation)
4. Add `test_warm_start_top_k_exceeds_available` — `top_k=10` with 3
   trials copies all 3
5. Add `test_warm_start_skips_non_complete_trials` — source with a mix of
   COMPLETE, FAIL, PRUNED, and RUNNING trials only copies COMPLETE ones
6. Add `test_warm_start_preserves_user_attrs` — user attrs from source
   trials appear on target trials
7. Run tests: `pytest tests/test_optimization/test_study_warm_start.py -v`

**Depends on:** None

### Phase 2: _training_loss_fallback edge cases

**Files to create:** None

**Files to modify:**
- `tests/test_optimization/test_objective.py`

**Steps:**
1. Add `test_training_loss_fallback_clamps_negative` — negative training
   loss is clamped to 0.0
2. Add `test_training_loss_fallback_exact_zero` — loss=0.0 passes through
   unchanged
3. Add `test_training_loss_fallback_exact_one` — loss=1.0 passes through
   unchanged
4. Add `test_training_loss_fallback_single_metric_pareto` — pareto mode
   with 1 metric returns `(clamped_loss, cost_score)` — same shape as
   mean mode but different semantics (single metric vs. averaged metric)
5. Run tests: `pytest tests/test_optimization/test_objective.py -v`

**Depends on:** None

### Phase 3: Validation data edge cases

**Files to create:** None

**Files to modify:**
- `tests/test_validation/test_validation_data.py`

**Steps:**
1. Add `test_load_validation_dataset_missing_dir` — raises
   `FileNotFoundError` for nonexistent path
2. Add `test_validation_dataset_round_trip_preserves_sim_time` — save/load
   preserves `sim_time_per_sim` value
3. Add `test_validation_dataset_round_trip_multi_key` — dataset with
   multiple param/data keys round-trips correctly
4. Add `test_make_condition_grid_all_modes` — linspace + logspace + values
   combined in one call
5. Add `test_make_condition_grid_all_none` — all None/omitted returns
   empty dict
6. Add `test_make_condition_grid_single_point` — `n=1` produces a
   single-element list
7. Run tests: `pytest tests/test_validation/test_validation_data.py -v`

**Depends on:** None

## Verification & Validation

- **Automated**: All new tests pass (`pytest tests/ -v`), ruff clean
  (`ruff check src/ tests/`), CI green
- **Manual**: Review each test to confirm it exercises a genuinely new
  code path (not redundant with existing tests). Check coverage of the
  target functions increased by inspecting which branches the new tests
  exercise.

## Dependencies

- None beyond existing dev dependencies (pytest, optuna, numpy)

## Notes

All 3 phases implemented in a single session. 61 tests pass, ruff clean.

Simplification pass applied: extracted `_FLOAT_DIST` constant and
`_build_failed_trial()` helper in warm-start tests, extracted
`_single_metric_pareto_fallback()` helper in objective tests, merged
`top_k=0` and `top_k=-3` into a parametrized test, moved inline pytest
import to module level.

Review suggestions deferred (out of scope for Package F):
- Test `max_memory_mb` rejection path (objective.py:475-484)
- Test build failure path in GenericObjective (objective.py:487-501)
- Test `TrialPruned` re-raise from training (objective.py:598-600)
- Test partial/corrupted save in `load_validation_dataset`
These are valuable additions but belong in future test expansion work.

## Review Feedback

Reviewed in 1 iteration. 6 findings from reviewer (4 blockers, 2
warnings); all addressed:

- **Negative top_k test** (blocker → added step 3 in Phase 1)
- **Include RUNNING state** (blocker → updated step 5 in Phase 1)
- **Single-metric pareto semantics** (blocker → clarified step 4 in Phase 2)
- **String constant validation** (blocker → confirmed all string literals
  match codebase: `"pareto"`, `"mean"`, `"inference_time"`, `"param_count"`)
- **Missing file test scope** (warning → Phase 3 step 1 covers the primary
  path; both files missing is the common case since the entire dir is
  typically absent)
- **Multi-key round-trip** (warning → already in Phase 3 step 3)
