# Plan: Validation Contract Cleanup (Package D remaining)

**Created**: 2026-03-21
**Spec**: `docs/spec-validation-contract-cleanup.md`

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-21 | |
| Phase 1: Docstrings + type tightening | MERGED | 2026-03-21 | PR #50 |
| Ship | MERGED | 2026-03-21 | |

## Summary

**Motivation**: The `validate_fn` hook contract (required keys, penalty
fallback, timing semantics) is enforced in code but not documented.
`ObjectiveConfig.validation_data` is typed `Optional` but never `None`
in practice — the type annotation is misleading and the `None` fallback
branch is dead code.

**Outcome**: Users of `validate_fn` can read the contract in docstrings.
`ObjectiveConfig` has a tighter, more honest type. Dead code removed.

## Assumptions

- `ObjectiveConfig` is internal — no external code constructs it
  directly. Tightening `validation_data` to non-Optional is safe.
- All `ObjectiveConfig` construction sites in tests (~19 explicit
  `validation_data=None` + ~4 implicit omissions relying on the default)
  can trivially switch to a minimal `ValidationDataset` sentinel (the
  dataclass just needs lists and a seed — no real data needed for tests
  that don't run validation).

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| validate_fn contract | Docstring-only vs. docstring + runtime warning | Docstring-only | Contract already enforced in `_validate_metric_keys()`; runtime warning would be noise |
| Timing semantics | Document vs. change signature vs. drop for custom path | Document limitation | Custom validate_fn users have non-standard inference; metric computation time is typically small |
| validation_data optionality | Non-Optional vs. keep Optional vs. expose validate=False | Non-Optional | `optimize()` always provides data; None path was dead code |

## Scope

### In Scope
- Docstring updates to `ValidateFn`, `optimize()`, `ObjectiveConfig`
  (contract details, timing caveat, intermediate pruning note)
- Type change: `ObjectiveConfig.validation_data` from
  `ValidationDataset | None` to `ValidationDataset`
- Remove dead `None` fallback branch in `GenericObjective.__call__`
- Remove redundant `is not None` guards in objective
- Update ~15 test sites from `validation_data=None` to a minimal
  `ValidationDataset` sentinel

### Out of Scope
- Making intermediate validation metrics configurable (Package A1)
- Adding `validate=False` to `optimize()` (future work)
- Changing `ValidateFn` signature

## Implementation Plan

### Phase 1: Docstrings + type tightening

Single phase — all changes are small, tightly coupled, and not
independently shippable in meaningful sub-phases.

**Files to modify:**
- `src/bayesflow_hpo/types.py` — expand `ValidateFn` docstring
- `src/bayesflow_hpo/api.py` — expand `validate_fn` param docs in
  `optimize()`
- `src/bayesflow_hpo/optimization/objective.py` — expand
  `ObjectiveConfig.validate_fn` docstring; change
  `validation_data` type; remove `None` fallback and guards
- `tests/test_optimization/test_objective.py` — replace
  `validation_data=None` with minimal `ValidationDataset`
- `tests/test_optimization/test_multi_objective_pruning.py` — same

**Steps:**

1. **Add a `_DUMMY_VALIDATION_DATA` sentinel to test helpers.**
   Add `from bayesflow_hpo.validation.data import ValidationDataset`
   and create a module-level sentinel in each test file that needs it.
   This is the minimal valid instance — tests that don't exercise
   validation don't need real data, just a non-None value to satisfy
   the type.

   ```python
   from bayesflow_hpo.validation.data import ValidationDataset

   _DUMMY_VALIDATION_DATA = ValidationDataset(
       simulations=[],
       condition_labels=[],
       param_keys=["p"],
       data_keys=["x"],
       seed=0,
   )
   ```

2. **Replace all `validation_data=None` and implicit omissions in
   tests** with `_DUMMY_VALIDATION_DATA`. Cover both explicit
   `validation_data=None` (~16 sites in `test_objective.py`, ~3 in
   `test_multi_objective_pruning.py`) **and** implicit omissions where
   `ObjectiveConfig(...)` is called without a `validation_data` arg
   (~4 sites in `test_objective.py` that rely on the `= None` default).
   Grep for all `ObjectiveConfig(` calls to ensure none are missed.

3. **Tighten `ObjectiveConfig.validation_data` type.**
   Change from `ValidationDataset | None = None` to `ValidationDataset`
   (positional or required keyword — no default). Add an `isinstance`
   check in `__post_init__` to catch accidental `None` at construction
   time with a clear error message.

4. **Remove dead code in `GenericObjective.__call__`.**
   - Remove the `else` branch (objective.py:660-666) that substitutes
     penalty values when `validation_data is None`.
   - Remove the `if config.validation_data is not None:` guard around
     `PeriodicValidationCallback` injection (objective.py:573) — always
     inject the callback.
   - Remove the `if config.validation_data is not None:` guard in the
     validation step (objective.py:616) — always run validation.

5. **Expand docstrings** (no behavioral change):
   - `ValidateFn` (`types.py`): document required keys, penalty
     fallback, extra-key behavior, timing caveat, intermediate
     pruning note.
   - `optimize()` `validate_fn` param (`api.py`): same contract
     details + timing note.
   - `ObjectiveConfig.validate_fn` (`objective.py`): same.

6. **Run tests and lint.**

**Depends on:** None

## Verification & Validation

- **Automated**: `pytest tests/ -v` — all existing tests must pass
  after replacing `None` with the dummy sentinel. No new tests needed
  (no behavioral change).
- **Automated**: `ruff check src/ tests/` — lint clean.
- **Manual**: Read the three updated docstrings and confirm the
  contract is clear and consistent across all locations.
- **Manual**: Confirm `ObjectiveConfig` no longer accepts
  `validation_data=None` (attempt construction in a REPL → TypeError).

## Dependencies

- None — purely internal changes.

## Notes

_Living section — updated during implementation._

## Review Feedback

Reviewed in 1 iteration. 11 findings (3 blockers, 5 warnings,
3 suggestions). All blockers addressed in plan revision:

1. **BLOCKER (fixed)**: Undercounted test sites — ~4 implicit omissions
   of `validation_data` (no arg at all, relying on default). Plan now
   says to grep all `ObjectiveConfig(` calls.
2. **BLOCKER (fixed)**: Missing `ValidationDataset` import in test
   files. Plan Step 1 now includes the import statement.
3. **BLOCKER (clarified)**: `PeriodicValidationCallback` test sites
   (3 in `test_multi_objective_pruning.py`) — already covered by
   Step 2 replacement; mock prevents real validation from running.
4. **WARNING (accepted)**: Line references may drift — verify before
   patching.
5. **WARNING (accepted)**: `metrics` variable bind path after removing
   `else` branch confirmed safe (early return in `except`).
6. **WARNING (fixed)**: Added `isinstance` check in `__post_init__`
   to catch `None` at construction time.
7. **SUGGESTION (noted)**: `PeriodicValidationCallback.validation_data`
   annotation is `Any` — could be tightened in a follow-up. Out of
   scope for this package.
