# Plan: Package I — Small API Fixes

**Created**: 2026-03-22
**Author**: Claude Code

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Spec | DONE | 2026-03-22 | Decisions resolved in brainstorm |
| Plan | DONE | 2026-03-22 | |
| Phase 1: All three fixes + tests | MERGED | 2026-03-22 | PR #55 |
| Ship | MERGED | 2026-03-22 | |

## Spec

See [spec-small-api-fixes.md](spec-small-api-fixes.md) for the full
spec produced during brainstorming.

**Summary**: Three standalone fixes that improve error reporting and
edge-case correctness — `normalize_param_count` ValueError on
degenerate bounds, debug logging in `infer_keys_from_adapter`, and
`data_keys` validation in `make_bayesflow_infer_fn`.

**Design decisions** (settled):
- D1: Explicit `param_keys`/`data_keys` in `optimize()` — dropped (YAGNI)
- D2: `ValueError` over neutral return for degenerate bounds
- D3: Minimal logging scope (one `logger.debug()` line)

## Implementation Plan

### Phase 1: All three fixes + tests

All three fixes are small and independent. They ship as one PR to
minimize review overhead.

**Files to modify:**
- `src/bayesflow_hpo/objectives.py` — R1: ValueError guard
- `src/bayesflow_hpo/api.py` — R2: debug logging
- `src/bayesflow_hpo/validation/inference.py` — R3: available_keys param + remove silent skip
- `src/bayesflow_hpo/validation/pipeline.py` — R3: pass available_keys to make_bayesflow_infer_fn
- `tests/test_objectives.py` — R1: tests for ValueError and preserved edge cases
- `tests/test_infer_keys.py` — R2: test for debug log on missing transforms

**Files to create:**
- `tests/test_validation/test_inference.py` — R3: tests for data_keys validation

**Steps:**

1. **R1: `normalize_param_count` ValueError**
   - In `objectives.py`, replace line 95–96 (`if max_count <= min_count: return 0.0`)
     with `raise ValueError(f"max_count ({max_count}) must be greater than min_count ({min_count})")`
   - Add matching guard in `denormalize_param_count` (line 101–112) for
     consistency — same condition raises the same error
   - Update docstrings for both functions to document the `ValueError`
     under a `Raises` section

2. **R2: Debug logging in `infer_keys_from_adapter`**
   - In `api.py`, add `logger.debug("Adapter has no 'transforms' attribute; skipping key inference")`
     between lines 65 and 66 (before `return result`)

3. **R3: Validate `data_keys` in `make_bayesflow_infer_fn`**
   - In `validation/inference.py`, add `available_keys: set[str] | None = None`
     parameter to `make_bayesflow_infer_fn`
   - Add validation block: if `available_keys` is not None, compute
     `missing = set(data_keys) - available_keys` and raise `KeyError`
     if non-empty
   - Remove `if k in sim_data` guard from the closure — change to
     `conditions = {k: sim_data[k] for k in data_keys}`
   - In `validation/pipeline.py` line 52–56, pass
     `available_keys=set(validation_data.simulations[0].keys()) if validation_data.simulations else None`
     to `make_bayesflow_infer_fn` (guards against empty datasets)

4. **Tests**
   - `test_objectives.py`: add `TestNormalizeParamCount` class:
     - `test_raises_on_max_le_min` — explicit `max_count=10, min_count=100`
     - `test_raises_on_max_eq_min` — `max_count=100, min_count=100`
     - `test_raises_after_auto_tightening` — `max_count=1` (auto-tightened
       `min_count` = `max(1, 0)` = 1, so `max_count <= min_count`)
     - `test_zero_param_count_returns_worst` — preserves existing behavior
     - `test_negative_min_count_autocorrected` — preserves existing behavior
     - `test_denormalize_raises_on_max_le_min` — explicit equal bounds
       (`max_count=100, min_count=100`), not relying on auto-tighten
       (denormalize has no auto-tighten path)
   - `test_infer_keys.py`: add `test_debug_log_on_missing_transforms` —
     adapter with no `transforms` attr, assert `logger.debug` called
     (patch target: `"bayesflow_hpo.api.logger"`)
   - `tests/test_validation/test_inference.py`: new file:
     - `test_missing_data_keys_raises_keyerror` — pass `available_keys`
       missing a required key
     - `test_available_keys_none_skips_check` — no error when param is None
     - `test_closure_raises_on_missing_key` — call the returned `infer_fn`
       with a dict missing a data key, assert `KeyError`

**Depends on:** None

## Verification & Validation

- **Automated**: `pytest tests/test_objectives.py tests/test_infer_keys.py tests/test_validation/test_inference.py -v`
  plus full `pytest tests/ -v` and `ruff check src/ tests/`
- **Manual**: None needed — all changes are guard clauses with clear test coverage

## Dependencies

- No new dependencies

## Notes

- `denormalize_param_count` has the same degenerate-bounds issue —
  when `max_count == min_count`, `log10(max_count / min_count)` = `log10(1)` = 0,
  producing a silent wrong answer (not a ZeroDivisionError, since numpy
  returns `inf` for `x / 0.0`). Adding the same ValueError guard for
  consistency.
- The `available_keys` parameter on `make_bayesflow_infer_fn` defaults
  to `None` for backwards compatibility — external callers who import
  the function directly are unaffected.
- `denormalize_param_count` is not in the existing `test_objectives.py`
  import block — the test must add it.
- `tests/test_validation/` already exists with test files but no
  `__init__.py` — pytest collects them fine via `testpaths` config.
  New test file follows the same pattern.

## Review Feedback

Plan reviewed in 1 iteration (6 findings: 2 blockers, 2 warnings, 2 suggestions).

**Blockers (resolved):**
- F3: `denormalize_param_count` has no auto-tighten path — fixed test
  description to use explicit equal bounds
- F4: `simulations[0]` raises `IndexError` on empty `ValidationDataset` —
  added ternary guard in pipeline.py step

**Warnings (noted):**
- F1: `denormalize` failure mode is silent wrong answer, not divide-by-zero —
  corrected rationale in Notes section
- F2: No `__init__.py` in `tests/test_validation/` — verified existing tests
  collect fine without it

**Suggestions (incorporated):**
- F5: Patch target for logger test must be `"bayesflow_hpo.api.logger"` — noted in test step
- F6: `denormalize_param_count` must be added to `test_objectives.py` imports — noted in Notes
