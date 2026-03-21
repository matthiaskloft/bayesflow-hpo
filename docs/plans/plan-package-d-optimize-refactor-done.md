# Plan: Package D — `optimize()` Refactor

**Created**: 2026-03-21
**Author**: Claude
**Spec**: `docs/spec-package-d-optimize-refactor.md`

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-21 | |
| Phase 1: Extract helpers from `optimize()` | MERGED | 2026-03-21 | PR #49 |
| Phase 2: Fix `_TrackingDict` + deduplicate registration | MERGED | 2026-03-21 | PR #49 |
| Ship | MERGED | 2026-03-21 | All phases shipped in one PR |

## Summary

**Motivation**: `optimize()` in `api.py` is ~130 lines mixing orchestration
with inline logic for validation data setup, objective construction, direction
derivation, and study creation. `_TrackingDict` has false-positive
unused-key warnings when builders iterate via `items()`/`values()`. Builder
registration has duplicated alias loops.

**Outcome**: `optimize()` becomes a 6-step linear orchestrator calling named
private helpers. `_TrackingDict` correctly tracks `items()`/`values()` iteration.
Registration dedup reduces repeated code. All changes are internal — no public
API changes.

## Assumptions

- The spec's 5 extracted helpers map cleanly to the current `optimize()` code
  (verified by reading `api.py` lines 96-429).
- No external consumers depend on `optimize()`'s internal structure.
- `_TrackingDict` is only used in `check_pipeline()`, so the `items()`/`values()`
  fix only affects pre-flight validation.

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Helper count | 2 (minimal) vs 5 (full decomposition) | 5 | Full decomposition makes `optimize()` self-documenting; each helper is independently testable |
| `_TrackingDict.items()`/`values()` | (a) Track all via `__iter__`, (b) Track `items()`+`values()` only, (c) Document limitation | (b) | `__iter__` would falsely mark all keys on `dict(td)`; `items()`/`values()` are the common iteration patterns |
| Registration dedup | (a) Extract `_register_with_aliases()`, (b) Leave as-is | (a) | Eliminates 4 duplicated lines in each registration function |
| `_create_and_run_study` parameter bundling | (a) Pass 7 individual params, (b) Bundle into `study_kwargs` dict | (b) | Keeps function signature manageable per spec |

## Scope

### In Scope

- Extract 5 private helpers from `optimize()` in `api.py`:
  `_infer_and_validate_keys`, `_setup_validation_data`, `_build_objective`,
  `_derive_directions`, `_create_and_run_study`
- Override `items()` and `values()` on `_TrackingDict` in `pipeline.py`
- Extract `_register_with_aliases()` in `registration.py`
- Tests for each change

### Out of Scope

- Public API changes
- Decision-needed items (timing semantics, validation_data optionality, validate_fn contract docs)
- `default_validate_fn` export (already done)

## Implementation Plan

### Phase 1: Extract helpers from `optimize()`

Decompose `optimize()` into 5 named private helpers plus the orchestrator.
This is the largest change and should be done first since it touches the
most code.

**Files to modify:**
- `src/bayesflow_hpo/api.py` — extract helpers, rewrite `optimize()` body
- `tests/test_api.py` — add unit tests for each extracted helper

**Steps:**

1. Add `_infer_and_validate_keys(adapter)` — move lines 313-333 (key
   inference via `infer_keys_from_adapter` + validation + condition
   fallback). Returns `tuple[list[str], list[str]]`.

2. Add `_setup_validation_data(simulator, validation_simulator, param_keys,
   data_keys, validation_conditions, sims_per_condition)` — move lines
   335-343. Internally calls `generate_validation_dataset(simulator=val_sim,
   param_keys=param_keys, data_keys=data_keys,
   condition_grid=validation_conditions,
   sims_per_condition=sims_per_condition)` — note `condition_grid=` is the
   kwarg name in `generate_validation_dataset`, mapped from `optimize()`'s
   `validation_conditions` parameter. Returns `ValidationDataset`.

3. Add `_build_objective(*, simulator, adapter, search_space, validation_data,
   epochs, num_batches, early_stopping_patience, early_stopping_window,
   max_param_count, max_memory_mb, n_posterior_samples, objective_metrics,
   objective_mode, cost_metric, report_frequency, build_approximator_fn,
   train_fn, validate_fn, checkpoint_pool)` — move lines 358-380.
   Constructs `ObjectiveConfig` + `GenericObjective`. Returns
   `GenericObjective`. Uses keyword-only arguments matching `ObjectiveConfig`
   fields.

4. Add `_derive_directions(objective, directions, objective_metrics,
   objective_mode, cost_metric)` — move lines 382-400. Returns
   `tuple[list[str], list[str]]` (directions, metric_names).

5. Add `_create_and_run_study(objective, study_kwargs, n_trials,
   max_total_trials, show_progress_bar)` — move lines 402-429. `study_kwargs`
   bundles `{study_name, directions, metric_names, storage, resume,
   warm_start_from, warm_start_top_k}`. The helper internally translates
   `resume` to `load_if_exists=resume or storage is None` before calling
   `create_study()` and handles the delete-if-needed logic. Returns `Study`.

6. Rewrite `optimize()` body as 6-step orchestrator calling helpers.

7. Add tests for each helper (note: existing `test_api.py` tests cover
   `_infer_and_validate_keys` indirectly via `_patched_optimize()` — new
   unit tests focus on direct helper invocation):
   - `test__infer_and_validate_keys_*` — success, missing param, missing data,
     condition fallback
   - `test__setup_validation_data_*` — uses validation_simulator, falls back
     to simulator
   - `test__build_objective_*` — constructs correct ObjectiveConfig
   - `test__derive_directions_*` — auto-derive, explicit, wrong length
   - `test__create_and_run_study_*` — delete-if-needed, resume, study creation

**Depends on:** None

### Phase 2: Fix `_TrackingDict` + deduplicate registration

Two small, independent fixes bundled together since each is too small for
a solo phase.

**Files to modify:**
- `src/bayesflow_hpo/pipeline.py` — add `items()` and `values()` overrides
  to `_TrackingDict`, update docstring
- `src/bayesflow_hpo/registration.py` — extract `_register_with_aliases()`
- `tests/test_pipeline.py` — add tests for `items()`/`values()` tracking
- `tests/test_registration.py` — add tests for `_register_with_aliases()`

**Steps:**

1. Add `items()` and `values()` overrides to `_TrackingDict`:
   ```python
   def items(self):
       self.accessed_keys.update(self.keys())
       return super().items()

   def values(self):
       self.accessed_keys.update(self.keys())
       return super().values()
   ```

2. Update `_TrackingDict` docstring to: "Tracks ``__getitem__``, ``get``,
   ``__contains__``, ``pop``, ``items()``, and ``values()``. Note:
   ``__iter__`` is intentionally not overridden because
   ``dict(tracking_dict)`` calls ``__iter__`` internally, which would
   falsely mark all keys as accessed."

3. Extract `_register_with_aliases()` helper in `registration.py`:
   ```python
   def _register_with_aliases(register_fn, name, builder, aliases, overwrite):
       register_fn(name=name, builder=builder, overwrite=overwrite)
       for alias in aliases or []:
           register_fn(name=alias, builder=builder, overwrite=overwrite)
   ```

4. Refactor `register_custom_inference_network` and
   `register_custom_summary_network` to use `_register_with_aliases()`.

5. Add tests:
   - `test_tracking_dict_items_marks_accessed` — verify `items()` marks keys
   - `test_tracking_dict_values_marks_accessed` — verify `values()` marks keys
   - `test_tracking_dict_dict_copy_does_not_mark` — verify `dict(td)` does NOT
     mark keys (regression guard)
   - `test_register_with_aliases_registers_all` — verify builder registered
     under name + aliases

**Depends on:** None (independent of Phase 1)

## Verification & Validation

- **Automated**: All existing tests must pass unchanged (behavioral
  equivalence). New tests cover each extracted helper and fix.
- **Manual**: Run `ruff check src/ tests/` — no new lint errors.
- **Regression**: Existing `test_api.py` tests exercise `optimize()` end-to-end
  via `_patched_optimize()` — these implicitly validate the refactored
  orchestrator.

## Dependencies

- No new dependencies required.

## Notes

_Living section — updated during implementation._

## Review Feedback

Reviewed in 1 iteration. 10 findings (1 blocker, 5 warnings, 4 suggestions).

**Blocker (resolved):** `generate_validation_dataset` uses `condition_grid=`
not `validation_conditions=` — clarified in Phase 1 Step 2.

**Warnings (addressed):**
- `_build_objective` signature clarified to use explicit keyword-only args
  matching `ObjectiveConfig` fields (Finding 2).
- `_create_and_run_study` must translate `resume` → `load_if_exists=resume or
  storage is None` internally — documented in Phase 1 Step 5 (Finding 4).
- `_TrackingDict` replacement docstring specified in Phase 2 Step 2 (Finding 5).
- `dict(td)` uses `__iter__` not `items()` — regression test guards this
  (Finding 6).
- Spec Requirement 1 says "4 helpers" but table shows 5 — cosmetic spec
  inconsistency, plan correctly uses 5 (Finding 1).

**Suggestions (noted):**
- Line range corrected to 313-333 for `_infer_and_validate_keys` (Finding 7).
- Existing tests provide indirect coverage; new tests focus on direct helper
  invocation (Finding 8).
- `_register_with_aliases` correctly uses leading underscore (Finding 9).
- Phase independence confirmed — Phase 2 can ship before Phase 1 (Finding 10).
