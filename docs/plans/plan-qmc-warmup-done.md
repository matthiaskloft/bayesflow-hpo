# Plan: QMC Warm-up (Package A3)

**Created**: 2026-03-23
**Author**: Claude Code + Matze

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Spec | DONE | 2026-03-23 | All 9 review findings addressed |
| Plan | DONE | 2026-03-23 | Reviewed in 1 iteration |
| Phase 1: QMC wrapper + wiring + tests | MERGED | 2026-03-26 | 34 tests, 617 total passing |
| Ship | MERGED | 2026-03-26 | PR #57 |

## Spec

### Summary

**Motivation**: The first N trials of any Optuna sampler are effectively
random (startup/warm-up phase). Replacing them with a Sobol
quasi-random sequence provides better space-filling coverage, giving
the main sampler (TPE, GP, BoTorch, etc.) a more informative initial
dataset to learn from. This is especially valuable in moderate-to-high
dimensional search spaces (5–30 dims), which are typical for BayesFlow
HPO.

**Outcome**: Users can set `qmc_startup_trials=16` on `optimize()` (or
`create_study()`) to run 16 Sobol-sequence trials before the main
sampler takes over. The feature composes with all existing sampler
presets, warm-start, and budget-aware sampling.

### Requirements

- `optimize()` accepts `qmc_startup_trials: int = 0`. When > 0, the
  first N non-rejected trials use Optuna's `QMCSampler` (Sobol).
- `create_study()` also accepts `qmc_startup_trials` so users who
  build studies manually can use it.
- The composite wrapper counts only non-rejected, QMC-sampled trials
  toward the N-trial quota. Budget-rejected trials do not count.
- If `qmc_startup_trials` is not a power of 2, log a warning
  explaining that Sobol's low-discrepancy guarantee is optimal at
  n = 2^m, but proceed with the user's exact value.
- Composes with `warm_start_from`: both run independently. Warm-started
  trials provide exploitation knowledge; QMC adds space-filling
  coverage. The wrapper counts only its own QMC-generated trials.
- Composes with all 7 sampler presets and custom `BaseSampler`
  instances.
- Budget-aware constraints (`_budget_constraints_func`) apply to QMC
  trials the same way they apply to any other trial.
- QMC coverage applies only to continuous and integer dimensions.
  Categorical dimensions (e.g., network type in `NetworkSelectionSpace`)
  are always sampled by QMCSampler's `independent_sampler` (RandomSampler
  by default). This is inherent to Optuna's `QMCSampler` and acceptable
  for our use case — the primary benefit is space-filling over the
  continuous hyperparameters.
- `_resolve_n_startup_trials()` must handle `QMCWarmupSampler`:
  the wrapper exposes `n_startup_trials` as
  `max(qmc_startup_trials, main_sampler_n_startup_trials)` so pruning
  warmup is correctly configured.
- `QMCWarmupSampler` delegates `before_trial()` and `after_trial()`
  hooks to the currently active sub-sampler. This is required for
  samplers like NSGA-II/III that maintain population state in
  `after_trial()`.
- `QMCWarmupSampler` is single-worker only (consistent with
  `optimize_until()` which always uses `n_jobs=1`). No thread-safety
  locks on internal state.
- `QMCWarmupSampler` is internal — not exported from `__init__.py`.
  The public API is the `qmc_startup_trials` parameter on
  `create_study()` and `optimize()`.

### Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Swap mechanism | (1) Composite wrapper sampler (2) Two-phase study.optimize() (3) Separate warm-start study | Composite wrapper | Clean, self-contained, no Optuna internals mutation. Wrapper delegates to QMCSampler for first N trials, then main sampler. |
| QMC sequence type | (1) Sobol only (2) Expose qmc_type param (3) Halton default | Sobol only | Optuna's own benchmarks (PR #2423) show Sobol "much better than Halton". Sobol superior in dims > 6 (typical HPO: 5–30 dims). No need to expose qmc_type — users can pass a custom sampler if they need Halton. |
| Power-of-2 handling | (1) Auto-round up (2) Warn but don't change (3) Silent | Warn but don't change | Sobol works at non-power-of-2 counts with slightly weaker guarantees. Warning educates without overriding user intent. |
| warm_start interaction | (1) Composable (2) Mutually exclusive (3) QMC replaces startup | Composable | Warm-started trials cluster near prior optima (exploitation). QMC provides orthogonal space-filling coverage (exploration). Both are valuable together. |
| Rejected trial counting | (1) Don't count rejected (2) Count all | Don't count rejected | Ensures user gets N actual space-filling data points. Consistent with optimize_until() counting semantics. |
| Switch signal | (1) Count non-rejected QMC-sampled trials (2) Count all study trials (3) Tag-based | Count non-rejected QMC-sampled | Wrapper tracks its own QMC-generated non-rejected completions. Most precise without per-trial overhead. |
| File location | (1) optimization/study.py (2) New qmc_sampler.py (3) Package root | optimization/study.py | Co-located with `_resolve_sampler()` and `create_study()`. All sampler logic in one place. |

### Scope

#### In Scope

- `QMCWarmupSampler` composite wrapper in `optimization/study.py`
- `qmc_startup_trials` parameter on `create_study()` and `optimize()`
- Power-of-2 warning
- Tests for the wrapper, integration with budget rejection, warm-start
  composition, and all sampler presets
- Docstring updates for both functions

#### Out of Scope

- Exposing `qmc_type` parameter (Sobol only)
- Exposing `scramble` parameter (False — Optuna's default for
  distributed safety; scrambling requires a manually shared seed
  across workers to preserve low-discrepancy properties)
- Research on QMC warm-up effectiveness (separate TODO item in A3)
- Halton support
- Auto-selecting `qmc_startup_trials` based on search space dimensionality

### Architecture Overview

```
optimize(qmc_startup_trials=16, sampler="tpe")
    │
    ▼
_create_and_run_study()
    │
    ▼
create_study(qmc_startup_trials=16, sampler="tpe")
    │
    ├─ _resolve_sampler("tpe") → TPESampler
    │
    ├─ if qmc_startup_trials > 0:
    │      main_sampler = TPESampler
    │      sampler = QMCWarmupSampler(main_sampler, qmc_startup_trials)
    │
    ▼
optuna.create_study(sampler=QMCWarmupSampler)
    │
    ▼
optimize_until() → study.optimize()
    │
    ▼
QMCWarmupSampler.sample_relative() / sample_independent()
    │
    ├─ if n_qmc_completed < qmc_startup_trials:
    │      delegate to QMCSampler
    │
    └─ else:
           delegate to main_sampler
```

**`QMCWarmupSampler`** extends `optuna.samplers.BaseSampler`:
- Wraps a `QMCSampler` (Sobol, scramble=False, no seed — Sobol
  without scrambling is inherently deterministic) and a main sampler
  (user-chosen)
- Tracks how many non-rejected trials it generated during the QMC
  phase via an internal counter, incremented when a trial completes
  without `rejected_reason` in its `user_attrs`
- Delegates `sample_relative()`, `sample_independent()`,
  `infer_relative_search_space()`, `before_trial()`, and
  `after_trial()` to the currently active sub-sampler
- After the QMC phase, the wrapper is transparent — all calls go to
  the main sampler
- Exposes `n_startup_trials` property returning
  `max(qmc_startup_trials, _resolve_n_startup_trials(main_sampler))`
  so `_resolve_n_startup_trials()` works correctly when called on
  the wrapper
- Single-worker only — no thread-safety locks (consistent with
  `optimize_until()` which uses `n_jobs=1`)

**Counting mechanism**: The wrapper tracks pending QMC trials in a
`set[int]` of trial numbers (added during `sample_relative()`/
`sample_independent()` when delegating to QMC). In `after_trial()`,
the wrapper checks if the trial completed without rejection — if so,
it increments an `_n_qmc_completed` counter. This avoids O(N)
`study.trials` scans on every sample call and is correct for
single-worker use.

### Constraints

- Must work with Optuna >= 4.0.0 (our minimum)
- `QMCSampler` has been stable since Optuna 3.0.0 (still marked
  experimental but API hasn't changed)
- No new dependencies
- Must not break existing `sampler` parameter behavior when
  `qmc_startup_trials=0` (default — no wrapper, same as today)
- The wrapper must be transparent to Optuna's internal mechanisms
  (pruning, constraints_func, etc.)

### Open Questions

_None — all questions resolved during spec review._

## Implementation Plan

### Phase 1: QMC wrapper + wiring + tests

Single phase — the wrapper, API wiring, and tests are tightly coupled
and must ship together.

**Files to create:**
- `tests/test_optimization/test_qmc_warmup.py` — all QMCWarmupSampler tests

**Files to modify:**
- `src/bayesflow_hpo/optimization/study.py` — add `QMCWarmupSampler`
  class, `_is_power_of_two()` helper, update `create_study()` signature
  and body
- `src/bayesflow_hpo/api.py` — add `qmc_startup_trials` parameter to
  `optimize()` and `_create_and_run_study()`, forward to `create_study()`
- `docs/api_reference.md` — add `qmc_startup_trials` to `create_study()`
  signature
- `docs/defaults.md` — add QMC default row
- `docs/optimization.md` — add QMC warm-up usage example

**Steps:**

1. **Add `QMCWarmupSampler` to `optimization/study.py`**:
   - Class extending `optuna.samplers.BaseSampler`
   - `__init__(self, main_sampler, qmc_startup_trials)`:
     creates internal `QMCSampler(qmc_type="sobol", scramble=False)`
     (no `seed` — Sobol without scrambling is inherently deterministic),
     stores `main_sampler` and `qmc_startup_trials`
   - Validate: `if qmc_startup_trials < 0: raise ValueError(...)`
   - **Counting via `after_trial()`**: track pending QMC trial numbers
     in `_pending_qmc_trials: set[int]` (added in `sample_relative()`/
     `sample_independent()` when delegating to QMC). In `after_trial()`,
     check if trial is COMPLETE and non-rejected → increment
     `_n_qmc_completed: int` counter and remove from pending. If
     rejected, just remove from pending. This avoids O(N) `study.trials`
     scans on every sample call.
   - `_is_qmc_phase` property: returns
     `self._n_qmc_completed < self._qmc_startup_trials`
   - `_active_sampler` property: returns `_qmc_sampler` if
     `_is_qmc_phase`, else `_main_sampler`
   - Delegate methods: `infer_relative_search_space()`,
     `sample_relative()`, `sample_independent()`, `before_trial()`,
     `after_trial()` — all delegate to `_active_sampler` (passing
     through all arguments). Note: `before_trial()` is called before
     `sample_relative()`, so the current trial's number is not yet
     in `_pending_qmc_trials` — this is correct because the
     `_active_sampler` decision is based on prior completed trials.
   - `n_startup_trials` property: returns
     `max(self._qmc_startup_trials,
     _resolve_n_startup_trials(self._main_sampler))` — must pass
     `self._main_sampler`, NOT `self`, to avoid recursion
   - Add `_is_power_of_two(n: int) -> bool` module-level helper

2. **Update `create_study()` in `study.py`**:
   - Add `qmc_startup_trials: int = 0` parameter
   - After sampler resolution (line ~346), if `qmc_startup_trials > 0`:
     - Log warning if not power of 2
     - Wrap: `sampler = QMCWarmupSampler(sampler, qmc_startup_trials)`
   - Update docstring with `qmc_startup_trials` parameter docs
   - Validate: `if qmc_startup_trials < 0: raise ValueError(...)`

3. **Wire through `api.py`**:
   - Add `qmc_startup_trials: int = 0` to `optimize()` signature
     (after `warm_start_top_k`)
   - Add `qmc_startup_trials: int = 0` to `_create_and_run_study()`
     signature
   - Forward to `create_study()` call in `_create_and_run_study()`
   - Update `optimize()` and `_create_and_run_study()` docstrings

4. **Update docs**:
   - `docs/api_reference.md`: add `qmc_startup_trials` to
     `create_study()` signature and `optimize()` parameter table
   - `docs/defaults.md`: add row for QMC startup default (0)
   - `docs/optimization.md`: add QMC warm-up usage example; also fix
     pre-existing bug (non-existent `warm_start_metric_index` param
     in warm-start example at line ~219)

5. **Write tests in `tests/test_optimization/test_qmc_warmup.py`**:
   - `TestQMCWarmupSampler`:
     - `test_delegates_to_qmc_during_warmup` — verify QMCSampler is
       used for first N trials
     - `test_switches_to_main_after_warmup` — verify main sampler
       takes over
     - `test_non_rejected_counting` — budget-rejected trials don't
       count toward QMC quota
     - `test_before_trial_delegates_to_active` — hooks go to correct
       sub-sampler
     - `test_after_trial_delegates_to_active` — hooks go to correct
       sub-sampler
     - `test_n_startup_trials_property` — returns max of QMC and
       main sampler startup
     - `test_infer_relative_search_space_delegates`
   - `TestCreateStudyQMC`:
     - `test_qmc_zero_no_wrapper` — default (0) produces no wrapper,
       existing sampler type preserved
     - `test_qmc_positive_wraps_sampler` — `qmc_startup_trials=8`
       produces `QMCWarmupSampler`
     - `test_qmc_with_string_preset` — composes with "tpe", "gp",
       "nsga2" etc.
     - `test_qmc_power_of_two_warning` — logs warning for non-power-of-2
     - `test_qmc_no_warning_for_power_of_two` — no warning for 8, 16, 32
     - `test_qmc_with_custom_sampler` — wraps custom BaseSampler
     - `test_qmc_negative_raises` — `qmc_startup_trials=-1` raises
       ValueError
   - `TestResolveNStartupTrialsQMC`:
     - `test_qmc_wrapper_returns_n_startup_trials` — verify property
       works with `_resolve_n_startup_trials()`

**Depends on:** None

## Verification & Validation

- **Automated**: Unit tests for `QMCWarmupSampler` (delegation logic,
  counting, switch point, `before_trial`/`after_trial` delegation,
  `n_startup_trials` property), integration tests with budget rejection,
  warm-start composition, all sampler presets. Existing sampler type
  assertions in `test_study.py` must pass unchanged when
  `qmc_startup_trials=0` (default). New tests for `qmc_startup_trials > 0`
  assert `isinstance(study.sampler, QMCWarmupSampler)`.
- **Manual**: Run a small study with `qmc_startup_trials=8` and verify
  via trial user_attrs or logs that the first 8 non-rejected trials
  used QMC sampling

## Dependencies

- `optuna.samplers.QMCSampler` (available since Optuna 3.0.0, our
  minimum is 4.0.0)

## Notes

_Living section — updated during implementation._

### References

- Optuna QMCSampler: `optuna.samplers.QMCSampler` — uses Scipy's
  `scipy.stats.qmc` implementations
- Optuna PR #2423 (kstoneriv3): Sobol "performed much better than
  Halton" in benchmarks; Sobol chosen as default `qmc_type`
- `scramble=False` is Optuna's default for distributed safety —
  scrambling requires a manually shared seed across workers to preserve
  low-discrepancy properties (Optuna `_qmc.py` source, lines 155, 176–178)
- Optuna Issue #1797: Original SobolSampler proposal; benchmarks showed
  QMCSampler statistically significantly better than RandomSampler
- Sobol' sequence optimal at n = 2^m samples (Optuna docs, Scipy docs)
- Halton works best for d ≤ ~6; Sobol superior in higher dimensions
  (search results from Kucherenko et al.; web search consensus)

## Review Feedback

### Spec Review (2026-03-23, code-architect agent)

9 findings (2 blockers, 4 warnings, 3 suggestions). All addressed:

**Blockers (resolved):**
1. `_resolve_n_startup_trials` returns fallback 10 when wrapper is
   active → resolved by adding `n_startup_trials` property to
   `QMCWarmupSampler` returning
   `max(qmc_startup_trials, main_sampler_n_startup)`.
2. Fix to `_resolve_n_startup_trials` missing from implementation
   checklist → added to requirements; will be in Phase 1 checklist.

**Warnings (resolved):**
3. Categorical dims fall through to `independent_sampler` → documented
   in requirements as inherent QMCSampler behavior, acceptable.
4. Mutable wrapper state not thread-safe → documented as single-worker
   only (consistent with `optimize_until()` using `n_jobs=1`).
5. `before_trial`/`after_trial` delegation not specified → added to
   component design; required for NSGA-II/III population management.
6. Existing sampler type assertions break → added to test plan;
   `qmc_startup_trials=0` preserves current behavior.

**Suggestions (resolved):**
7. `scramble=False` rationale cited wrong source → corrected to
   distributed safety reason from Optuna source.
8. `QMCWarmupSampler` export decision not stated → decided internal,
   documented in requirements.
9. `_create_and_run_study` parameter wiring not explicit → will be
   explicit in Phase 1 checklist.

### Plan Review (2026-03-23, code-architect agent)

7 findings (2 blockers, 4 warnings, 3 suggestions). Plan reviewed in
1 iteration — all findings addressed in-place:

**Blockers (resolved):**
1. `seed` parameter has no effect when `scramble=False` → removed
   `seed` from constructor; Sobol without scrambling is deterministic.
2. `_create_and_run_study()` docstring not in checklist → added.

**Warnings (addressed):**
3. `n_startup_trials` recursion risk → implementation must pass
   `self._main_sampler` explicitly (documented in step 1).
4. O(N) `study.trials` scan on every sample call → switched to
   `after_trial()` counter approach with `_pending_qmc_trials` set.
5. `before_trial()` ordering semantics → documented in step 1 that
   `before_trial()` is called before trial number is recorded.
6. Existing `test_study.py` tests unaffected → confirmed, default
   `qmc_startup_trials=0` means no wrapper.

**Suggestions (addressed):**
7. Pre-existing `warm_start_metric_index` doc bug → will fix in step 4.
8. Add `qmc_startup_trials` to both API ref locations → added to step 4.
9. Negative value validation → added `ValueError` guard to steps 1–2
   and test.
