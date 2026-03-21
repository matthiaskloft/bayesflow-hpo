# Plan: Pruning Review & Refactor (Package A1)

**Created**: 2026-03-21
**Author**: Claude
**Spec**: [spec-pruning-review-refactor.md](spec-pruning-review-refactor.md)

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-21 | |
| Phase 1: Strategy functions + unit tests | IMPLEMENTED | 2026-03-22 | 41 tests, lint clean, review findings addressed |
| Phase 2: Callback refactor + metric alignment | IMPLEMENTED | 2026-03-22 | 11 tests, lint clean, review in progress |
| Phase 3: API wiring + startup auto-detect | IMPLEMENTED | 2026-03-22 | 12 new tests, lint clean |
| Phase 4: Pruner string presets in create_study() | IMPLEMENTED | 2026-03-22 | 10 tests, 3 new refs, lint clean |
| Ship | DONE | 2026-03-22 | All 4 phases shipped |

## Summary

**Motivation**: The current multi-objective pruning implementation has four
correctness/design issues: (1) hard-coded metrics disconnected from
`objective_metrics`, (2) scale-sensitive geometric mean composite with no
normalization, (3) `n_startup_trials=5` misaligned with TPE's 25, and
(4) no way to disable or customize the strategy. Optuna has no built-in
multi-objective pruning (Issue #3450, open since April 2022), so we must
maintain our own — but it should be correct and configurable.

**Outcome**: Users select a pruning strategy via `pruning_strategy=` in
`optimize()`. Four strategies available: `"none"`, `"dominance"` (default),
`"mo-sha"`, `"primary"`. Intermediate validation auto-aligns with
`objective_metrics`. Startup trials auto-detected from the sampler.
Single-objective studies gain pruner string presets in `create_study()`.

## Assumptions

- Optuna will not add multi-objective `trial.report()` before this ships
  (Issue #3450 has been open since April 2022 with no merged PR)
- Non-dominated sorting can be implemented in pure numpy without pymoo
  (confirmed: standard O(M*N^2) algorithm, ~30 lines)
- The `validate_fn` contract change (must return all `objective_metrics`)
  is acceptable — current callers hardcode `calibration_error`/`nrmse`

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Strategy architecture | One parameterized strategy vs. named strategies vs. ABC/protocol | Named strategies (4 functions) | Discoverable, documentable, each maps to a literature reference |
| Default strategy | `"dominance"` vs. `"mo-sha"` vs. `"none"` | `"dominance"` | Conservative (AND rule), works with few trials, easy to reason about |
| Primary metric syntax | String encoding vs. separate param vs. tuple | Tuple: `("primary", "metric")` | Python-native, type-checkable, visually ties metric to strategy |
| Intermediate metrics | Hard-coded vs. auto-align vs. configurable | Auto-align with `objective_metrics` | Hard-coded was a design bug; `validate_fn` covers custom subsets |
| Startup detection | Fixed 10 vs. match sampler vs. keep 5 | Auto-detect from `sampler.n_startup_trials`, fallback 10 | Pruning against random-phase trials is unreliable |
| Pruner presets scope | `optimize()` + `create_study()` vs. `create_study()` only | `create_study()` only | `optimize()` doesn't forward `pruner`; users call `create_study()` directly |

Full rationale for each decision is in the
[spec](spec-pruning-review-refactor.md#design-decisions).

## Scope

### In Scope

- New `optimization/pruning_strategies.py` with 3 strategy functions
- Refactored `PeriodicValidationCallback` with strategy dispatch
- `pruning_strategy` parameter on `optimize()` and `ObjectiveConfig`
- Auto-detect `n_startup_trials` from sampler
- Remove hard-coded `_INTERMEDIATE_METRICS`; use `objective_metrics`
- Pruner string presets (`"median"`, `"hyperband"`, `"none"`) in `create_study()`
- Per-metric user attribute schema (migration from `val_score_step_*`)
- Update `docs/references.md` with MO-ASHA, Hyperband, and related citations

### Out of Scope

- Strategy protocol/ABC for user-defined strategies
- Configurable quantile threshold for `"dominance"`
- Changes to `MovingAverageEarlyStopping`
- Sampler presets (Package A2)
- QMC warm-up (Package A3)

## Implementation Plan

### Phase 1: Strategy functions + unit tests

Create the new `pruning_strategies.py` module with all three strategy
functions and comprehensive unit tests. This phase is purely additive —
nothing calls these functions yet.

**Files to create:**
- `src/bayesflow_hpo/optimization/pruning_strategies.py` — strategy
  implementations
- `tests/test_optimization/test_pruning_strategies.py` — unit tests

**Files to modify:**
- None

**Steps:**

1. **Write module docstring for `pruning_strategies.py`**: Document the
   four-strategy architecture and its literature motivation. Cite:
   - Schmucker et al. (2021) for MO-ASHA as the primary reference for
     multi-objective pruning in multi-fidelity HPO (read in full)
   - Emmerich & Deutz (2018) for Pareto dominance fundamentals
     (Definition 5) and scalarization limitations (Proposition 9)
     (read in full)
   - Deb et al. (2002) for non-dominated sorting (already in references.md)

2. **Implement `should_prune_dominance()`**: Per-objective median check
   with range normalization (AND rule). Gathers per-metric user attrs
   (`val_{metric}_step_{N}`) from completed non-rejected trials.
   Normalizes each metric to [0, 1] using observed range to eliminate
   scale sensitivity — Schmucker et al. (2021, Section 6) found that
   scalarization "tends to penalize one objective heavier than the other"
   while "globally informed techniques are more robust towards objectives
   of different magnitude." Prunes only if the trial is worse than the
   median on ALL objectives. Handles NaN/Inf (immediate prune), missing
   attrs (skip trial), and degenerate ranges (single-value → skip
   normalization).
   **Docstring must cite**: Schmucker et al. (2021) — simplified
   adaptation of MO-ASHA's dominance-based promotion (Algorithm 1).
   Scale sensitivity of scalarization: Emmerich & Deutz (2018,
   Proposition 9) for theoretical limitation; Schmucker et al. (2021,
   Section 6) for empirical confirmation.

3. **Implement `should_prune_mo_sha()`**: Non-dominated sorting at each
   step. Gathers all completed trial score vectors at the current step.
   Uses `_non_dominated_sort()` for NSGA-II-style ranking. At each rung,
   selects top `|rung| / η` configurations (MO-ASHA Algorithm 2, line 11:
   `mo_selector(rung k, |rung k| / η)`). Prunes if the current trial
   is NOT in the selected set. η=3 is the default in MO-ASHA (Algorithm
   2 header: "Data: R, r0, s, η (default η = 3)") and is recommended
   by Li et al. (2018, Section 3.6: "in practice we suggest taking η to
   be equal to 3 or 4"). Same NaN/Inf/missing-attr handling as dominance.
   **Docstring must cite**: Schmucker et al. (2021) Algorithms 1–2 for
   the multi-objective selector and MO-ASHA procedure; Li et al. (2018,
   Section 3.6) for the η=3 convention.

4. **Implement `should_prune_primary()`**: Single-metric median comparison.
   Identical logic to the current `_should_prune_multi_objective()` but
   reads `val_{metric}_step_{N}` instead of `val_score_step_{N}`.
   **Docstring must cite**: Akiba et al. (2019) — equivalent to Optuna's
   MedianPruner applied to a single user-chosen objective.

5. **Implement `_non_dominated_sort()`**: Pure numpy helper for step 3.
   Input: (N, M) array of objective values. Output: list of front indices.
   Matches MO-ASHA Algorithm 1 lines 1–6: iteratively extract
   non-dominated fronts F1, ..., Fm by removing dominated points.
   **Docstring must cite**: Deb et al. (2002) for the O(MN²) fast
   non-dominated sorting algorithm (NSGA-II); confirmed via Schmucker
   et al. (2021) Algorithm 1 `non_dom_sorting(P)` pseudocode.

6. **Write tests** (~25-30 tests across all strategies):
   - Startup threshold (no pruning below `n_startup_trials`)
   - At-median boundary (no pruning for equal)
   - Above-median pruning
   - NaN/Inf handling (immediate prune)
   - NaN in reference scores (filtered out)
   - Budget-rejected trials excluded
   - Step independence (step 1 scores don't affect step 2)
   - `n_startup_trials=0` disables pruning
   - Dominance: scale normalization works (metrics on different scales)
   - Dominance: AND rule (bad on one metric but good on another → no prune)
   - MO-SHA: non-dominated sorting correctness
   - MO-SHA: bottom-fraction pruning with η=3
   - MO-SHA: small reference sets (fewer than η trials)
   - Primary: single-metric pruning matches expected behavior
   - Missing per-metric attrs gracefully skipped
   - Cross-schema migration: trials with old `val_score_step_*` format
     are silently skipped (strategy returns False, conservative)

**Depends on:** None

### Phase 2: Callback refactor + metric alignment

Refactor `PeriodicValidationCallback` to accept a strategy name and
`objective_metrics`, replace the hard-coded metrics and geometric mean,
and dispatch to the strategy functions from Phase 1. Update existing
tests to match the new interface.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/optimization/validation_callback.py` — major refactor
- `tests/test_optimization/test_multi_objective_pruning.py` — update tests

**Steps:**

1. **Update `PeriodicValidationCallback.__init__()`**: Add parameters
   `pruning_strategy: str | tuple[str, str] = "dominance"` and
   `objective_metrics: list[str] | None = None` (defaults to
   `["calibration_error", "nrmse"]` for backwards compatibility with
   callers that don't yet pass the new param — the default is removed
   in Phase 3 once all callers are updated). Remove the
   `_INTERMEDIATE_METRICS` module constant. Parse tuple strategy syntax
   to extract `_strategy_name` and `_primary_metric`. Add validation:
   reject unknown strategy names, validate `"primary"` has a valid
   metric key (default to `objective_metrics[0]` if bare string).

2. **Refactor `_run_lightweight_validation()`**: Return
   `dict[str, float] | None` instead of `float | None`. When using the
   default path, pass `metrics=self.objective_metrics` to
   `run_validation_pipeline()` instead of `_INTERMEDIATE_METRICS`. When
   using `validate_fn`, validate that all `objective_metrics` keys exist
   in the returned dict — log a warning and return `None` if any are
   missing.

3. **Refactor `on_epoch_end()`**: Store per-metric user attrs
   (`val_{metric}_step_{N}` for each metric in the result dict) instead
   of the single composite `val_score_step_{N}`. Dispatch to the
   appropriate strategy function based on `self.pruning_strategy`:
   - `"dominance"` → `should_prune_dominance(trial, scores, step, n_startup)`
   - `"mo-sha"` → `should_prune_mo_sha(trial, scores, step, n_startup)`
   - `"primary"` → extract `scores[self._primary_metric]` as scalar,
     then `should_prune_primary(trial, primary_score, step, n_startup)`
   Single-objective path remains unchanged (`trial.report()` +
   `trial.should_prune()`), but uses `scores[objective_metrics[0]]`
   instead of the geometric mean composite.

4. **Update `validation_callback.py` module docstring**: Replace the
   geometric mean / median-based description with the pluggable strategy
   architecture. **Must cite**: Schmucker et al. (2021) for MO-ASHA
   (read in full); note that Optuna does not support `trial.report()`
   for multi-objective studies (Issue #3450, verified from Optuna source).
   Reference the four strategies and where their implementations live
   (`pruning_strategies.py`).

5. **Migrate unit tests for `_should_prune_multi_objective`**: The eight
   unit tests in `test_multi_objective_pruning.py` that call `_PRUNE()`
   directly (lines 56–152) test the median-comparison logic that now
   lives in `should_prune_dominance()` and `should_prune_primary()`.
   Move these tests to `test_pruning_strategies.py` (created in Phase 1)
   by rewriting them to call the new strategy functions. Remove the
   now-dead `_should_prune_multi_objective` from `validation_callback.py`
   and the old import from the test file.

6. **Update callback integration tests** in `test_multi_objective_pruning.py`:
   - Update `test_callback_stores_step_keyed_attr` to check for
     per-metric attrs (e.g., `val_calibration_error_step_1`) instead of
     `val_score_step_1`. Update the mock return value from scalar `0.01`
     to `{"calibration_error": 0.01, "nrmse": 0.02}`. Pass
     `objective_metrics=["calibration_error", "nrmse"]` to the callback
     constructor.
   - Update `test_callback_raises_trial_pruned`: same mock return value
     change to `dict[str, float]`. Pass `pruning_strategy="dominance"`
     and `objective_metrics` to the constructor.
   - Update `test_single_objective_uses_trial_report`: mock returns
     `{"calibration_error": 0.5}`, callback uses `objective_metrics[0]`
     for `trial.report()`.
   - Add tests for strategy dispatch (each strategy function is called
     with correct arguments based on `pruning_strategy`)
   - Add test for `validate_fn` returning missing metrics → warning

**Depends on:** Phase 1 (strategy functions must exist)

### Phase 3: API wiring + startup auto-detect

Wire `pruning_strategy` through `optimize()` → `ObjectiveConfig` →
`PeriodicValidationCallback`. Implement sampler startup auto-detection.
Skip callback creation for `"none"`.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/api.py` — add `pruning_strategy` parameter
- `src/bayesflow_hpo/optimization/objective.py` — wire to callback;
  auto-detect startup; skip callback for `"none"`
- `src/bayesflow_hpo/__init__.py` — re-export if needed
- `tests/test_optimization/test_multi_objective_pruning.py` — add
  integration tests

**Steps:**

1. **Add `pruning_strategy` to `optimize()`**: New parameter
   `pruning_strategy: str | tuple[str, str] = "dominance"` in the
   "Training" parameter group (after `early_stopping_window`). Add
   validation: reject unknown strategy names, validate tuple structure.
   Pass through to `_build_objective()`.

2. **Add `pruning_strategy` to `ObjectiveConfig`**: New field
   `pruning_strategy: str | tuple[str, str] = "dominance"`. Add
   validation in `__post_init__()`: check strategy name is one of
   `{"none", "dominance", "mo-sha", "primary"}`. For tuple form,
   validate first element is `"primary"` and second is a string.

3. **Wire in `GenericObjective.__call__()`** (objective.py ~573-589):
   - When `pruning_strategy == "none"`: skip `PeriodicValidationCallback`
     entirely (don't append to callbacks list)
   - Otherwise: pass `pruning_strategy` and `objective_metrics` to the
     callback constructor

4. **Implement startup auto-detection**: Perform in `optimize()` after
   the study is created (in `_create_and_run_study()`) but before
   `ObjectiveConfig` is constructed. This avoids per-trial overhead
   and ensures the config always holds a resolved `int`.
   - Change `ObjectiveConfig.pruning_n_startup_trials` default from `5`
     to `None` (sentinel: `int | None = None`).
   - In `optimize()` / `_build_objective()`, if
     `pruning_n_startup_trials is None`, resolve it:
     `getattr(study.sampler, "n_startup_trials", 10)`.
   - Pass the resolved `int` to `ObjectiveConfig`. The callback always
     receives a concrete `int` — never `None`.
   - `ObjectiveConfig.__post_init__()` validates that if
     `pruning_n_startup_trials` is not `None`, it is `>= 0`.

5. **Update docstrings**:
   - `optimize()`: Document the `pruning_strategy` parameter with all
     four valid values, tuple syntax for `"primary"`, and the default
     `"dominance"`. **Must cite**: Schmucker et al. (2021) for
     `"mo-sha"` (read in full); note Optuna Issue #3450 context for
     why custom MO pruning is needed.
   - `ObjectiveConfig`: Document `pruning_strategy` (same as above) and
     updated `pruning_n_startup_trials` behavior (`None` → auto-detect
     from sampler, fallback 10).

6. **Add integration tests**:
   - `pruning_strategy="none"` → no `PeriodicValidationCallback` in
     callback list (mock `_build_objective` or inspect callbacks)
   - `pruning_strategy="dominance"` → callback created with correct
     strategy
   - `pruning_strategy=("primary", "calibration_error")` → callback
     created with correct primary metric
   - Invalid strategy name → `ValueError`
   - Invalid tuple form → `ValueError`
   - Startup auto-detection from TPE sampler → `n_startup_trials=25`
   - Startup auto-detection fallback → `n_startup_trials=10`
   - Explicit `pruning_n_startup_trials` overrides auto-detection

**Depends on:** Phase 2 (callback must accept new parameters)

### Phase 4: Pruner string presets in create_study()

Add string-based pruner selection to `create_study()` for single-objective
studies. Independent from the multi-objective pruning changes.

**Files to create:**
- `tests/test_optimization/test_study.py` — pruner preset tests

**Files to modify:**
- `src/bayesflow_hpo/optimization/study.py` — string preset resolution

**Steps:**

1. **Update `create_study()` signature**: Change
   `pruner: Any | None = None` to
   `pruner: str | optuna.pruners.BasePruner | None = None`. Add a
   `_resolve_pruner()` helper that maps strings to pruner instances:
   - `"median"` → `MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)`
   - `"hyperband"` → `HyperbandPruner(min_resource=1, reduction_factor=3)`
   - `"none"` → `NopPruner()`
   `_resolve_pruner()` is only called when `isinstance(pruner, str)`.
   The `None` path remains a separate branch in `create_study()` that
   creates the default `MedianPruner` (existing behavior, unchanged).
   `BasePruner` instances are passed through to Optuna directly.

2. **Update `create_study()` docstring**: Document the three string
   presets in a table within the `pruner` parameter docstring. Note that
   `"hyperband"` outperforms `"median"` with TPE per Optuna benchmarks.
   **Docstring must cite**: Li et al. (2018, Section 3.6) for Hyperband
   and η=3 convention (read in full); Akiba et al. (2019) for
   MedianPruner (already in references.md).

3. **Write tests** (~6-8 tests):
   - `pruner="median"` creates `MedianPruner`
   - `pruner="hyperband"` creates `HyperbandPruner`
   - `pruner="none"` creates `NopPruner`
   - `pruner=None` creates default `MedianPruner` (existing behavior)
   - Custom `BasePruner` object passed through unchanged
   - Invalid string → `ValueError`
   - Verify default behavior unchanged (no regression)

4. **Update `docs/references.md`**: Add only references that have been
   read in full text. Check against existing entries — Deb et al. (2002)
   and Akiba et al. (2019) are already present.

   **New entries to add to "All References" section:**

   - Schmucker, R., Donini, M., Zafar, M. B., Salinas, D., & Archambeau,
     C. (2021). Multi-objective asynchronous successive halving. *arXiv
     preprint*. https://doi.org/10.48550/arxiv.2106.12639

   - Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar,
     A. (2018). Hyperband: A novel bandit-based approach to hyperparameter
     optimization. *Journal of Machine Learning Research*, *18*(185), 1–52.

   - Emmerich, M. T. M., & Deutz, A. H. (2018). A tutorial on
     multiobjective optimization: Fundamentals and evolutionary methods.
     *Natural Computing*, *17*(3), 585–609.
     https://doi.org/10.1007/s11047-018-9685-y

   **New summaries to add (based on full-text reading):**

   - **Schmucker et al. (2021) — MO-ASHA**: Extends ASHA to
     multi-objective settings. Proposes two families of candidate
     selection: scalarization-based (RW, ParEGO, Golovin) and
     dominance-based (NSGA-II, EpsNet). Algorithm 1 defines
     `non_dom_sorting` + selectors; Algorithm 2 defines async MO-ASHA
     with default η=3 and promotion rule `mo_selector(rung k,
     |rung k|/η)`. Key finding: dominance-based approaches "consistently
     outperform multi-fidelity HPO based on MO scalarization" —
     scalarization tends to focus on one objective and can be worse than
     random search. Evaluated on NAS-201, Adult fairness, Wikitext2.
   - **Li et al. (2018) — Hyperband**: Proposes Hyperband, combining
     random search with adaptive resource allocation via Successive
     Halving. Algorithm 1: outer loop over brackets s=smax..0, inner SHA
     with ni=⌊n·η^(-i)⌋ survivors per round. Default η=3 (Section 3.6:
     "in practice we suggest taking η to be equal to 3 or 4"; theoretical
     optimum η=e≈2.718). smax=⌊log_η(R)⌋ brackets, B=(smax+1)·R budget.
     Over 20× faster than random search on deep learning benchmarks.
     Section 6 suggests quasi-random (Sobol) sampling as a promising
     extension — directly supports Package A3's QMC warm-up.
   - **Emmerich & Deutz (2018) — MOO tutorial**: Tutorial on
     multiobjective optimization fundamentals. Definition 5: formal
     Pareto dominance. Equations 3–4: non-dominated sorting as
     recursive extraction of non-dominated layers. Proposition 7:
     Θ(n²) complexity for finding non-dominated elements. Proposition
     9: linear scalarization can only find solutions on convex Pareto
     fronts — non-convex regions are unreachable. Covers NSGA-II
     (Section 6.1), indicator-based (SMS-EMOA), and decomposition-
     based (MOEA/D) approaches.

   **New index entries to add:**

   Under "### Pruning Strategies" (new section):
   - **MO-ASHA**: Schmucker et al. (2021)
   - **Hyperband / Successive Halving**: Li et al. (2018)
   - **MedianPruner**: Akiba et al. (2019) *(already indexed under Optuna)*

   Under "### Multi-Objective Optimization" (new section):
   - **Tutorial/Fundamentals**: Emmerich & Deutz (2018)
   - **NSGA-II**: Deb et al. (2002) *(already indexed under Samplers)*

**Depends on:** None (independent from Phases 1-3; can be done in parallel
or any order)

## Verification & Validation

- **Automated**:
  - All new strategy functions have unit tests with edge cases
    (NaN/Inf, missing attrs, boundary conditions, scale sensitivity)
  - Callback refactor tests verify strategy dispatch and per-metric
    attr storage
  - Integration tests verify end-to-end wiring from `optimize()` through
    to strategy functions
  - Pruner preset tests verify string→object resolution
  - Existing test suite passes without regression (233+ tests)
  - Ruff lint clean on `src/` and `tests/`

- **Literature compliance** (per CLAUDE.md source-backed mandate):
  - Only papers read in full text are cited. Read papers:
    Schmucker et al. (2021) MO-ASHA, Li et al. (2018) Hyperband,
    Emmerich & Deutz (2018) MOO tutorial.
    Already in references.md: Deb et al. (2002) NSGA-II, Akiba et al.
    (2019) Optuna.
  - Every strategy function docstring cites its primary reference
  - `_non_dominated_sort()` cites Deb et al. (2002), confirmed via
    MO-ASHA Algorithm 1 lines 1–6
  - `should_prune_mo_sha()` matches MO-ASHA Algorithm 2 structure:
    `mo_selector(rung k, |rung k|/η)` with default η=3
  - `create_study()` pruner docstring cites Li et al. (2018, Section 3.6)
  - `docs/references.md` contains 3 new references (Schmucker, Li,
    Emmerich & Deutz) with APA 7 format, full-text-based summaries,
    and index entries under new "Pruning Strategies" and
    "Multi-Objective Optimization" sections
  - No implementation detail relies on unverified LLM knowledge — all
    algorithms verified against paper pseudocode

- **Manual**:
  - Run the `getting_started.ipynb` example to verify no regression
  - Verify that `pruning_strategy="none"` results in no
    `PeriodicValidationCallback` in the callbacks list (inspect via
    debug logging or test assertion — no intermediate validation
    log messages is a secondary check)
  - Verify that resumed studies with old `val_score_step_*` attrs
    don't crash (graceful skip)

## Dependencies

- No new external dependencies
- Non-dominated sorting implemented in pure numpy
- Optuna >= 3.0 (already required; `NopPruner` and `HyperbandPruner`
  exist in all supported versions)

## Notes

_Living section — updated during implementation._

## Review Feedback

Reviewed in 1 iteration. 10 findings (2 blockers, 5 warnings, 3 suggestions).

**Blockers addressed:**
1. Eight unit tests importing `_should_prune_multi_objective` by name —
   added explicit migration step in Phase 2 Step 5 to move these tests
   to `test_pruning_strategies.py` and remove the old function.
2. `pruning_n_startup_trials=None` sentinel could reach callback —
   committed to resolving in `optimize()` before `ObjectiveConfig`
   construction; callback always receives a concrete `int`.

**Warnings addressed:**
3. Mock return values in existing callback tests — Phase 2 Step 6 now
   explicitly specifies `dict[str, float]` mock values.
4. `_resolve_pruner()` dispatch — Phase 4 Step 1 now explicitly states
   it is only called for `isinstance(pruner, str)`.
5. `objective_metrics` backward compatibility — Phase 2 Step 1 now gives
   `objective_metrics` a default of `None` → `["calibration_error", "nrmse"]`
   in the callback constructor, removed in Phase 3.
6. Auto-detect location — committed to `optimize()` after study creation,
   not `ObjectiveConfig.__post_init__()`.
7. `test_study.py` does not exist — moved from "Files to modify" to
   "Files to create" in Phase 4.

**Suggestions noted:**
8. `should_prune_primary` scalar extraction — added explicit extraction
   path `scores[self._primary_metric]` in Phase 2 Step 3.
9. Manual verification for `"none"` — changed to callback list inspection.
10. Cross-schema migration test — added to Phase 1 Step 6 test list.
