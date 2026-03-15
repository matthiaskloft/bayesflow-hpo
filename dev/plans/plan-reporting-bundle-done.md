# Plan: Reporting Bundle (TODOs 1+2+3)

**Created**: 2026-03-15
**Author**: Claude

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-15 | |
| Phase 1: `trial_table()` | MERGED | 2026-03-15 | PR #30 |
| Phase 2: `best_config()` + `compare_trials()` | MERGED | 2026-03-15 | PR #30 |
| Phase 3: Slim down `summarize_study()` | MERGED | 2026-03-15 | PR #30 |
| Ship | MERGED | 2026-03-15 | PR #30 |

## Summary

**Motivation**: `summarize_study()` is a monolithic 175-line function that mixes trial counts, Pareto info, a leaderboard, and hyperparameters into one wide text block. Floats print with full precision (e.g. `0.05454749016213018`), the output exceeds 80 characters, and there's no way to get a CSV-exportable trial table or compare specific trials side by side.

**Outcome**: Four focused reporting functions:
- `trial_table()` — ranked DataFrame of top-k trials with objectives, hyperparams, and metrics; CSV-savable
- `best_config()` — pretty-printed hyperparameter dict for a specific trial
- `compare_trials()` — side-by-side comparison of 2–5 trials
- `summarize_study()` — slimmed to just trial counts + best trial + objectives (delegates detail to the above)

All floats are rounded contextually. Output fits in 80-char terminals.

## Assumptions

- `_objective_column_names()` and `_fmt_param_count()` remain shared utilities — no changes needed
- `trials_to_dataframe()` stays as-is (raw data extraction); `trial_table()` builds on top of it for display
- The existing test helpers (`_make_study`, `_make_study_with_rejected`) cover enough scenarios for the new functions

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Where to put new functions | New file `reporting.py` vs keep in `extraction.py` | Keep in `extraction.py` | All functions operate on `optuna.Study` → same module. Avoids import churn and circular deps. File stays under 400 lines. |
| `trial_table()` return type | Formatted string vs DataFrame | DataFrame | Users want `.to_csv()`, `.to_markdown()`, and programmatic access. Display rounding applied via pandas Styler or float formatting. |
| `best_config()` return type | str vs dict | dict (with `__repr__` via pretty-print) + optional print | Dict is programmatically useful; function also prints when `verbose=True` (default). |
| `compare_trials()` return type | Styled DataFrame vs plain DataFrame | Plain DataFrame | Styled objects break `.to_csv()` and `.to_markdown()`. Users can apply `df.style` themselves in notebooks. |
| Float rounding strategy | Global `round()` vs per-column | Per-column contextual | Learning rates need scientific notation (4 sig figs), dropout 2 decimals, widths/depths as int, times 1 decimal. Use a `_round_value()` helper keyed on column name patterns. |
| `summarize_study()` backward compat | Keep `top_k` param vs remove it | Remove `top_k` (pre-1.0) | Package is pre-1.0 so no deprecation needed. `top_k` was only used for the leaderboard which moves to `trial_table()`. Keep `select_by`. |

## Scope

### In Scope
- `trial_table()` — new function in `extraction.py`
- `best_config()` — new function in `extraction.py`
- `compare_trials()` — new function in `extraction.py`
- `_round_value()` — private helper for contextual float formatting
- Refactor `summarize_study()` to be compact (trial counts + best trial only)
- Tests for all new functions
- Update `results/__init__.py` and top-level `__init__.py` exports
- Update `__all__`

### Out of Scope
- Changes to `trials_to_dataframe()` (stays as raw extraction)
- Visualization / plotting changes (TODO 4)
- Search space default audits (TODO 5)
- Quickstart notebook updates (TODO 6 — will be a follow-up after this lands)

## Implementation Plan

### Phase 1: `trial_table()`

**Files to create:** None

**Files to modify:**
- `src/bayesflow_hpo/results/extraction.py` — add `_round_value()` helper and `trial_table()` function
- `tests/test_results/test_extraction.py` — add `TestTrialTable` test class
- `src/bayesflow_hpo/results/__init__.py` — export `trial_table`
- `src/bayesflow_hpo/__init__.py` — export `trial_table`

**Steps:**
1. Add `_round_value(key: str, value: Any) -> Any` helper that rounds floats contextually:
   - Keys containing `lr`, `learning_rate`, or `initial_lr` → `f"{v:.2e}"` (scientific notation)
   - Keys containing `dropout` → `round(v, 2)`
   - Keys containing `dim`, `width`, `depth`, `units`, `heads`, `layers` → `int(v)`
   - Keys containing `time` (but not `dim`) → `round(v, 1)`
   - Other floats → `round(v, 4)` (general 4 sig fig)
   - Non-floats (int, str, bool, None) → pass through
2. Implement `trial_table()`:
   ```python
   def trial_table(
       study: optuna.Study,
       top_k: int | None = None,
       select_by: int = 0,
       metrics: list[str] | None = None,
       trained_only: bool = True,
   ) -> pd.DataFrame
   ```
   - Get trained trials (reuse filtering logic from `trials_to_dataframe`)
   - Sort by `select_by` objective
   - Slice to `top_k`
   - Build DataFrame with columns: `rank`, `trial`, objective columns, `param_count` (formatted), hyperparams, optional metric columns
   - Apply `_round_value()` to all cells
3. Add tests:
   - `TestRoundValue`: learning rate → scientific, dropout → 2dp, dims → int, time → 1dp, generic float → 4dp, non-float passthrough, NaN/inf passthrough
   - `TestTrialTable`: basic ranking, top_k filtering, top_k > n_trials, metrics inclusion, single-objective, empty study, rejected-only study, CSV round-trip
4. Update exports

**Depends on:** None

### Phase 2: `best_config()` + `compare_trials()`

**Files to create:** None

**Files to modify:**
- `src/bayesflow_hpo/results/extraction.py` — add `best_config()` and `compare_trials()`
- `tests/test_results/test_extraction.py` — add `TestBestConfig` and `TestCompareTrials`
- `src/bayesflow_hpo/results/__init__.py` — export both
- `src/bayesflow_hpo/__init__.py` — export both

**Steps:**
1. Implement `best_config()`:
   ```python
   def best_config(
       study: optuna.Study,
       trial_number: int | None = None,
       select_by: int = 0,
   ) -> dict[str, Any]
   ```
   - If `trial_number` is given, look up that trial; otherwise find best by `select_by`
   - Return dict of hyperparameters with values rounded via `_round_value()`
   - Print the config as a formatted block (key-value pairs, 80-char width)
2. Implement `compare_trials()`:
   ```python
   def compare_trials(
       study: optuna.Study,
       trial_numbers: list[int],
       metrics: list[str] | None = None,
   ) -> pd.DataFrame
   ```
   - Validate 2–5 trial numbers exist (raise `ValueError` if outside range or trial not found)
   - Build plain DataFrame: rows = hyperparams + objectives + metrics, columns = trial numbers
   - No styled highlighting — return a plain DataFrame so `.to_csv()` and `.to_markdown()` work. Users can apply `df.style` in notebooks if desired.
   - Apply `_round_value()` to all cells
3. Add tests:
   - `TestBestConfig`: by objective, by trial number, non-existent trial (ValueError), single-trial study
   - `TestCompareTrials`: basic 2-trial case, 3-trial case, <2 trials (ValueError), >5 trials (ValueError), non-existent trial (ValueError), duplicate trial numbers
4. Update exports

**Depends on:** Phase 1 (uses `_round_value()`)

### Phase 3: Slim down `summarize_study()`

**Files to create:** None

**Files to modify:**
- `src/bayesflow_hpo/results/extraction.py` — refactor `summarize_study()`
- `tests/test_results/test_extraction.py` — update `TestSummarizeStudy` assertions

**Steps:**
1. Refactor `summarize_study()` to output only:
   - Study name + separator
   - Trial counts (trained / rejected / pruned / failed) — same as now
   - Objectives line — same as now
   - Best trial block: trial number + objective values + param count (rounded)
   - Pareto front count (multi-objective only)
   - A hint: `"Use trial_table() for detailed results, best_config() for hyperparameters."`
2. Remove from `summarize_study()`:
   - The top-k leaderboard (→ `trial_table()`)
   - The hyperparameters section (→ `best_config()`)
   - The `_fmt_trial()` inner function (replaced by simpler inline formatting)
3. Ensure all output fits within 80 characters
4. Update existing tests to match new output format (remove assertions about leaderboard/hyperparams)
5. Remove `top_k` parameter (pre-1.0, no deprecation needed)

**Depends on:** Phase 1 and Phase 2

## Verification & Validation

- **Automated**: All existing `TestSummarizeStudy`, `TestTrialsToDataframe`, `TestGetParetoTrials` tests still pass. New test classes for `trial_table`, `best_config`, `compare_trials` cover: basic usage, edge cases (empty study, single trial), rounding correctness, and CSV export round-trip for `trial_table`.
- **Manual**: Run `ruff check src/ tests/` to confirm lint passes. Run the quickstart notebook's last cell replacing `summarize_study()` with `trial_table()` + `best_config()` to visually verify output fits 80 chars and floats are clean.
- **CI**: Existing `pytest` + `ruff` CI workflow covers all changes.

## Dependencies

- `optuna` (already a dependency)
- `pandas` (already a dependency)
- No new dependencies

## Notes

_Living section — updated during implementation._

- `top_k` parameter on `summarize_study()`: since the package is pre-1.0, we can remove it rather than deprecate. Decision to be made during Phase 3 implementation.
- `_round_value()` is intentionally private — it encodes heuristics about HPO parameter naming conventions that shouldn't be part of the public API.

## Review Feedback

Reviewed in 1 iteration. 8 findings (1 blocker, 3 warnings, 4 suggestions).

**Resolved:**
- **Blocker**: `compare_trials()` highlighting was underspecified → clarified as plain DataFrame (no styled output)
- **Warning**: `_round_value()` missed `initial_lr` pattern → added to learning rate check
- **Warning**: Missing test scenarios → expanded test lists for all functions + `_round_value()` helper
- **Warning**: `summarize_study()` docstring outdated → will update in Phase 3

**Noted (no plan change needed):**
- `trial_table()` vs `trials_to_dataframe()` separation is correct
- `compare_trials()` 2–5 limit is reasonable; boundary tests added
- Export coverage is complete
- `top_k` on `summarize_study()` → remove entirely (pre-1.0)
