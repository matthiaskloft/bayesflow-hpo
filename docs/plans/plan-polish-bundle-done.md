# Plan: Polish Bundle — Display, Inference Time, Plots, Checkpoints, Notebook

**Created**: 2026-03-16
**Author**: Claude

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-16 | |
| Phase 1: Display & reporting fixes | MERGED | 2026-03-16 | PR #37 |
| Phase 2: Inference time rework | MERGED | 2026-03-16 | PR #38 |
| Phase 3: Plot refactoring | MERGED | 2026-03-16 | PR #39 |
| Phase 4: Checkpoint-aware retraining | MERGED | 2026-03-16 | PR #39 |
| Phase 5: Notebook update & rename | MERGED | 2026-03-16 | PR #39 |
| Ship | MERGED | 2026-03-16 | |

## Summary

**Motivation**: After the initial HPO feature set stabilized, several UX
rough edges remain: redundant row numbers in tables, an inference time
metric that reports a ratio rather than interpretable seconds, noisy log
output showing "best" instead of per-metric values, a `plot_study()` layout
cluttered with 3D/parallel subplots, no way to warm-start retraining from
a checkpointed model, and an outdated quickstart notebook.

**Outcome**: Cleaner tabular output, an inference time cost metric in
seconds-per-dataset, focused log output, a streamlined `plot_study()`
layout, `build_continuous_approximator()` that optionally loads checkpoint
weights, and a renamed "Getting Started" notebook reflecting all changes.

## Assumptions

- `inference_time_s` as "seconds per dataset averaged over conditions" is
  more interpretable than the current `inference_time_ratio` (normalized
  against simulation time). The ratio is removed as a default cost metric.
- The 3D Pareto and parallel coordinates subplots in `plot_study()` are
  not useful enough to justify the visual clutter — they remain available
  as standalone functions.
- Checkpoint loading in `build_continuous_approximator()` is opt-in via a
  new parameter, not forced. But `optimize()` could default to using it
  when a checkpoint pool is available.

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Row index in tables | Reset index / set trial as index / hide on display | Hide default index on display via `.to_string(index=False)` in print contexts; keep `trial_number`/`rank` as regular columns | Avoids breaking `.iloc[]`/column access patterns while removing visual clutter |
| Inference time semantics | Keep ratio / raw total seconds / seconds-per-dataset | Seconds-per-dataset | Most interpretable: "how long does one inference call take?" |
| Inference time source | Total validation wall-clock / pure inference timing from pipeline | Pure inference timing from `ValidationResult.timing["inference"]` | Excludes metric computation time; reflects actual inference cost |
| Progress log content | Show "best" / show per-metric values / show cost + metrics | Show cost metrics + chosen objective metrics | User wants to see the actual metric values, not just a single "best" number |
| `plot_study()` 3-obj layout | Keep 3D+parallel / remove both / keep one | Remove both from `plot_study()`, keep as standalone functions | Reduces clutter; users who want them can call them directly |
| Parallel coordinates cost axis | Raw values / inverted + log-transformed | Invert cost metric and log-transform | Makes "better = up" consistent across all axes; log scale handles the large range |
| Checkpoint loading | Always load / opt-in parameter / auto-detect | New `checkpoint_dir` parameter, auto-detect from `CheckpointPool` | Explicit is better than implicit; auto-detect is a convenience default |

## Scope

### In Scope
1. Remove row numbers from `trials_to_dataframe()` and `trial_table()`
2. Rework `inference_time_s` to seconds-per-dataset (averaged over conditions)
3. Clean up progress logs: show cost metrics + chosen metrics, drop "best"
4. Remove 3D subplot and parallel subplot from `plot_study()`
5. Update `plot_parallel_coordinates()`: invert cost metric, log-transform cost metric
6. `summarize_study()`: indicate time unit "(s)" for time metrics
7. `build_continuous_approximator()`: add optional checkpoint weight loading
8. Rename `quickstart.ipynb` → `getting_started.ipynb` and update content

### Out of Scope
- Removing `plot_pareto_3d()` or `plot_parallel_coordinates()` as standalone functions
- Changing the Optuna study objective structure (still minimize-all)
- Adding new validation metrics
- Changing the checkpoint pool eviction strategy

## Implementation Plan

### Phase 1: Display & reporting fixes

Small formatting changes across three files.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/results/extraction.py` — set DataFrame index in `trials_to_dataframe()` and `trial_table()`, add time unit to `summarize_study()`
- `src/bayesflow_hpo/optimization/study.py` — rework progress log in `optimize_until()` to show cost + chosen metrics instead of "best"
- `src/bayesflow_hpo/optimization/objective.py` — update `_log_trial_summary()` to show all objective metric values + cost
- `tests/test_extraction.py` — update assertions for new index behavior
- `tests/test_study.py` — update log assertions if any

**Steps:**
1. In `trials_to_dataframe()`: do NOT set index — keep `trial_number` as a regular column. Instead, document that callers can use `.to_string(index=False)` or `.style.hide(axis="index")` for clean display.
2. In `trial_table()`: same approach — keep `rank` and `trial` as regular columns, no index change. The row numbers are only visible in certain display contexts (Jupyter shows them, `.to_markdown()` does not).
3. In `summarize_study()`: append `" (s)"` to time-related metrics in the display (training_time_s, inference_time_s). Change the inference_time objective display to show "(s)" unit.
4. In `_log_trial_summary()` (objective.py): replace the current hardcoded nrmse/correlation logging with a loop over `objective_metrics` + cost metric name. Pass these as parameters.
5. In `optimize_until()` progress log: replace `best: {best_str}` with per-objective values from the best trial so far (e.g., `cal_error: 0.012 | nrmse: 0.05 | inference_time: 0.3s`).
6. Update tests to match new output format.

**Depends on:** None

### Phase 2: Inference time rework

Change the inference time cost metric from a simulation-time ratio to
seconds-per-dataset, sourced from the pure inference timing in the
validation pipeline.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/validation/pipeline.py` — already tracks `timing["inference"]`; no change needed
- `src/bayesflow_hpo/optimization/objective.py` — extract per-dataset inference time from validation result; store as `inference_time_s`
- `src/bayesflow_hpo/objectives.py` — replace `compute_inference_time_ratio()` with `compute_inference_time_per_dataset()`; remove ratio logic
- `src/bayesflow_hpo/results/extraction.py` — remove `inference_time_ratio` from `DEFAULT_RESULT_ATTRS`; keep `inference_time_s`
- `tests/test_objectives.py` — update tests for new function
- `tests/test_extraction.py` — update expected columns

**Steps:**
1. In `objectives.py`: rename `compute_inference_time_ratio()` to `compute_inference_time_per_dataset(inference_time: float, n_datasets: int) -> float` that returns `inference_time / max(n_datasets, 1)`. Remove `sim_time_per_sim` and ratio logic.
2. In `objective.py` `GenericObjective.__call__()`:
   - **Inline** the default validation path: when `config.validate_fn is None`, call `run_validation_pipeline()` directly in the objective (not via `default_validate_fn`) and extract both `result.summary` and `result.timing["inference"]`. This avoids double-calling the pipeline.
   - When `config.validate_fn is not None` (custom hook), measure total wall-clock time and divide by `n_conditions` as a fallback. Document that custom `validate_fn` users get wall-clock time (includes metric computation) rather than pure inference time.
   - Store `inference_time_s` = pure inference time / n_conditions (seconds per dataset).
   - Compute `cost_score` = `inference_time_s` directly (no normalization needed — Optuna minimizes it).
   - Keep `default_validate_fn` as a standalone function for backward compatibility but it is no longer called internally.
3. Remove `inference_time_ratio` from `DEFAULT_RESULT_ATTRS` and `_log_trial_summary`.
4. Keep `FAILED_TRIAL_COST = 1e6` — even though the cost metric changes from ratio to seconds, 1e6 seconds (~278 hours) is universally unattainable and safely dominates any real inference time. This preserves Pareto front stability.
5. Update tests.

**Depends on:** Phase 1 (for consistent log output format)

### Phase 3: Plot refactoring

Simplify `plot_study()` and improve `plot_parallel_coordinates()`.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/results/visualization.py` — remove 3D+parallel from `_plot_study_3obj()`, update parallel coordinates to invert/log-transform cost
- `tests/test_visualization.py` — update tests

**Steps:**
1. In `plot_study()`: change `n_obj < 2` check to raise `ValueError` only for single-objective. Remove the `n_obj > 3` upper bound. For any `n_obj >= 2`, always call `_plot_study_2obj()` which uses the first two objectives. Update the docstring: "`plot_study()` produces a 2x2 grid using the first two objectives. For 3+ objective studies, use `plot_pareto_3d()` and `plot_parallel_coordinates()` separately for full-dimensional views."
2. Remove `_plot_study_3obj()` function entirely.
3. In `plot_parallel_coordinates()`:
   - After building the data matrix, identify the cost metric axis (last axis by convention).
   - Invert cost metric values: `data[:, -1] = -data[:, -1]` so that "better" (lower cost) maps to higher normalized values.
   - Apply log-transform to cost metric before normalization: `data[:, -1] = np.log1p(np.abs(data[:, -1]))`.
   - Update axis label to indicate transformation (e.g., `"-log(inference_time)"`).
4. Update `plot_study()` docstring to remove 3-obj documentation.
5. Update tests — remove 3-obj `plot_study` test, add parallel coordinates inversion test.

**Depends on:** Phase 2 (inference time semantics must be settled first)

### Phase 4: Checkpoint-aware retraining

Allow `build_continuous_approximator()` to load pre-trained weights from
a checkpoint, so users can warm-start retraining instead of training from
scratch.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/builders/workflow.py` — add `checkpoint_dir` parameter to `build_continuous_approximator()`
- `src/bayesflow_hpo/results/export.py` — no changes needed (already supports save/load)
- `tests/test_builders.py` — add checkpoint loading test
- `src/bayesflow_hpo/__init__.py` — ensure `CheckpointPool` is re-exported if not already

**Steps:**
1. Add `checkpoint_dir: str | Path | None = None` parameter to `build_continuous_approximator()`.
2. After building the approximator, if `checkpoint_dir` is not None:
   - Look for `weights.weights.h5` in the directory.
   - Call `approximator.load_weights(str(checkpoint_dir / "weights.weights.h5"))`.
   - Log that weights were loaded.
3. If loading fails (file not found, incompatible shapes), log a warning and continue with the freshly built model.
4. Document the parameter in the docstring, including the relationship with `CheckpointPool.best_checkpoint_dir`.
5. Add a test that builds an approximator, saves weights, rebuilds with checkpoint, and verifies weights are loaded.

**Depends on:** None (independent of phases 1-3)

### Phase 5: Notebook update & rename

Rename the notebook and update it to reflect all changes from phases 1-4.

**Files to create:**
- `examples/getting_started.ipynb` — renamed and updated notebook

**Files to modify:**
- `README.md` — update notebook link/reference
- `examples/quickstart.ipynb` — delete (replaced by getting_started.ipynb)

**Steps:**
1. Copy `quickstart.ipynb` to `getting_started.ipynb`.
2. Update the title from "Quickstart" to "Getting Started".
3. Update Section 3 (Inspect Results) to reflect:
   - `plot_study()` now always uses 2x2 layout
   - Tables no longer show row numbers
   - `summarize_study()` shows time in seconds
4. Update Section 4 (Model Selection & Retraining) to show checkpoint loading:
   - Use `build_continuous_approximator(config, adapter, search_space, checkpoint_dir=pool.best_checkpoint_dir)` instead of building from scratch.
5. Update inference time references (replace `inference_time_ratio` with `inference_time_s`).
6. Delete `quickstart.ipynb`.
7. Update README.md notebook reference.
8. Re-run the notebook to ensure outputs are current.

**Depends on:** Phases 1-4 (all code changes must be complete)

## Verification & Validation

- **Automated**:
  - `pytest tests/ -v` — all existing tests pass after updates
  - `ruff check src/ tests/` — lint clean
  - New tests for: index-less tables, per-dataset inference time, parallel coordinates inversion, checkpoint loading

- **Manual**:
  - Run `getting_started.ipynb` end-to-end from a fresh clone
  - Verify `trial_table()` output has no row numbers when displayed
  - Verify `summarize_study()` shows "(s)" for time metrics
  - Verify `plot_study()` produces a clean 2x2 grid (no 3D subplot)
  - Verify `plot_parallel_coordinates()` shows inverted+log-transformed cost axis
  - Verify progress logs show metric values instead of "best"

## Dependencies

- No new external dependencies
- BayesFlow 2.x (existing)
- Optuna (existing)

## Notes

- The `inference_time_ratio` user attribute is removed from new trials.
  Old studies loaded via `resume=True` may still have this attribute; the
  extraction code should handle its absence gracefully.
- `FAILED_TRIAL_COST` stays at `1e6` — 1e6 seconds (~278 hours) safely
  dominates any real inference time and preserves Pareto front stability.
- `plot_pareto_3d()` and `plot_parallel_coordinates()` remain available as
  standalone functions for users who want them.

## Review Feedback

Reviewed in 1 iteration. 5 findings (3 blockers, 2 warnings) — all addressed:

1. **BLOCKER (resolved)**: Phase 2 option (b) would double-call `run_validation_pipeline()`. Fixed: inline default validation in objective, call pipeline once, extract both metrics and timing. Custom `validate_fn` falls back to wall-clock / n_datasets (documented).
2. **BLOCKER (resolved)**: `FAILED_TRIAL_COST` change from 1e6 to 1e3 risks Pareto instability. Fixed: keep `FAILED_TRIAL_COST = 1e6`.
3. **BLOCKER (resolved)**: Removing 3-obj `plot_study()` branch silently changes behavior. Fixed: `plot_study()` now accepts any `n_obj >= 2` and always uses 2x2 layout; docstring updated to direct users to standalone 3D/parallel functions.
4. **WARNING (resolved)**: Ambiguous index decision for tables. Fixed: keep `trial_number`/`rank` as regular columns, no index change. Callers use `.to_string(index=False)` for clean display.
5. **WARNING (resolved)**: Custom `validate_fn` cannot provide pure inference timing. Fixed: documented that custom hooks get wall-clock / n_datasets (includes metric computation).
