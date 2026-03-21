# Plan: plot_study() Redesign

**Created**: 2026-03-16
**Spec**: `docs/superpowers/specs/2026-03-16-plot-study-redesign.md`

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-16 | |
| Phase 1: Core sub-plot rewrites | MERGED | 2026-03-16 | [PR #43](https://github.com/matthiaskloft/bayesflow-hpo/pull/43) |
| Phase 2: Orchestrator + integration | MERGED | 2026-03-16 | [PR #45](https://github.com/matthiaskloft/bayesflow-hpo/pull/45) |
| Ship | MERGED | 2026-03-16 | All phases complete |

## Summary

**Motivation**: The current `plot_study()` has a rigid 2×2 layout, plots objective[0] vs `param_count` instead of actual objectives against each other, only shows single-objective history, and cannot encode a 3rd dimension on 2D Pareto projections. Multi-objective studies (especially 3-objective) lose important information.

**Outcome**: `plot_study()` produces an adaptive 3-row grid (Pareto projections / per-objective history / per-objective importance) that scales correctly for 2- and 3-objective studies. Each sub-plot function works standalone with `max_cols` wrapping or embedded via an `axes` array, matching the dual-mode pattern already used by `plot_metric_panels()`.

## Assumptions

- Optuna `study.metric_names` (or `_metric_names` fallback) continues to work as the objective name source — already tested in `test_extraction.py::TestObjectiveColumnNames`.
- `viridis_r` is the correct colormap direction for minimization-dominated studies (dark = low/good). This is a deliberate change from `plot_pareto_projections()` which currently uses `viridis`.
- The existing `_draw_pareto_overlay()` and `_pareto_front_2d()` helpers are correct and reusable — they compute 2D non-dominated fronts and draw step-lines + markers.
- `plot_pareto_3d()`, `plot_parallel_coordinates()`, and `plot_metric_scatter()` are unchanged by this work.

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Grid layout engine | Manual subplots, GridSpec, subfigures | `GridSpec` | Needed for spanning columns (centering 1 Pareto panel in 2-col grid) and variable row heights. GridSpec is the standard matplotlib tool for this. |
| Centering logic for fewer panels | Left-align, center-span, skip columns | Center-span | When a row has fewer panels than `n_cols`, the panel spans multiple GridSpec columns. Avoids awkward empty space on one side. |
| 3rd dimension encoding | Color only, size only, both, configurable | Configurable `third_dim` param (`"color"` / `"size"` / `"none"`) | Different use cases benefit from different encodings. Color is the default since it's most informative. |
| History plot style | Scatter + step, step only | Step only (best-so-far) | Scatter points add clutter without insight in multi-panel layout. The best-so-far line is what users actually care about. |
| Importance failure handling | Raise, skip panel, drop entire row | Drop entire row if ALL fail, placeholder per panel otherwise | Importance can fail for small studies. Dropping the row keeps the figure clean; individual placeholders handle partial failures. |
| `select_by` removal | Keep on plot_study, remove | Remove from `plot_study()` | Each objective gets its own panels, so `select_by` is meaningless. It stays on `plot_parallel_coordinates()` where it's still needed. |
| Objective count limit | Arbitrary, 2-3 only, unlimited with wrapping | 2-3 for `plot_study()`, unlimited for standalone | >3 objectives produce too many pairwise panels (6+ Pareto pairs for 4 objectives). Standalone functions handle this via `max_cols`. |

## Scope

### In Scope

- Rewrite `plot_pareto_front()`: pairwise projections, correct axis labels, `third_dim`, `axes` array, `max_cols`
- Rewrite `plot_optimization_history()`: per-objective panels, step-only, `axes` array, `max_cols`
- Extend `plot_param_importance()`: per-objective targeting, `axes` array, `max_cols`, graceful degradation
- Rewrite `plot_study()`: 3-row GridSpec orchestrator, 2-3 objective support, `third_dim`, `figsize`
- Add `max_cols` to `plot_pareto_projections()` and `plot_metric_panels()`
- Update all tests for changed signatures and new behavior
- Remove `select_by` and `metrics` params from `plot_study()`

### Out of Scope

- `plot_pareto_3d()` — unchanged
- `plot_parallel_coordinates()` — unchanged
- `plot_metric_scatter()` — unchanged (removed from `plot_study()` grid but still standalone)
- New plot types not in the spec
- Notebook updates (separate follow-up)

## Implementation Plan

### Phase 1: Core Sub-Plot Rewrites

Rewrite the three sub-plot functions that `plot_study()` depends on, plus add `max_cols` to two existing functions. Each function gets the dual-mode pattern (standalone figure vs embedded axes).

**Files to modify:**
- `src/bayesflow_hpo/results/visualization.py` — rewrite `plot_pareto_front()`, `plot_optimization_history()`, `plot_param_importance()`; add `max_cols` to `plot_pareto_projections()` and `plot_metric_panels()`
- `tests/test_visualization.py` — update tests for all changed signatures

**Steps:**

1. **Rewrite `plot_pareto_front()`** (`plot_pareto_front` function in visualization.py):
   - New signature: `(study, axes=None, *, third_dim="color", max_cols=3, figsize=None)`
   - Generate all pairwise `(i, j)` objective combinations
   - For each pair, draw two Pareto layers (matching existing `plot_pareto_projections()` pattern):
     - Layer 1: `_draw_pareto_overlay(ax, xs, ys, draw_step=True, draw_markers=False)` — 2D non-dominated step line per projection
     - Layer 2: separately scatter `study.best_trials` as `c.ACCENT` star markers (`c.PARETO_MARKER`, `c.PARETO_SIZE`) — these are N-D Pareto-optimal, may not be non-dominated in the 2D view
   - For ≥3 objectives, encode omitted objective via `third_dim`:
     - `"color"`: scatter with `cmap="viridis_r"`, add `plt.colorbar(sc, ax=ax)` labeled with omitted objective name
     - `"size"`: `_normalize_to_sizes(omitted_values)` for marker sizes
     - `"none"`: uniform markers (`c.PRIMARY`, `c.ALPHA_TRIAL`)
   - For 2 objectives, `third_dim` is ignored (no omitted objective)
   - Standalone mode: create figure with `max_cols`-wrapped subplots grid
   - Embedded mode: draw into provided axes array
   - Use `_objective_column_names()` for all axis labels

2. **Rewrite `plot_optimization_history()`** (`plot_optimization_history` function in visualization.py):
   - New signature: `(study, axes=None, *, max_cols=3, figsize=None)`
   - One panel per objective (not just first)
   - Step line only (remove scatter), color `c.BEST_LINE`
   - Direction-aware best-so-far per objective:
     ```python
     direction = study.directions[obj_idx]
     best_func = max if direction == optuna.study.StudyDirection.MAXIMIZE else min
     running_best = list(itertools.accumulate(values, best_func))
     ```
   - Y-axis label and title from `_objective_column_names(study)[obj_idx]`
   - Standalone/embedded dual mode

3. **Extend `plot_param_importance()`** (`plot_param_importance` function in visualization.py):
   - New signature: `(study, axes=None, top_k=10, *, max_cols=3, figsize=None)`
   - One bar chart per objective using per-objective target callable:
     ```python
     target = lambda t, idx=obj_idx: t.values[idx]
     optuna.importance.get_param_importances(study, target=target)
     ```
   - Per-panel graceful degradation (placeholder text on failure)
   - Return `None` if ALL panels fail (signal for orchestrator)
   - Remove `target_name` param (auto per-objective replaces it)

4. **Add `max_cols` to `plot_pareto_projections()`** and **`plot_metric_panels()`**:
   - Use `max_cols` to wrap grid rows: `n_rows = ceil(n_panels / max_cols)`, create `(n_rows, min(n_panels, max_cols))` subplots grid

5. **Update tests**: Adapt `TestPlotParetoFront`, `TestPlotOptimizationHistory`, `TestPlotParamImportance`, `TestPlotParetoProjections`, `TestPlotMetricPanels` for new signatures and behaviors. Add tests for `third_dim` modes, per-objective panels, and dual-mode axes.

**Depends on:** None

### Phase 2: Orchestrator + Integration

Build the new `plot_study()` orchestrator using GridSpec, wire up the rewritten sub-plots, and finalize tests.

**Files to modify:**
- `src/bayesflow_hpo/results/visualization.py` — rewrite `plot_study()` and remove `_plot_study_2obj()`
- `src/bayesflow_hpo/results/__init__.py` — update exports if any public API changes
- `tests/test_visualization.py` — update `TestPlotStudy` for new grid, add 3-obj orchestrator tests

**Steps:**

1. **Rewrite `plot_study()`**:
   - New signature: `(study, *, third_dim="color", figsize=None)`
   - Validate: raise `ValueError` for <2 or >3 objectives
   - Compute `n_pairs` (1 for 2-obj, 3 for 3-obj) and `n_cols = max(n_pairs, n_obj)`
   - Create `GridSpec(3, n_cols)` figure
   - Row 0: `plot_pareto_front(study, axes=pareto_axes, third_dim=third_dim)` — center-span if `n_pairs < n_cols`
   - Row 1: `plot_optimization_history(study, axes=history_axes)`
   - Row 2: `plot_param_importance(study, axes=importance_axes)` — drop row if returns `None`
   - Auto figsize: `(5 * n_cols, 4.5 * n_rows)`

2. **Remove `_plot_study_2obj()`** — no longer needed

3. **Update `__init__.py`** — verify exports (no new public functions, but signature changes are breaking)

4. **Update `TestPlotStudy`**:
   - Test 2-obj: 3 rows, 2 cols, centered Pareto panel
   - Test 3-obj: 3 rows, 3 cols, 3 Pareto + 3 history + 3 importance panels
   - Test <2 obj: ValueError
   - Test >3 obj: ValueError with helpful message
   - Test importance failure: figure shrinks to 2 rows
   - Test empty study: placeholder text, no crash

**Depends on:** Phase 1

## Verification & Validation

- **Automated**:
  - `pytest tests/test_visualization.py -v` — all tests pass
  - `ruff check src/` — no lint errors
  - Verify color constants are used (existing `TestColorConstantsUsed` tests)
- **Manual**:
  - Generate `plot_study()` with a real 2-objective Optuna study and visually confirm: Pareto panel centered, 2 history panels, 2 importance panels
  - Generate `plot_study()` with a real 3-objective study and confirm: 3 Pareto pairs with `viridis_r` 3rd-dim encoding, 3 history panels, 3 importance panels
  - Call `plot_pareto_front()` standalone (no axes) and verify auto-grid works
  - Test `third_dim="size"` and `third_dim="none"` modes visually

## Dependencies

- `matplotlib.gridspec.GridSpec` — already a matplotlib dependency
- No new package dependencies

## Notes

_Living section — updated during implementation._

- The `_draw_pareto_overlay()` helper already handles the 2D step-line + star-marker pattern. Reuse it directly in the new `plot_pareto_front()` for each pairwise panel.
- `study.best_trials` returns N-dimensional Pareto-optimal trials. These should be highlighted in every 2D projection, even if they're not non-dominated in that specific 2D view.
- The `viridis_r` change from `viridis` is intentional: dark purple = low values = "good" for minimization objectives, which is the common case in HPO.

**Phase 1 implementation notes:**
- Added `_setup_grid()` shared helper to reduce grid-creation duplication across all rewritten functions. Handles `n_panels=0` edge case.
- `_plot_study_2obj()` updated as temporary compat shim — passes duplicated axes so all objectives draw on the same axis. This is visually imperfect but keeps existing `plot_study()` tests passing. Phase 2 replaces it entirely with GridSpec.
- For >3 objectives, `plot_pareto_front()` encodes the first omitted dimension (by index) rather than ignoring all omitted dimensions. This is reasonable since encoding multiple omitted dims simultaneously isn't feasible.
- `plot_param_importance()` returns `None` when all per-objective panels fail (Optuna importance evaluator needs sufficient trials). Test fixtures have 4-6 trials which is below Optuna's threshold, so these tests are skipped.

**Phase 2 implementation notes:**
- `plot_study()` now uses `GridSpec` with 3 rows × `n_cols` columns. When importance fails, row 2 axes are removed and figure shrinks to 2/3 height.
- `_plot_study_2obj()` removed entirely.
- `select_by` and `metrics` params removed from `plot_study()`. `third_dim` and `figsize` added.
- >3 objectives raises `ValueError` with helpful message directing users to standalone functions.
- Loop variable renamed from `c` to `col` to avoid shadowing `_colors as c` import (caught by ruff).

## Review Feedback

Reviewed in 2 iterations.

**Iteration 1 — 3 blockers addressed:**

1. **`_draw_pareto_overlay()` dual-layer pattern** (was blocker): Clarified in Phase 1 Step 1 — call with `draw_markers=False` for the 2D step line, then separately plot `study.best_trials` as accent stars. This matches the existing pattern in `plot_pareto_projections()`.

2. **`plot_param_importance()` per-objective target** (was blocker): Clarified in Phase 1 Step 3 — use `target=lambda t, idx=obj_idx: t.values[idx]` callable per objective. Optuna's `get_param_importances(target=...)` accepts any callable `FrozenTrial -> float`.

3. **Direction-aware best-so-far** (was blocker): Clarified in Phase 1 Step 2 — use `study.directions[obj_idx]` to pick `min` vs `max` via `itertools.accumulate`.

**Warnings noted (no plan changes needed):**

- **Breaking API** (`ax` → `axes`): This is a pre-1.0 package; no deprecation path needed. Document in release notes.
- **`plot_pareto_front()` semantics change**: The old obj[0]-vs-param_count behavior moves to `plot_metric_panels()`. Document migration path in release notes.
- **`viridis_r` inconsistency with `plot_pareto_projections()`**: Intentional per spec. `plot_pareto_projections()` is unchanged (out of scope); users can update it separately.
- **`max_cols` wrapping algorithm**: Clarified in Phase 1 Step 4 — ceiling division for row count.
- **Return type for all-fail importance**: `None` return documented; `plot_study()` checks `if result is None:`.
- **Test edge cases**: Added importance failure and empty study tests to Phase 2 Step 4.
