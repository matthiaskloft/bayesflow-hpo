# Plan: Rework Plotting for 2D and 3D Objectives

**Created**: 2026-03-15
**Author**: Claude

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-15 | |
| Phase 1: Core 3-objective support + plot_study | TODO | | |
| Phase 2: Polish 2D plots + tests | TODO | | |
| Phase 3: Update notebook | TODO | | |
| Ship | TODO | | |

## Summary

**Motivation**: The current visualization module (shipped in plan-multi-metric-plots) assumes 2 objectives everywhere: `plot_pareto_front` hardcodes objective[0] vs `param_count` user attr, and there's no way to visualize 3-objective Pareto surfaces. With `objective_mode="pareto"` users can run 3-objective studies (e.g. calibration_error + nrmse + param_count_norm), but the plots can't display them. There's also no single entry point that auto-produces the right panel for a given study.

**Outcome**: Users get:
- A `plot_study(study)` convenience function that auto-detects 2 vs 3 objectives and produces a standard multi-panel figure
- 3-objective support: paired 2D Pareto projections, metric-vs-metric scatter with param_count as marker size, and parallel coordinates
- Polished 2D plots: better axis formatting, consistent styling

## Assumptions

- 3-objective studies use `objective_mode="pareto"` which produces `(metric_1, metric_2, cost_score)` — all minimize-is-better
- `study.directions` always has length equal to the number of objectives
- Objective names are available via `_objective_column_names(study)`
- `trial.user_attrs["param_count"]` stores the raw (unnormalized) param count for all trained trials
- matplotlib is the only plotting dependency (no plotly) — `mpl_toolkits.mplot3d` is part of matplotlib and available without extra installs
- 1-objective and 4+-objective studies are out of scope (per TODO #4)

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| 3D Pareto surface | (a) True 3D scatter with mplot3d, (b) Paired 2D projections, (c) Both | (c) Both | 3D scatter gives the gestalt; 2D projections are readable and precise. `plot_pareto_3d` for the scatter, `plot_pareto_projections` for the 2×1 pairwise views |
| Parallel coordinates | (a) matplotlib manual, (b) optuna.visualization, (c) pandas.plotting | (a) matplotlib | Keeps us dependency-free; optuna's built-in requires plotly; pandas parallel_coordinates requires categorical coloring not objectives |
| `plot_study` dispatch | (a) Single function returning figure, (b) Return dict of axes | (a) Single figure | Users want one call for the whole summary; advanced users call individual functions |
| 3D Pareto front computation | (a) Use `study.best_trials`, (b) Compute from scratch | (a) `study.best_trials` | Optuna already computes the true multi-objective Pareto front; no need to reimplement |
| Cost metric display in 3D/projections | (a) Color only, (b) Size only, (c) `cost_display` kwarg | (c) `cost_display` kwarg | User wants flexibility: `cost_display='color'` (default) or `'size'`. Applied to both `plot_pareto_3d` and `plot_pareto_projections` for the omitted dimension |
| Keep or remove existing functions | Keep all 5, rework internals | Keep all | They're already exported and tested; backwards compatibility matters |
| `plot_study` return type | (a) Figure, (b) Axes array, (c) Dict | (a) Figure | Users want one object for `savefig()`; advanced users access `fig.axes` for customization |
| 3-obj layout strategy | (a) Simple 2×3 grid, (b) GridSpec with spanning | (b) GridSpec | 3D plot needs to span top row for readability; GridSpec handles this cleanly |
| Color scheme | (a) Custom colors, (b) Match BayesFlow, (c) BayesFlow-inspired | (b)+(c) | Match BayesFlow primary `#132a70` where feasible, use complementary accents flexibly. Define in `results/_colors.py` so other bayesflow packages can copy it |

## Scope

### In Scope

- New `plot_pareto_3d(study)` — 3D scatter of all 3 objectives with Pareto front highlighted
- New `plot_pareto_projections(study)` — 3 paired 2D Pareto projections (obj0 vs obj1, obj0 vs obj2, obj1 vs obj2)
- New `plot_parallel_coordinates(study, top_k)` — parallel coordinates of objectives for top-k trials
- New `plot_study(study)` — auto-detecting convenience function:
  - 2-objective: 2×2 grid (pareto front, optimization history, param importance, metric scatter if 2+ user-attr metrics available)
  - 3-objective: 2×3 grid (3D pareto, 3 projections, parallel coordinates, param importance)
- Polish existing 2D plots: consistent color scheme, better legend placement
- Tests for all new functions
- Update notebook to use `plot_study()`

### Out of Scope

- 1-objective or 4+-objective visualization (per TODO #4 spec)
- Interactive/plotly plots
- Custom color themes or style sheets
- Radar charts or GP surrogate visualization
- Saving figures to disk (users call `plt.savefig()` themselves)

## Implementation Plan

### Phase 1: Core 3-objective support + plot_study

**Files to create:**
- `src/bayesflow_hpo/results/_colors.py` — BayesFlow-aligned color constants (easily findable for other packages to copy)

**Files to modify:**
- `src/bayesflow_hpo/results/visualization.py` — add new plot functions, import colors from `_colors.py`
- `src/bayesflow_hpo/results/__init__.py` — export new functions
- `src/bayesflow_hpo/__init__.py` — export + `__all__`
- `tests/test_visualization.py` — add tests for new functions

**Steps:**

1. **Create `results/_colors.py`** — BayesFlow-aligned color scheme:
   ```python
   # BayesFlow-aligned color palette for HPO visualizations.
   # Oriented at the bayesflow package (bayesflow.utils.plot_utils).
   # Other bayesflow-* packages can copy this file for consistency.

   PRIMARY = "#132a70"       # BayesFlow deep blue — trial scatter points
   SECONDARY = "gray"        # Secondary/prior — reference lines, bands
   ACCENT = "red"            # Pareto-front markers, observed values
   BEST_LINE = "#E74C3C"     # Best-so-far step lines (warm red)
   ALPHA_TRIAL = 0.4         # Default scatter transparency
   PARETO_MARKER = "*"       # Pareto point marker style
   PARETO_SIZE = 90          # Pareto marker size
   ```
   Import these in `visualization.py` and use throughout all plot functions.

2. **Add `plot_pareto_3d(study, ax=None, *, cost_display='color', xlabel=None, ylabel=None, zlabel=None)`**:
   - Create `Axes3D` via `fig.add_subplot(projection='3d')` if no ax provided
   - Filter to trained trials with all 3 values present and valid:
     ```python
     trials = [
         t for t in _trained_trials(study)
         if t.values and len(t.values) >= 3
         and all(v is not None and not math.isnan(v) for v in t.values[:3])
     ]
     ```
   - Scatter using `t.values[0], t.values[1], t.values[2]`
   - `cost_display` kwarg controls how `param_count` is shown:
     - `'color'` (default): color by param_count (log-normalized, `cmap="viridis"`) with colorbar
     - `'size'`: map param_count to marker size (log-scaled, 20–200 range)
   - Highlight `study.best_trials` as red stars
   - Auto-label axes from `_objective_column_names(study)`
   - Handle <3 objectives gracefully (text message on empty axes)

3. **Add `plot_pareto_projections(study, axes=None, *, cost_display='color')`**:
   - Create 1×3 subplots if no axes provided
   - For each pair `(i, j)` in `[(0,1), (0,2), (1,2)]`:
     - Scatter `t.values[i]` vs `t.values[j]` for trained trials
     - Show the omitted 3rd dimension via `cost_display`:
       - `'color'` (default): color points by the omitted objective value (with colorbar labeled by its name)
       - `'size'`: map omitted objective to marker size
     - Compute and draw 2D Pareto front via existing `_pareto_front_2d`
     - Highlight study Pareto trials as red stars
   - Auto-label from objective column names
   - Handle <3 objectives by showing available pairs only

4. **Add `plot_parallel_coordinates(study, ax=None, *, top_k=20, select_by=0, metric_order=None)`**:
   - Get trained trials sorted by `select_by` objective, take top_k
   - Axis order: `metric_order` if provided, else `_objective_column_names(study)`
   - Draw one vertical axis per objective (auto-labeled from metric names)
   - Each trial is a line connecting its objective values across axes
   - Color lines by `select_by` objective value (`viridis_r` colormap); best trials get thicker lines (linewidth=2 vs 0.8)
   - Normalize each axis independently to [0, 1] for visual comparability
   - Handle `top_k > len(trained_trials)` by clamping

5. **Add `plot_study(study, *, select_by=0, metrics=None)`**:
   - Detect `n_objectives = len(study.directions)`
   - **2-objective**: 2×2 figure
     - `(0,0)`: `plot_pareto_front(study)`
     - `(0,1)`: `plot_optimization_history(study)`
     - `(1,0)`: `plot_param_importance(study)`
     - `(1,1)`: `plot_metric_scatter(study, m0, m1)` if 2+ user-attr metrics, else `plot_metric_panels(study)`
   - **3-objective**: Use `GridSpec(2, 3)` for subplot spanning:
     ```
     Row 0: [3D Pareto scatter — spans cols 0:2] [Parallel coords — col 2]
     Row 1: [Proj obj0 vs obj1]  [Proj obj0 vs obj2]  [Proj obj1 vs obj2]
     ```
     - `gs[0, 0:2]`: `plot_pareto_3d(study)` (projection='3d', spans 2 cols)
     - `gs[0, 2]`: `plot_parallel_coordinates(study)`
     - `gs[1, 0:3]`: `plot_pareto_projections(study)` — 3 projection subplots
   - **Other (1-obj, 4+-obj)**: raise `ValueError(f"plot_study supports 2 or 3 objectives, got {n}")` with clear message
   - Return the `Figure` object (users access individual axes via `fig.axes`)

6. **Export**: Add `plot_pareto_3d`, `plot_pareto_projections`, `plot_parallel_coordinates`, `plot_study` to `results/__init__.py`, `__init__.py`, and `__all__`.

7. **Tests**: Add to `tests/test_visualization.py`:
   - New `three_objective_study` fixture: study with `directions=["minimize"]*3`, `metric_names=["calibration_error", "nrmse", "param_count_norm"]`, 6 trials with realistic 3-value tuples and user_attrs (including param_count)
   - **Pareto correctness test**: Verify `study.best_trials` for the 3-obj fixture returns a valid Pareto set (no trial dominates another across all 3 objectives)
   - `TestPlotPareto3D`: returns Axes3D, handles <3 objectives, labels correct, handles trial with NaN value
   - `TestPlotParetoProjections`: returns axes array of length 3, handles 2-obj study (1 panel)
   - `TestPlotParallelCoordinates`: returns axes, handles `top_k` > trial count, handles `metric_order` override
   - `TestPlotStudy`: 2-obj study returns Figure with 4 axes, 3-obj study returns Figure with expected layout, single-obj raises `ValueError`

**Depends on:** None

### Phase 2: Polish 2D plots + tests

**Files to modify:**
- `src/bayesflow_hpo/results/visualization.py` — minor style improvements
- `tests/test_visualization.py` — expand edge-case coverage

**Steps:**

1. **Apply `_colors.py` palette** (created in Phase 1) across all 9 plot functions:
   - Replace hardcoded colors in existing functions (`plot_pareto_front`, `plot_optimization_history`, `plot_metric_scatter`, `plot_metric_panels`, `plot_param_importance`) with imports from `_colors.py`
   - Ensure new Phase 1 functions already use `_colors.py` (verify)

2. **Better legend placement**: Use `ax.legend(loc="best", framealpha=0.8)` consistently across all plots. For `plot_metric_scatter` with iso-lines, keep legend inside but use `fontsize="small"` to reduce overlap.

3. **Axis formatting**: Apply `_param_count_formatter()` wherever param_count appears on an axis. Currently applied in `plot_pareto_front` (y-axis) and `plot_metric_panels` (x-axis). Also apply in `plot_pareto_projections` for any axis showing `param_count_norm` (detect by objective name containing "param").

4. **Layout**: Add `fig.tight_layout()` or `fig.set_constrained_layout(True)` to `plot_study()` to prevent overlapping labels in the 2×2 and 2×3 layouts.

5. **Edge-case tests**: Add tests for:
   - `plot_pareto_3d` with exactly 3 trials (minimal Pareto)
   - `plot_pareto_3d` with a trial that has `values=[0.05, float('nan'), 0.3]` (NaN filtered out)
   - `plot_pareto_projections` with 2-objective study (should produce 1 panel, not 3)
   - `plot_study` with empty study (no trained trials — should still return Figure with placeholder text)
   - `plot_parallel_coordinates` with `top_k` larger than trial count (clamps gracefully)
   - Verify color constants are used (spot-check scatter facecolors)

**Depends on:** Phase 1

### Phase 3: Update notebook

**Files to modify:**
- `examples/quickstart.ipynb` — replace manual plot cells with `plot_study()`

**Steps:**

1. Replace the multi-cell plot section with a single `plot_study()` call:
   ```python
   fig = hpo.plot_study(study)
   fig.savefig("study_overview.png", dpi=150, bbox_inches="tight")
   ```

2. Add a markdown cell explaining:
   - `plot_study()` auto-detects 2 vs 3 objectives
   - Individual plot functions are available for customization
   - For 3-objective studies, show the alternative `plot_pareto_3d` / `plot_pareto_projections` calls

3. Keep one cell showing individual function usage for users who want customization:
   ```python
   fig, ax = plt.subplots()
   hpo.plot_pareto_front(study, ax=ax)
   ```

**Depends on:** Phase 2

## Verification & Validation

- **Automated**: `pytest tests/test_visualization.py -v` passes. `ruff check src/ tests/` clean.
- **Manual**:
  - Run `examples/quickstart.ipynb` end-to-end with a 2-objective study and confirm `plot_study()` produces a readable 2×2 panel
  - Create a 3-objective mock study in a scratch script and verify:
    - 3D Pareto scatter renders with correct rotation/labels
    - Projections show 3 sensible 2D views
    - Parallel coordinates shows lines connecting objectives
  - Confirm all individual plot functions still work standalone (backwards compatible)
  - Check that the 2×2 and 2×3 layouts don't have overlapping labels (`tight_layout` or `constrained_layout`)

## Dependencies

- matplotlib (already required — `mpl_toolkits.mplot3d` is included)
- optuna (already required)
- numpy (already required)

## Notes

_Living section — updated during implementation._

- This plan builds on the completed `plan-multi-metric-plots` which added the current 5 plot functions.
- The 3D scatter via `mplot3d` is intentionally basic — for publication-quality 3D Pareto surfaces users should use plotly or paraview.
- `plot_study()` is the main user-facing addition; the individual 3-obj functions exist for power users.

## Review Feedback

Reviewed in 1 iteration. 9 findings (2 blockers, 4 warnings, 3 suggestions).

**Blockers addressed:**
1. **`plot_study()` validation**: Added explicit `ValueError` for unsupported objective counts (1-obj, 4+-obj). Updated plan step 4.
2. **3D Pareto correctness**: Added explicit Pareto-set validation test to Phase 1 test plan. Documented that `study.best_trials` computes the multi-objective Pareto set.

**Warnings addressed:**
3. **3D color-mapping**: Specified `cmap="viridis"` with log-normalized param_count and labeled colorbar. Updated step 1.
4. **Return type clarity**: Added design decision row. `plot_study()` returns `Figure`; users access `fig.axes` for customization.
5. **Edge cases (NaN, missing metrics)**: Added explicit filter in `plot_pareto_3d` step 1 checking `len(t.values)`, `None`, and `math.isnan`. Added NaN edge-case test.
6. **Parallel coordinates order**: Added `metric_order` parameter to step 3; defaults to `_objective_column_names(study)`.

**Suggestions noted (no plan changes needed):**
7. Phase 2 polish scope — made explicit with per-function list and color constant names.
8. `ax` parameter consistency — all existing functions already accept `ax=None`; new functions follow same pattern.
9. 2×3 layout — added ASCII diagram to step 5 showing GridSpec assignment.

**User approval round (2026-03-15):**
- `plot_pareto_3d`: Approved with change — added `cost_display` kwarg (`'color'`|`'size'`) instead of hardcoded color mapping
- `plot_pareto_projections`: Approved with change — omitted 3rd dimension shown via `cost_display` kwarg
- `plot_parallel_coordinates`: Approved after explanation of purpose
- `plot_study`: Approved as-is (ValueError for 1/4+ obj)
- Color scheme: Match BayesFlow `#132a70` primary color; define in `results/_colors.py` for cross-package reuse
- 3-obj GridSpec layout: Approved
- Phase 2 polish: Approved
- Phase 3 notebook: Approved
