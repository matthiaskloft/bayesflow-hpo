# Plan: Multi-Metric Visualization Improvements

**Created**: 2026-03-11
**Author**: Claude

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-11 | |
| Phase 1: New plot functions in visualization.py | MERGED | 2026-03-11 | PR #5 merged |
| Phase 2: Update notebook to use new API | MERGED | 2026-03-11 | PR #8 merged |
| Ship | DONE | 2026-03-11 | All phases shipped |

## Summary

**Motivation**: The visualization API was designed before multi-metric objectives were added. `plot_pareto_front` hardcodes "Calibration error" as the x-axis label even when the objective is `mean(calibration_error+nrmse)`. The optimization history plot is ad-hoc code in the notebook. There are no plots for per-metric trade-offs, per-metric-vs-param_count panels, or iso-mean contour lines — all essential for understanding multi-metric HPO results.

**Outcome**: Users get six plot functions covering all key multi-metric views:
- `plot_pareto_front` — fixed labeling
- `plot_optimization_history` — convergence curve
- `plot_metric_scatter` — pairwise metric scatter with iso-mean lines and 2D Pareto front
- `plot_metric_panels` — per-metric vs param_count subplots
- `plot_param_importance` — now targetable to any metric

The quickstart notebook demonstrates all of them.

## Assumptions

- `_objective_column_names()` reliably resolves metric names across Optuna versions (checks `study.metric_names` and `study._metric_names` fallback).
- Individual per-metric values are stored in `trial.user_attrs` by `GenericObjective` for all trained (non-rejected) trials.
- matplotlib is the only plotting dependency (already required).

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Where to put `plot_optimization_history` | (a) Notebook-only, (b) `visualization.py` | (b) | Reusable; removes ad-hoc code from notebook |
| Per-metric scatter vs. parameterize pareto | (a) New `plot_metric_scatter`, (b) Extend `plot_pareto_front` | (a) | Pareto front is objective-vs-param_count; metric scatter is a different concept |
| Fix `plot_pareto_front` x-axis label | (a) From `metric_names`, (b) Accept `xlabel` param | Both | Auto-derive, allow override |
| Per-metric importance | (a) Multi-panel, (b) `target_name` param | (b) | Small API surface; users call it in a loop |
| Iso-mean lines in metric scatter | (a) Always show, (b) Auto-detect "mean" mode from `metric_names`, (c) Explicit param | (b) with (c) override | Auto-detect `"mean("` prefix in `metric_names[0]`; `show_iso_lines` param for override |
| Per-metric-vs-param_count panels | (a) Extend `plot_pareto_front`, (b) New `plot_metric_panels` | (b) | Keep `plot_pareto_front` focused on study objectives; new function for decomposed view |
| Pareto front in metric scatter | (a) Use `study.best_trials`, (b) Compute true 2D Pareto | (b) | `study.best_trials` reflects study-objective Pareto, not per-metric Pareto |

## Scope

### In Scope

- Fix `plot_pareto_front` to derive x-axis label from `_objective_column_names()`
- Add `plot_optimization_history(study)` — convergence with best-so-far line
- Add `plot_metric_scatter(study, x_metric, y_metric)` — scatter with:
  - True 2D Pareto front (step-line connecting non-dominated points)
  - Iso-mean lines when "mean" mode detected (diagonal `y = 2c - x` contours)
  - Color points by mean value for visual gradient
- Add `plot_metric_panels(study, metrics, ...)` — one subplot per metric vs param_count
- Add `target_name` to `plot_param_importance`
- Export new functions; update `__init__.py` and `__all__`
- Tests for all new/modified functions
- Update quickstart notebook section 3

### Out of Scope

- Interactive/Plotly visualizations
- Radar/parallel-coordinates for top-k comparison
- GP surrogate extraction from Optuna samplers
- Custom color themes

## Implementation Plan

### Phase 1: New plot functions in visualization.py

**Files to modify:**
- `src/bayesflow_hpo/results/visualization.py` — all plot changes
- `src/bayesflow_hpo/results/__init__.py` — export new functions
- `src/bayesflow_hpo/__init__.py` — export new functions + `__all__`
- `tests/test_visualization.py` (create) — unit tests

**Steps:**

1. **Fix `plot_pareto_front`**: Use `_objective_column_names(study)[0]` for x-axis label. Add optional `xlabel`/`ylabel` override params.

2. **Add `plot_optimization_history(study, ax=None)`**:
   - Filter to trained trials (exclude `rejected_reason`).
   - Scatter `t.values[0]` per trial; step-line for best-so-far.
   - Y-axis label from `_objective_column_names(study)[0]`.

3. **Add `_pareto_front_2d(xs, ys)` helper**:
   - Given two lists of values (both minimize), return indices of non-dominated points.
   - Sort by x, sweep for monotonically decreasing y.
   - Used by `plot_metric_scatter` and `plot_metric_panels`.

4. **Add `plot_metric_scatter(study, x_metric, y_metric, ax=None, show_iso_lines=None)`**:
   - Read metrics from `trial.user_attrs`; skip trials missing either metric (log warning).
   - Color points by `mean(x, y)` using a sequential colormap.
   - Compute true 2D Pareto front via `_pareto_front_2d`; draw as red step-line with star markers.
   - **Iso-mean lines**: auto-detect "mean" mode by checking if `_objective_column_names(study)[0]` starts with `"mean("`. When active, draw dashed diagonal lines at best/median/worst mean values. Override with `show_iso_lines` param.

5. **Add `plot_metric_panels(study, metrics=None, ax=None)`**:
   - Default `metrics` from `trial.user_attrs` intersection with known metric keys (`calibration_error`, `nrmse`, `correlation`, etc.).
   - Create 1×N subplots (or accept pre-created axes array).
   - Each panel: metric (y) vs param_count (x, log scale), with 2D Pareto front line.
   - Shared x-axis label "Parameter count".

6. **Add `target_name` to `plot_param_importance`**: Build target function `lambda t: t.user_attrs.get(target_name, float("inf"))` filtering out rejected trials. Update title to show metric name.

7. **Export**: Add `plot_optimization_history`, `plot_metric_scatter`, `plot_metric_panels` to `results/__init__.py` and `__init__.py` + `__all__`.

8. **Tests**: Create `tests/test_visualization.py`:
   - `make_mock_study()` fixture: 5 trials with realistic user_attrs, multi-objective values, one rejected trial.
   - Test each function returns `matplotlib.axes.Axes`.
   - Test label auto-derivation.
   - Test `_pareto_front_2d` correctness on known inputs.
   - Test edge cases: no trained trials, single-objective study, missing metrics.

**Depends on:** None

### Phase 2: Update notebook to use new API

**Files to modify:**
- `examples/quickstart.ipynb` — section 3 ("Inspect Results")

**Steps:**

1. Restructure section 3 with two figure blocks:

   **Figure 1** (2×2): Overview
   - (0,0) Pareto front — `hpo.plot_pareto_front(study)`
   - (0,1) Metric scatter (cal_error vs nrmse) — `hpo.plot_metric_scatter(study, "calibration_error", "nrmse")` with auto iso-mean lines
   - (1,0) Optimization history — `hpo.plot_optimization_history(study)`
   - (1,1) Param importance — `hpo.plot_param_importance(study)`

   **Figure 2** (1×2): Per-metric panels
   - `hpo.plot_metric_panels(study, metrics=["calibration_error", "nrmse"])`

2. Update markdown cells:
   - Explain iso-mean lines: "Dashed diagonals show iso-objective contours — trials on the same line achieve equal mean(cal_error, nrmse)."
   - Explain metric panels: "Per-metric views decompose the composite objective to show how each metric individually trades off against model size."

3. Re-run notebook to generate fresh output.

**Depends on:** Phase 1

## Verification & Validation

- **Automated**: `pytest tests/test_visualization.py -v` passes. `ruff check src/` clean.
- **Manual**: Run `examples/quickstart.ipynb` end-to-end and confirm:
  - Pareto front x-axis reads `"mean(calibration_error+nrmse)"` (auto-derived)
  - Metric scatter shows color gradient, 2D Pareto step-line, and iso-mean diagonals
  - Metric panels show per-metric vs param_count with individual Pareto fronts
  - Optimization history converges with correct y-label
  - All subplots readable at default figure sizes

## Dependencies

- matplotlib (already required)
- optuna (already required)
- numpy (already required)

## Notes

_Living section — updated during implementation._

### Phase 1 Implementation Notes (2026-03-11)
- Extracted `_trained_trials` helper to DRY repeated filtering logic.
- Replaced nested ternary in param count formatter with `_format_param_count` function.
- `top_k` kept as positional param in `plot_param_importance` to avoid breaking callers.
- Metric auto-detection in `plot_metric_panels` checks across all trials (not just first).
- Colorbar label in `plot_metric_scatter` shows `mean(x_metric, y_metric)` instead of bare "mean".
- Added autouse `_close_figures` fixture for test cleanup safety.

## Review Feedback

Reviewed in 1 iteration (prior session). Key findings addressed:
- **Metric name resolution**: All functions use `_objective_column_names()` helper, which handles both Optuna >=4.x and fallback.
- **Missing user_attrs**: `plot_metric_scatter` and `plot_metric_panels` filter trials with missing metrics and log warnings.
- **Pareto semantics**: `plot_metric_scatter` computes true 2D Pareto independently, not from `study.best_trials`.
- **Rejected trials in importance**: `target_name` handler returns `inf` for rejected trials.
- Added `plot_metric_panels` for per-metric vs param_count panels (user request).
- Added iso-mean line support for "mean" mode detection (user request).
