# plot_study() Redesign Spec

## Problem

The current `plot_study()` has several issues:

1. **Incorrect axes**: `plot_pareto_front()` plots objective[0] vs `param_count` instead of actual study objectives against each other.
2. **Incorrect colors**: color mappings don't reflect the right variables; colormap direction can be misleading.
3. **Single-objective history**: `plot_optimization_history()` only shows the first objective; multi-objective studies need visibility into all objectives.
4. **Rigid 2x2 layout**: fixed grid doesn't adapt to the number of objectives.
5. **No 3rd dimension encoding**: 3-objective studies lose information when projected to 2D.

## Design

### Architecture

```
plot_study()                           # thin orchestrator: stacks 3 rows
  ├── plot_pareto_front()              # Row 0: all pairwise objective projections
  ├── plot_optimization_history()      # Row 1: one best-so-far panel per objective
  └── plot_param_importance()          # Row 2: one importance panel per objective
```

Each sub-plot function operates in **dual mode**:

- **Standalone** (no axes passed): creates its own `Figure` with auto-grid layout
- **Embedded** (axes array passed): draws into provided axes

### `plot_study()` — Orchestrator

```python
def plot_study(
    study: optuna.Study,
    *,
    third_dim: str = "color",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
```

- **Raises `ValueError`** for <2 objectives (same as current).
- **Raises `ValueError`** for >3 objectives, with message directing users to standalone functions (`plot_pareto_front()`, `plot_optimization_history()`, `plot_param_importance()`) which handle arbitrary objective counts via `max_cols`.
- Creates a `matplotlib.gridspec.GridSpec` with 3 rows and `n_cols` columns.
- Passes pre-allocated axes arrays to sub-plot functions (embedded mode).
- Drops the importance row (shrinks to 2 rows) if all importance panels fail.
- Auto figsize: `(5 * n_cols, 4.5 * n_rows)` where `n_cols = max(n_pairs, n_obj)`.

**Grid layout:**

| Objectives | n_pairs | n_cols | Row 0 (Pareto) | Row 1 (History) | Row 2 (Importance) |
|-----------|---------|--------|----------------|-----------------|-------------------|
| 2 | 1 | 2 | 1 panel, centered | 2 panels | 2 panels |
| 3 | 3 | 3 | 3 panels | 3 panels | 3 panels |

Centering: when a row has fewer panels than `n_cols`, the panel spans multiple `GridSpec` columns to center it. For the 2-obj case (1 Pareto panel in a 2-col grid), the single panel spans both columns.

### `plot_pareto_front()` — Reworked

```python
def plot_pareto_front(
    study: optuna.Study,
    axes: Any | None = None,
    *,
    third_dim: str = "color",   # "color" | "size" | "none"
    max_cols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> Any:  # Figure (standalone) or axes array (embedded)
```

**Behavior:**

- Generates **all pairwise 2D projections** of the study's objectives.
- For each pair `(i, j)`, plots `objective[i]` on X vs `objective[j]` on Y.
- Axis labels are derived from `_objective_column_names(study)` (the actual study metric names).
- **3rd dimension encoding** (`third_dim`): when ≥3 objectives, the omitted objective for each pair is encoded as:
  - `"color"`: colormap `viridis_r` (dark purple = low/good, yellow = high/bad). Includes colorbar labeled with the omitted objective name.
  - `"size"`: marker size mapped linearly from the omitted objective values.
  - `"none"`: uniform markers, no encoding.
- **Pareto overlay** (two layers, mirroring existing `plot_pareto_projections()` pattern):
  1. **2D step line**: computed per-projection via `_draw_pareto_overlay()` / `_pareto_front_2d()` — shows the non-dominated front in the 2D view.
  2. **Study-level Pareto markers**: `study.best_trials` highlighted as accent-colored star markers — these are Pareto-optimal in the full N-dimensional objective space (may not be non-dominated in the 2D projection).
- Non-Pareto trials: `c.PRIMARY` color with `c.ALPHA_TRIAL` transparency.
- **Colormap note**: uses `viridis_r` (deliberate change from existing `plot_pareto_projections()` which uses `viridis`), so dark = low/good for minimization objectives.

**Standalone mode** (no `axes`): creates a `Figure` with `max_cols`-wrapped grid.
**Embedded mode** (`axes` provided): draws into the axes array.

**Replaces** the current `plot_pareto_front()` which plotted obj[0] vs `param_count`.

### `plot_optimization_history()` — Reworked

```python
def plot_optimization_history(
    study: optuna.Study,
    axes: Any | None = None,
    *,
    max_cols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> Any:
```

**Behavior:**

- Creates **one panel per objective**.
- Each panel shows a **best-so-far step line only** (no scatter points).
- Step line color: `c.BEST_LINE`.
- Best-so-far computation respects `study.directions[i]`: uses `min` for `MINIMIZE`, `max` for `MAXIMIZE`.
- Y-axis label and title: the actual objective name from `_objective_column_names(study)`.
- X-axis: trial number (chronological order).

**Standalone mode**: auto-grid with `max_cols` wrapping.
**Embedded mode**: draws into provided axes array.

### `plot_param_importance()` — Extended

```python
def plot_param_importance(
    study: optuna.Study,
    axes: Any | None = None,
    top_k: int = 10,
    *,
    max_cols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> Any:
```

**Behavior:**

- Creates **one importance bar chart per objective**.
- For each objective, uses `optuna.importance.get_param_importances()` with that objective as the target.
- **Graceful degradation**: if importance computation fails for an objective, that panel shows "Importance unavailable" placeholder text.
- Returns `None` if **all** panels fail (signals `plot_study()` to drop the row).
- Bar color: `c.PRIMARY`.

**Standalone mode**: auto-grid with `max_cols` wrapping.
**Embedded mode**: draws into provided axes array.

### Other Standalone Functions Updated

**`plot_pareto_projections()`**: add `max_cols: int = 3` parameter for row wrapping.

**`plot_metric_panels()`**: add `max_cols: int = 3` parameter for row wrapping.

These functions remain available for direct use but are **not called by `plot_study()`**.

### Removed from `plot_study()` Grid

- `plot_metric_scatter()` — still available as standalone function, unchanged.
- `plot_metric_panels()` — still available as standalone function, gets `max_cols`.

### Color Scheme

| Element | Color | Source |
|---------|-------|--------|
| Trial scatter | `c.PRIMARY` (#132a70) | `_colors.py` |
| Trial transparency | `c.ALPHA_TRIAL` (0.4) | `_colors.py` |
| Pareto markers | `c.ACCENT` (red), star, size 90 | `_colors.py` |
| Best-so-far line | `c.BEST_LINE` (#E74C3C) | `_colors.py` |
| 3rd dim colormap | `viridis_r` (dark = good/low) | matplotlib |
| Reference lines | `c.SECONDARY` (gray) | `_colors.py` |

### Parameters Removed

- `select_by` is removed from `plot_study()`. It served no purpose in the new design where each objective gets its own panels. It remains on `plot_parallel_coordinates()` where it's still needed.
- `metrics` parameter removed from `plot_study()` (metric panels no longer in grid).

## Breaking Changes

1. **`plot_pareto_front()` signature changes**: now takes `axes` (array) instead of `ax` (single), plus new `third_dim`/`max_cols` params. Old `xlabel`/`ylabel` params removed (auto-derived). Function now auto-generates all pairwise projections.
2. **`plot_optimization_history()` signature changes**: now takes `axes` (array) instead of `ax` (single), plus `max_cols`. No longer shows scatter points.
3. **`plot_param_importance()` signature changes**: now takes `axes` (array) instead of `ax` (single), plus `max_cols`. Single-objective targeting via `target_name` removed in favor of auto-per-objective behavior.
4. **`plot_study()` signature**: `select_by` and `metrics` params removed; `third_dim` and `figsize` added.
5. **`plot_study()` raises `ValueError`** for >3 objectives (previously silently used first 2).

## Unchanged Functions

The following functions are **not modified** by this redesign:

- `plot_pareto_3d()` — standalone 3D scatter, unchanged
- `plot_parallel_coordinates()` — standalone parallel coords, unchanged (still uses its own `select_by`)
- `plot_metric_scatter()` — standalone 2-metric scatter, unchanged

## Edge Cases

- **Empty study** (no completed trials): each sub-plot function shows placeholder text on the provided axes and returns them normally.
- **2-objective study with `third_dim`**: `third_dim` is ignored (no 3rd objective to encode).
- **All importance panels fail**: importance row is dropped, figure shrinks to 2 rows.
- **4+ objectives**: `plot_study()` raises with helpful message; users call standalone functions directly.
