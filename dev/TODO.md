# bayesflow-hpo — Project TODOs

Tracked items for ongoing development. Updated by contributors and Claude Code sessions.

## Open

### 1. Prettify `summarize_study()` output
**File:** `results/extraction.py`

Current output is too wide and prints raw floats (e.g. `0.05454749016213018`).
- Round floats contextually: dropout to 2 decimals, learning rate to scientific notation or 4 sig figs, widths/depths as integers
- Tighten the top-k leaderboard: narrower columns, abbreviate long metric names
- Keep it readable in an 80-char terminal

### 2. New `trial_table()` function for full trial export
**File:** `results/extraction.py` (new function)

A function that returns a formatted DataFrame of the k best trials (default = all),
including objective values, all search space hyperparameters, and optionally
further user-attribute metrics. Must be `.to_csv()`-savable.

```python
def trial_table(
    study: optuna.Study,
    top_k: int | None = None,       # None = all trials
    select_by: int = 0,             # objective index to rank by
    metrics: list[str] | None = None,  # extra user_attrs to include
    trained_only: bool = True,
) -> pd.DataFrame
```

Differences from `trials_to_dataframe()`: ranked, top-k filtered, includes
objective columns alongside hyperparameters, and rounds floats for display.

### 3. Split reporting into focused functions
**File:** `results/extraction.py`

`summarize_study()` currently mixes trial counts, Pareto info, leaderboard, and
hyperparameters in one 60-char-wide text block. Proposal:

| Function | Purpose |
|---|---|
| `summarize_study()` | **Keep** — compact overview: trial counts + best trial + its objectives (no leaderboard, no hyperparameters) |
| `trial_table()` | **New** (TODO #2) — full ranked table of top-k trials with objectives, hyperparams, metrics; CSV-savable |
| `best_config()` | **New** — returns the hyperparameter dict of a specific trial (or best by `select_by`), pretty-printed with rounding |
| `compare_trials()` | **New** — side-by-side comparison of 2–5 specific trials (objectives + hyperparams diff) |

This keeps `summarize_study()` as the quick "how did the study go?" answer,
while detailed inspection uses dedicated functions.

### 4. Rework plotting for 2D and 3D objectives
**File:** `results/visualization.py`

Current plots assume 2 objectives (quality metric + param count). We need two
well-designed standard scenarios:

**a) 2-objective (e.g. calibration_error + param_count):**
- Pareto front scatter (exists, needs polish)
- Optimization history (exists)
- Param importance (exists)

**b) 3-objective (e.g. calibration_error + nrmse + param_count):**
- 3D Pareto surface or paired 2D Pareto projections
- Metric-vs-metric scatter with param_count as color/size
- Parallel coordinates plot for objectives

Other objective counts (1, 4+) are **out of scope**.

Review each existing plot function and decide: keep / rework / remove.
Add a convenience `plot_study(study)` that auto-detects 2D vs 3D and produces
the standard panel.

### 5. Review search space defaults against BayesFlow
**Files:** `search_spaces/summary/deep_set.py`, `search_spaces/inference/flow_matching.py`, all search spaces

Audit from quickstart run output:
```
ds_dropout   : 0.05454749016213018   (continuous float, no step)
fm_dropout   : 0.004116898859160489  (continuous float, no step)
ds_summary_dim: 63                   (Int, no step — should it be powers of 2 or step=8?)
```

Known issues found:
- **`fm_subnet_depth`** high=4, but BayesFlow FlowMatching TIME_MLP default is depth=5. The search space can never reach the framework default. **Fix: raise high to 5 or 6.**
- **Dropout dimensions** (ds, fm, cf, etc.): continuous float produces ugly values. Consider adding `step=0.01` or `step=0.05` for cleaner output and faster convergence.
- **`ds_summary_dim`**: range 4–64 with no step. Consider `step=8` or powers of 2 for more meaningful exploration.

Run a full audit across all search spaces (coupling_flow, diffusion, consistency,
set_transformer, fusion_transformer, time_series_*) to check every dimension
against BayesFlow source defaults.

### 6. Quickstart: add model selection and retraining workflow
**File:** `examples/quickstart.ipynb`

The quickstart ends after `optimize()` and `summarize_study()`. It's missing the
critical final step: choosing a trial from the results and retraining it as the
production model.

Add cells for:
1. Inspect study results (use `trial_table()` once available, or `trials_to_dataframe()`)
2. Pick a trial (from Pareto front or top-k)
3. Reconstruct the approximator from the trial's hyperparameters
4. Retrain with full budget (more epochs, no early stopping)
5. Save the final workflow with `save_workflow_with_metadata()`

## Done

### Remove multi_objective.ipynb (2026-03-15)
Removed the `examples/multi_objective.ipynb` notebook and updated README examples table.

### Dev docs: BayesFlow fit() compatibility note (2026-03-15)
Updated developer-facing docs to record BayesFlow 2.0.8 fit() keyword behavior:
- Added quickstart guidance using a compatibility `train_fn` that maps `batches_per_epoch` -> `num_batches`.
- Updated optimization/index docs to match the current approximator-based `train_fn` signature and default training path.

### Trial counting docs & reporting (2026-03-15)
Clarified trial counting for users:
- Split progress output into 4 categories: trained, rejected, failed, pruned (dropped redundant "total")
- Added startup log message explaining what each category means
- Added Notes section to `optimize()` docstring documenting the full trial lifecycle and safety caps
- Added `_count_budget_rejected()` and `_count_failed()` helpers in `study.py`

### Quickstart runnable example (2026-03-15)
Fixed `examples/quickstart.ipynb` to run end-to-end from a fresh clone:
- Removed stale kwargs (`param_keys`, `data_keys`, `validation_data`) that no longer exist in `optimize()` API
- Changed `n_trials=0, resume=True` → `n_trials=5, storage=None`
- Updated markdown to reflect that key inference and validation data generation happen inside `optimize()`

### Review CI checks (2026-03-15)
PR #9 (stale revert) was already closed. CI passes on main (lint + test 3.11/3.12/3.13). No action needed.

### Enhance code docs (2026-03-12)
Added/enhanced module-level docstrings on all 42 .py files, all `build()` methods, private helpers, and design-decision comments. All 233 tests pass, ruff clean.
