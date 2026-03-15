# bayesflow-hpo — Project TODOs

Tracked items for ongoing development. Updated by contributors and Claude Code sessions.

## Open

### 1–3. Reporting bundle: `trial_table()`, `best_config()`, `compare_trials()`, slim `summarize_study()`
**Plan:** [`dev/plans/plan-reporting-bundle.md`](plans/plan-reporting-bundle.md)
**Files:** `results/extraction.py`, `results/__init__.py`, `__init__.py`, `tests/test_results/test_extraction.py`
**Status:** Planned (3 phases)

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
**Status: mostly done — see Done section. Remaining minor items below.**

Remaining optional improvements (not blocking):
- **Dropout dimensions** (ds, fm, cf, etc.): continuous float produces ugly values (e.g. `0.05454749016213018`). Consider adding `step=0.01` or `step=0.05` for cleaner output and faster convergence. Left as-is because continuous sampling is standard Optuna practice for regularization parameters.
- **`cf_permutation`** choices are `["random", "orthogonal"]` but BayesFlow also accepts `"swap"` and `None`. Missing options narrow the search but are rarely useful.
- **Subnet widths** all cap at 256 (the BayesFlow default). Cannot explore wider architectures. This is intentional to keep search spaces tractable.

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

### Review search space defaults against BayesFlow (2026-03-15)
Full audit of all 11 search spaces against BayesFlow 2.x source defaults. Fixes applied:
- **`subnet_depth` high 4→6** in FlowMatchingSpace, DiffusionModelSpace, ConsistencyModelSpace, StableConsistencyModelSpace — BayesFlow `TIME_MLP_DEFAULT_CONFIG` uses 5 layers, so the old cap of 4 excluded the framework default
- **`tst_time_embed` choices**: replaced invalid `"sinusoidal"` (would raise `ValueError`) with valid BayesFlow options `["time2vec", "lstm", "gru"]`
- **`ds_summary_dim`**: added `step=4` for consistency with other summary network spaces (SetTransformer etc. use `step=8`)
- Updated docstrings in all changed search spaces and both docs files (`search_spaces.md`, `defaults.md`)

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
