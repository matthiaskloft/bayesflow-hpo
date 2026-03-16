# bayesflow-hpo — Project TODOs

Tracked items for ongoing development. Updated by contributors and Claude Code sessions.

## Open

No open items.

## Done

### Redesign plot_study() for multi-objective support (2026-03-16, PRs #43, #45)
Two-phase redesign of the visualization module:
- **Phase 1** (PR #43): Rewrote `plot_pareto_front()` (pairwise 2D projections with
  `third_dim` encoding), `plot_optimization_history()` (per-objective direction-aware
  step lines), `plot_param_importance()` (per-objective bar charts with graceful
  degradation). Added `max_cols` wrapping to `plot_pareto_projections()` and
  `plot_metric_panels()`. Added `_setup_grid()` shared helper for dual-mode axes.
- **Phase 2** (PR #45): Rewrote `plot_study()` as a 3-row GridSpec orchestrator
  (Pareto / History / Importance) supporting 2-3 objectives. Removed `_plot_study_2obj()`,
  `select_by`, and `metrics` params. >3 objectives raises `ValueError` with
  helpful message.

### Add Two Moons network selection example (2026-03-16, PR #44)
Added `examples/two_moons_optimization.ipynb` — demonstrates `NetworkSelectionSpace`
letting Optuna choose between CouplingFlow and FlowMatching on the Two Moons
benchmark. Fixed `optimize()` and `SelectionSpace.build()` for condition-only models.

### Rework inference time metric (2026-03-16, PRs #37-42)
Multi-phase rework of the inference time cost metric:
- Changed from ratio-based to seconds-per-dataset measurement
- Improved display: human-readable time units, per-metric logging
- Refactored checkpoint loading, plot naming, notebook rename
  (`quickstart.ipynb` → `getting_started.ipynb`)

### Fix fragile iso-line color assertion (2026-03-15, PR #36)
Replaced `line.get_color() in ("grey", "gray")` with
`to_hex(line.get_color()) == to_hex("gray")` for version-safe color
comparison. File: `tests/test_visualization.py`.

### Unify metric auto-detection in plot_metric_panels (2026-03-15)
Already resolved — `plot_metric_panels` calls `_get_metric_user_attrs()`
at line 369. No code change needed; moved from Open to Done.

### Rework plotting for 2D and 3D objectives (2026-03-15, PRs #32, #34, #35)
Added 3-objective support (`plot_pareto_3d`, `plot_pareto_projections`,
`plot_parallel_coordinates`) and `plot_study()` convenience entry point that
auto-detects 2 vs 3 objectives. Polished legends, axis formatting, and added
BayesFlow-aligned color palette (`_colors.py`). Updated quickstart to use
`plot_study()`.

### Quickstart: model selection & retraining workflow (2026-03-15, PR #33)
Added section 4 to `examples/quickstart.ipynb` with the full HPO-to-production workflow:
`trial_table()` → `best_config()` → `build_continuous_approximator()` → compile with
Adam/CosineDecay → retrain with full budget → `save_workflow_with_metadata()`.

### Review search space defaults against BayesFlow (2026-03-15, PR #29)
Full audit of all 11 search spaces against BayesFlow 2.x source defaults. Fixes applied:
- **`subnet_depth` high 4→6** in FlowMatchingSpace, DiffusionModelSpace, ConsistencyModelSpace, StableConsistencyModelSpace — BayesFlow `TIME_MLP_DEFAULT_CONFIG` uses 5 layers, so the old cap of 4 excluded the framework default
- **`tst_time_embed` choices**: replaced invalid `"sinusoidal"` (would raise `ValueError`) with valid BayesFlow options `["time2vec", "lstm", "gru"]`
- **`ds_summary_dim`**: added `step=4` for consistency with other summary network spaces (SetTransformer etc. use `step=8`)
- Updated docstrings in all changed search spaces and both docs files (`search_spaces.md`, `defaults.md`)

Remaining non-blocking items (intentionally left as-is):
- Dropout dimensions use continuous float (standard Optuna practice)
- `cf_permutation` omits `"swap"` and `None` (rarely useful)
- Subnet widths cap at 256 (intentional to keep search tractable)

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
