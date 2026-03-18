# bayesflow-hpo — Project TODOs

Tracked items for ongoing development. Updated by contributors and Claude Code sessions.

Items are grouped into packages of related work that should be shipped together.
Suggested execution order: F → C → D → A → B → G → E.

## Open

### Package A: Sampler & Pruner Presets

Tightly coupled — presets need researched defaults, and pruning warmup
depends on sampler config. QMC warm-up composes with sampler presets.

#### Add named sampler presets to create_study()

Add string-based sampler selection (`"tpe"`, `"gp"`, `"nsga2"`, `"nsga3"`,
`"auto"`, `"random"`) to `create_study()` alongside existing object parameter.
Each preset wires sensible defaults and auto-wires `constraints_func` when
`budget_aware=True`. See `HPO-BENCHMARK-PAPER_PLAN.md` in bayesflow_projects
for full design.

#### Research: detailed sampler preset defaults

For each of the 6 sampler presets, research and document optimal default
parameters:
- GP: internal normalization with conditional spaces, `n_startup_trials`
- NSGA-II/III: population size heuristics (function of search space dim)
- Auto: verify it selects sensibly for BayesFlow HPO workloads
- Document each sampler's internal HP scaling behavior (confirms no external
  transform layer needed)

#### Add pruner string presets to create_study()

Add `pruner="none"` (NopPruner) and `pruner="median"` (current default) as
convenience presets.

#### Align pruning warmup with sampler startup

Auto-align `PeriodicValidationCallback.n_startup_trials` with the sampler's
startup count. Current default (5) is too few for `NetworkSelectionSpace`
(25 architecture combos). Default to `sampler.n_startup_trials` (25 for TPE,
10 for GP, population_size for NSGA-II). User-overridable.

#### Add QMC warm-up option to optimize()

Add `qmc_startup_trials: int = 0` parameter. When > 0, first N trials use
`QMCSampler` (Sobol sequences), then swap to the main sampler. Composes with
any sampler preset.

#### Research: QMC warm-up effectiveness

Empirically test whether QMC startup improves convergence compared to random
startup, especially for GP and TPE. May become a secondary finding in the
HPO benchmark paper.

---

### Package B: Trial Selection & Results

Lexicographic selection builds on Pareto extraction; `select_by` bounds
check is in the same code path.

#### Add lexicographic-Pareto trial selection

Add `select_best_trial()` to `results/extraction.py` and integrate into
`best_config()` via an optional `priorities` parameter. Two-phase algorithm:
(1) satisficing — filter by priority thresholds in order, (2) Pareto selection
over remaining metrics. Direction inferred from `study.directions` for
objectives, explicit for user_attrs. See `HPO-BENCHMARK-PAPER_PLAN.md` in
bayesflow_projects for full design.

#### Add `select_by` bounds check (#22)

Validate `0 <= select_by < len(study.directions)` at entry of
`get_pareto_trials()` and `summarize_study()`.
**File:** `results/extraction.py:229, 251`

---

### Package C: API Consolidation & Usability

Naming alignment, explicit key overrides, and silent-failure fixes are all
about the public API surface.

#### Consolidate API naming against BayesFlow

Align parameter/method names with BayesFlow 2.x conventions.
Example: `batches_per_epoch` → `num_batches`.

#### Accept explicit `param_keys`/`data_keys` in optimize() (#5)

Add optional `param_keys` / `data_keys` parameters that override
adapter inference when provided.
**File:** `api.py:96-362`

#### Add debug logging to `infer_keys_from_adapter` (#4)

When the adapter has no `transforms` attribute, log at `DEBUG` level
so the inference path is visible.
**File:** `api.py:63-65`

#### Fix `normalize_param_count` edge case (#3)

Document the intended invariant and add a guard for
`max_count <= min_count` that returns `0.5` (neutral) instead of `0.0`.
**File:** `objectives.py:92-99`

---

### Package D: `optimize()` Refactor

Extract helpers first, then the tracking dict fix is testable against
the cleaner code.

#### Extract helpers from `optimize()` (~270 lines) (#8)

Extract `_setup_validation_data()` and `_build_objective()` to improve
readability and testability. No change to the public API.
**File:** `api.py:96-362`

#### Deduplicate builder registration loop (#9)

Extract a `_register_with_aliases(registry_fn, name, builder, aliases)`
helper to remove duplicated alias logic.
**File:** `registration.py:55-58, 90-92`

#### `_TrackingDict` — track `items()`/`values()` or document (#10)

A builder using `for k, v in hparams.items()` won't mark keys as accessed,
causing false-positive unused-key warnings. Either override iteration methods
or document the limitation in `check_pipeline`'s docstring.
**File:** `pipeline.py:51-84`

---

### Package E: Validation & Metrics

C2ST reimplementation and inference key validation are both in the
validation subsystem.

#### Reimplement C2ST as a multivariate posterior two-sample test

The `sbc_c2st` metric was removed because applying C2ST (Lopez-Paz &
Oquab 2017) to 1D SBC rank integers is theoretically redundant with
KS and chi-squared tests — a random forest on a single integer feature
is just a noisy histogram comparison.

A proper reimplementation should follow the standard SBI approach
(Lueckmann et al. 2021, sbibm): train a classifier on **multivariate
posterior draws** vs reference samples (not 1D ranks). This would
require a different metric signature that receives full posterior
arrays rather than per-parameter slices.

Design considerations:
- Metric signature: needs `(draws: [n_sims, n_samples, n_params],
  true_values: [n_sims, n_params])` — different from the current
  per-parameter `MetricFn` convention
- Classifier choice: MLP (sbibm default) or RF; consider
  probability-based statistic (L-C2ST, Linhart et al. 2023) instead
  of binarized accuracy
- Reference samples: requires either a reference posterior method or
  prior predictive draws (different from SBC rank-based null)
- Keep as optional metric with `requires="sklearn"` extra

#### Validate `data_keys` exist before `sample()` (#18)

`inference.py` silently skips missing data keys via dict comprehension.
Validate all `data_keys` exist in `sim_data` before calling `sample()`.
**File:** `validation/inference.py:32`

#### Research: multi-objective pruning improvement

Investigate whether Hyperband/SHA ideas can be adapted to multi-objective mode
(e.g., using the geometric mean score from `PeriodicValidationCallback`).
Current custom median pruner works but may be suboptimal.

---

### Package F: Testing Gaps

Independent test additions. Ideally done first to establish a safety net
before the other packages.

#### Test `warm_start_study` (#23)

**File:** `optimization/study.py:169-218`
Warm-start logic (ranking, trial copying, edge cases) has no unit tests.

#### Test `_training_loss_fallback` (#24)

**File:** `optimization/objective.py:281-334`
The validation-failure fallback path is critical but not directly tested.

#### Test `load_validation_dataset` round-trip (#25)

**File:** `validation/data.py:214-244`
`save_validation_dataset` → `load_validation_dataset` round-trip is not tested.

#### Test `make_condition_grid` edge cases (#26)

**File:** `validation/data.py:149-182`
`logspace` and mixed-mode grids are not tested.

---

### Package G: Search Space Gaps

#### Add `mlp_width` and `bidirectional` to `FusionTransformerSpace` (#29)

`SetTransformerSpace` and `TimeSeriesTransformerSpace` expose `mlp_width`;
`TimeSeriesNetworkSpace` exposes `bidirectional`. `FusionTransformerSpace`
has neither — inconsistent across transformer-based summary spaces.
**File:** `search_spaces/summary/fusion_transformer.py`

#### Validate `IntDimension` rejects `log=True` + `step` (#28)

Optuna's `trial.suggest_int()` raises `ValueError` when both `log=True`
and `step` (other than 1) are set. Add validation in
`BaseSearchSpace.sample()` or `IntDimension.__post_init__`.
**File:** `search_spaces/base.py:49`

---

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
