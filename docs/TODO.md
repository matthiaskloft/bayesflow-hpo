# bayesflow-hpo — Project TODOs

Tracked items for ongoing development. Updated by contributors and Claude Code sessions.

Items are grouped into packages of related work that should be shipped together.
Suggested execution order: D → A → B → G → E → H → I → J.
Package I (small API fixes) can be done at any time independently.

## Open

### Package I: Small API Fixes

Standalone fixes that don't depend on Package C.

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

#### Validate `data_keys` exist before `sample()` (#18)

`inference.py` silently skips missing data keys via dict comprehension.
Validate all `data_keys` exist in `sim_data` before calling `sample()`.
**File:** `validation/inference.py:32`

---

### Package D: `optimize()` Refactor

Extract helpers first, then the tracking dict fix is testable against
the cleaner code. Also standardize validation pipeline contracts that
touch `optimize()` internals.

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

#### Document `validate_fn` return contract

Must return `dict[str, float]` with at least the keys in `objective_metrics`;
missing keys get penalty substitution, extra keys are silently ignored.
Document in `optimize()` docstring and `ValidateFn` alias.
**Files:** `api.py:167-171`, `types.py:26`, `optimization/objective.py:129-152`

#### Fix timing semantics between default and custom validation paths

Default path extracts pure inference time from `result.timing["inference"]`;
custom `validate_fn` path measures total wall-clock (inference + metric
computation). Makes `cost_metric="inference_time"` non-comparable.
**Decision needed:** ask custom hooks to return timing dict (breaks
`ValidateFn`), or document the limitation, or drop inference time
normalization if samplers handle this internally.
**Files:** `optimization/objective.py:621-643`

#### Make `validation_data` required or expose `validate=False`

`ObjectiveConfig.validation_data` accepts `None` with a penalty
fallback path, but `optimize()` always generates the dataset.
**Decision needed:** make required (non-Optional) in ObjectiveConfig,
or expose `validate=False` in `optimize()`, or document as
internal-only.
**Files:** `optimization/objective.py:226,659-666`, `api.py:329-335`

#### Export `default_validate_fn`

Users writing custom `validate_fn` can't easily see the reference
implementation. Export it or add a usage example in the docstring.
**File:** `optimization/objective.py:95-126`

**Prime test case:** The bayesflow-irt IRT model (equivariant summary
networks, custom approximator) should work flawlessly through the
custom `build_approximator_fn` / `validate_fn` pathway. Use it as
the integration test when standardizing these contracts.

---

### Package A: Sampler & Pruner Presets

Tightly coupled — presets need researched defaults, and pruning warmup
depends on sampler config. QMC warm-up composes with sampler presets.
Includes a deep review of the current pruning implementation as a
prerequisite for pruner preset design.

#### Deep review: pruning feature

Comprehensive review of the pruning implementation, including literature
search for best practices and audit of the current code.

**Literature search:**
- Optuna's built-in pruners (MedianPruner, HyperbandPruner, SHA) and
  their applicability to multi-objective settings
- Multi-objective early stopping / pruning in the HPO literature
  (e.g., BOHB, DEHB, multi-fidelity multi-objective methods)
- Whether geometric mean of objectives is a sound composite score
  for pruning decisions, or if dominated-hypervolume-based pruning
  exists
- Warm-up heuristics: how many startup trials before pruning is
  reliable (current default: 5)

**Implementation audit:**
- `PeriodicValidationCallback` (validation_callback.py): review the
  custom median-based multi-objective pruning strategy (lines 31–82)
- Hard-coded intermediate metrics `["calibration_error", "nrmse"]`
  (line 28) — should these align with `objective_metrics`?
- `_should_prune_multi_objective()`: median threshold, reference
  trial selection (COMPLETE + non-rejected only), NaN/Inf handling
- Single-objective path: delegates to Optuna's `trial.report()` +
  study pruner — is this sufficient?
- Interaction with `pruning_n_startup_trials` and sampler startup
- Pruning score = `sqrt(nrmse * calibration_error)` — is geometric
  mean the right aggregation? What about user-defined objectives?

**Decision needed:** keep custom pruning, adopt a published
multi-objective pruning strategy, or make pluggable.
**Files:** `optimization/validation_callback.py`,
`optimization/objective.py:573-588`

#### Make intermediate validation configurable

`PeriodicValidationCallback` is always injected when `validation_data`
exists, hard-coded to `["calibration_error", "nrmse"]` regardless of
`objective_metrics`. No way to disable or customize.
**Decision needed:** accept `intermediate_metrics` in ObjectiveConfig,
provide disable flag, or document as intentional design.
**Files:** `optimization/validation_callback.py:28`,
`optimization/objective.py:573-588`

#### Make multi-objective pruning strategy pluggable

Custom median-based pruning is buried in the callback with no
configuration hooks.
**File:** `optimization/validation_callback.py:31-82`

#### Add named sampler presets to create_study()

Add string-based sampler selection (`"tpe"`, `"botorch"`, `"gp"`, `"nsga2"`,
`"nsga3"`, `"auto"`, `"random"`) to `create_study()` alongside existing object
parameter. Each preset wires sensible defaults and auto-wires `constraints_func`
when `budget_aware=True`. `"botorch"` requires `optuna-integration[botorch]`
(lazy import with clear error). See `HPO-BENCHMARK-PAPER_PLAN.md` in
bayesflow_projects for full design and `docs/references.md` for citations.

#### Research: detailed sampler preset defaults

For each of the 7 sampler presets, research and document optimal default
parameters:
- BoTorch: `n_startup_trials`, `device` (auto-detect GPU), categorical
  handling verification with NetworkSelectionSpace
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
10 for BoTorch/GP, population_size for NSGA-II). User-overridable.

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

### Package E: C2ST Metrics

New classifier two-sample test metrics for multivariate posterior
validation. Research-heavy; requires `sklearn` as optional dependency.

#### Background: C2ST variants for SBI

The `sbc_c2st` metric was removed because applying C2ST to 1D SBC rank
integers is theoretically redundant with KS and chi-squared tests — a
random forest on a single integer feature is just a noisy histogram
comparison.

Two C2ST variants are relevant for proper multivariate posterior
validation:

**Global C2ST** (López-Paz & Oquab, 2017) — the original classifier
two-sample test. A binary classifier is trained to discriminate samples
from two distributions P and Q; if classification accuracy significantly
exceeds chance, the distributions differ. In the SBI context, this
means comparing samples from the approximate posterior q_φ(θ|x_o) vs
the true posterior p(θ|x_o) for a fixed observation x_o. This requires
access to true posterior samples (e.g., from MCMC), which limits it to
settings where a reference posterior is available.

**L-C2ST** (Linhart et al., 2023) — a local variant that eliminates
the need for true posterior samples. Instead of comparing q(θ|x_o) vs
p(θ|x_o) at a fixed observation, L-C2ST works with joint samples: it
classifies (θ, x) pairs drawn from q(θ|x)p(x) [class 0] vs (θ, x)
pairs drawn from p(θ, x) [class 1]. The key insight (Linhart et al.,
2023, eq. 11) is that the optimal joint classifier d*(θ, x) equals the
optimal local classifier d*_x(θ), so the joint-sample approach recovers
local posterior diagnostics without needing true posterior samples.
Only requires samples from p(θ, x) — exactly what BayesFlow simulators
provide.

#### Implement L-C2ST (primary)

Implement L-C2ST for reference-free posterior validation.

Design considerations:
- Metric signature: needs `(draws: [n_sims, n_samples, n_params],
  true_values: [n_sims, n_params])` — different from the current
  per-parameter `MetricFn` convention
- Classifier: MLP (Linhart et al. recommend MLP for L-C2ST); returns
  probability-based statistic (not binarized accuracy)
- Training data: joint samples (θ, x) from the simulator — no
  reference posterior needed
- Keep as optional metric with `requires="sklearn"` extra
- Reference implementation available at
  https://github.com/JuliaLinhart/lc2st and integrated in the
  `sbi` Python package

#### Implement global C2ST (optional, requires reference posterior)

Implement standard C2ST (López-Paz & Oquab, 2017) for settings where
MCMC or other reference posterior samples are available. Useful for
post-hoc validation in the benchmark study where reference posteriors
can be computed for specific models (SDT, 2HTM, GVAR via Stan/brms).

Design considerations:
- Separate metric or mode flag on a shared C2ST implementation
- Inputs: approximate posterior samples + reference posterior samples
- Classifier: MLP or RF; binarized accuracy as test statistic
- Not usable as an HPO objective (requires MCMC per trial) — purely
  a post-hoc diagnostic

---

### Package H: Metric Constraints & Memory Auto-Detection

#### Add metric constraints on objective values

Add layered metric constraints to the optimization loop:

- **Soft constraints (feasibility-guided search):** Extend `_budget_constraints_func()`
  so trials violating user-specified metric thresholds (e.g., `calibration_error > 0.10`)
  are marked infeasible via Optuna's `constraints_func`. The sampler learns to avoid
  those regions while still considering them in its model.
- **Hard constraints (post-validation rejection):** After validation, check metrics
  against user-specified bounds. Violating trials are marked rejected (like budget
  rejection) — keeps the Pareto front clean.
- Both layers compose: hard thresholds reject clearly bad trials; soft constraints
  guide the sampler away from borderline regions.

Design considerations:
- New `MetricConstraints` config (or extend `ObjectiveConfig`) with per-metric
  upper/lower bounds
- Applies to objective metrics and optionally non-objective diagnostic metrics
  (e.g., SBC uniformity)
- Rejected-by-metric trials should not count toward `n_trained` (like budget rejection)

#### Auto-detect GPU memory budget

Add `auto_detect_memory_budget()` that queries available VRAM via
`torch.cuda.get_device_properties()` / `torch.cuda.mem_get_info()`, subtracts
a configurable safety margin (default 20%), and returns usable MB. Wire into
`optimize()` as `max_memory_mb="auto"` option alongside explicit numeric values.

Falls back gracefully when no GPU is available (use system RAM estimate or skip).

---

### Package I: Literature Audit

Standalone documentation/verification task. Can run at any point, but
best done after Packages C–G stabilize the codebase.

#### Audit all metrics and features against literature references

Systematic check that every built-in metric and major feature has a
verified literature reference in `docs/references.md`. For each:

- Confirm the implementation matches the method described in the paper
- Verify edge-case handling (e.g., degenerate inputs, numerical guards)
  is consistent with the original authors' recommendations
- Document any intentional deviations from the reference method and why
- Add missing references for metrics currently without citations
  (e.g., ECE, posterior contraction, z-score, coverage)

Scope: all 13 built-in metrics in `validation/registry.py`, the
budget-aware sampling design, the pruning strategy, and the
lexicographic-Pareto selection (once implemented).

---

## Done

### Package C: API Consolidation & Search Space Simplification (2026-03-21, PR #48)
Three-phase API consolidation and search space simplification:
- **Phase 1**: Replaced `enabled`/`include_optional` with `constant` field on
  `IntDimension`, `FloatDimension`, `CategoricalDimension`. Added `_UNSET`
  sentinel, `.constants` property on `BaseSearchSpace` and `CompositeSearchSpace`.
- **Phase 2**: Migrated all 10 concrete search spaces (5 inference + 5 summary)
  from `enabled=False` to `constant=<BF default>`. Expanded 6 abbreviated
  dimension names (`cf_use_actnorm`, `fm_use_optimal_transport`,
  `fm_time_power_law_alpha`, `ds_spectral_normalization`,
  `st_num_inducing_points`, `tst_time_embedding`). Simplified all `build()`
  methods to unconditionally read params.
- **Phase 3**: Renamed `batches_per_epoch` → `num_batches` across API, objective,
  pipeline, builders, docs, and examples. Fixed `default_train_fn` for BF 2.0.8+.

### Add edge-case tests — Package F (2026-03-21, PR #47)
16 new edge-case tests across `warm_start_study`, `_training_loss_fallback`,
`make_condition_grid`, and `load/save_validation_dataset`. Covers boundary
conditions, error paths, and mixed trial states.

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
- Added quickstart guidance using a compatibility `train_fn` that maps `num_batches` -> `num_batches`.
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

---

## Resolved Archive

<details>
<summary>Issues from the multi-objective pruning quality audit (2026-03-06) — all resolved</summary>

### ~~Broad `except Exception` in `_run_lightweight_validation` (pre-existing)~~ — RESOLVED

**File:** `optimization/validation_callback.py:186-215`

Now logs at `WARNING` level with `exc_info=True`, re-raises `TrialPruned`, and
tracks consecutive failures with a warning after 3.

### ~~Final validation not wrapped in try-except (pre-existing)~~ — RESOLVED

**File:** `optimization/objective.py` (step 8)

Falls back to training-loss-based objective values instead of penalty values.

### ~~`get_param_count` returns -1 on error~~ — RESOLVED

**File:** `objectives.py:43-61`

Now raises `ValueError` / `TypeError` instead of returning `-1`.

### ~~`api.py` delete-study catches only `KeyError`~~ — RESOLVED

**File:** `api.py:320-329`

Now catches generic `Exception` with `exc_info=True`.

### ~~`optimize_until` warning message doesn't mention pruning~~ — RESOLVED

**File:** `optimization/study.py:370-421`

Warning now includes failure breakdown with pruned count and guidance.

### ~~`OptunaReportCallback` stores per-epoch user attrs on every trial~~ — RESOLVED

**File:** `optimization/callbacks.py`

`report_frequency` is now configurable from `optimize()`.

### ~~`MedianPruner` docstring in `create_study` is misleading~~ — RESOLVED

**File:** `optimization/study.py`

Docstring now says "Single-objective only."

### ~~`optimize()` assumes `BasicWorkflow`~~ — RESOLVED

Resolved by custom approximator hooks (2026-03-14).

### ~~`run_validation_pipeline` assumes flat posterior shape~~ — RESOLVED

Resolved by the `validate_fn` hook.

### ~~ConsistencyModel `build()` casts `s0`, `s1`, `max_time` to `float` instead of `int`~~ — RESOLVED

**File:** `search_spaces/inference/consistency.py:123-130`

Changed `float(...)` to `int(...)` for `max_time`, `s0`, and `s1`,
matching their `IntDimension` declarations and BayesFlow's expected types.

</details>

<details>
<summary>Issues fixed in the package review PR (2026-03-14)</summary>

### ~~1. `_compile_for_compat` silently returns on total failure~~ — RESOLVED

**File:** `builders/workflow.py:40-67`

Now logs a warning when no compile signature succeeds.

### ~~2. `loguniform_int` can exceed upper bound after rounding~~ — RESOLVED

**File:** `utils.py:43`

Clamped result with `np.clip()`.  Also added `alpha > 0` validation.

### ~~6. `check_pipeline` uses very different defaults from `optimize()`~~ — RESOLVED

**File:** `pipeline.py:124-128`

Docstring now explains minimal defaults are intentional.

### ~~7. Missing `py.typed` marker~~ — RESOLVED

Created `src/bayesflow_hpo/py.typed` and added to `pyproject.toml`.

### ~~11. `TrainFn` callback list is unparameterized~~ — RESOLVED

**File:** `types.py:23`

Changed to `list[Any]`.

### ~~12. `_check_hook_arity` parameter `fn` is typed as `Any`~~ — RESOLVED

**File:** `pipeline.py:87`

Changed to `Callable[..., Any]`.

### ~~13. `builders/adapter.py` deprecation notice lacks version~~ — RESOLVED

Added "Deprecated since v0.2.0" with migration pointer.

### ~~14. `utils.py` `rng` parameter doesn't document `None` fallback~~ — RESOLVED

Docstring now describes `None` → global `np.random` behavior.

### ~~15. `PipelineError` has a one-line docstring~~ — RESOLVED

Expanded with common causes and debugging guidance.

### ~~16. CLAUDE.md architecture tree does not mention public API~~ — RESOLVED

Added "Public API" note to Key Patterns section.

### ~~17. `validation/pipeline.py` uses `time.time()`~~ — RESOLVED

Replaced with `time.perf_counter()`.

### ~~19. `make_coverage_metric` float-to-int truncation~~ — RESOLVED

Changed `int(level * 100)` to `round(level * 100)`.

</details>
