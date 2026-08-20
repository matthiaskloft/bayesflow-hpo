# bayesflow-hpo — Project TODOs

Tracked items for ongoing development. Updated by contributors and Claude Code sessions.

Items are grouped into packages of related work that should be shipped together.
Suggested execution order: I, then research follow-ups A2/A3.

## Open

### Near-term next steps

1. **Run the QMC warm-up benchmark** — `examples/qmc_warmup_benchmark.ipynb` is configured for final run (10 replications, 40 trials, 30 epochs). Execute and record results in docs.
2. Research the sampler preset defaults, especially BoTorch, GP, NSGA-II/III, and Auto.
3. ~~Run the literature audit to verify metrics and features against references.~~ (done — see Package I below)
4. Polish documentation and examples around the shipped features.
5. ~~Add docstring citations and inline literature comments.~~ (done — see Package I below)
6. ~~Revisit the `correlation` diagnostic: evaluate whether posterior medians are the more appropriate point summary for skewed posteriors.~~ (resolved — **keep the posterior mean**, do not switch to the median.) The posterior mean is the correlation-maximising estimator, not merely a convention: for any estimator `g`, the tower rule gives `Cov(g, t) = Cov(g, m)` with `m = E[t | D]`, so Cauchy-Schwarz bounds `corr(g, t)` by `sd(m) / sd(t)`, with equality iff `g` is affine in `m`. The median minimises L1 loss and has no such property, so substituting it would lower the achievable correlation — including for skewed posteriors, where the intuition that the median is "more representative" does not transfer to this metric. Verified empirically in `bayesflow-irt`'s Stan/NUTS ceiling benchmark, where posterior means satisfy the implied identity `NRMSE = sqrt(1 - r^2)` to within 0.002 at all four evaluation corners (see that repo's `docs/mcmc_ceiling_validation.md`).

   Worth adopting instead: report `nrmse - sqrt(1 - r^2)` alongside correlation. It is zero for a correctly scaled posterior mean, and positive when the ordering is right but the spread is wrong (usually under-shrinkage, i.e. an over-confident posterior) — a failure neither correlation nor NRMSE reveals alone.

---

### Package K: Training Search Space — Batch, Learning Rate, Budget

Motivated by a 3x3 batch x learning-rate grid run in `bayesflow-irt`
(9 arms x 3 seeds, simulation-matched at 102,400 simulations). Full results
and discussion: https://github.com/matthiaskloft/bayesflow-hpo/issues/67

1. **`batch_size` is currently `IntDimension("batch_size", constant=256)`** in
   `search_spaces/training.py`. At a constant batch the batch x learning-rate
   interaction is invisible by construction, and that interaction was the
   largest effect measured: the penalty for reusing a learning rate tuned at a
   smaller batch grows with batch size (5.4x / 4.1x / 3.6x on `b` calibration
   at batch 32 / 64 / 128).

2. **`initial_lr` is capped at `5e-3`.** Three of the four Pareto-optimal
   configurations in the grid used `8e-3`, so the cheap end of the front is
   outside the search space entirely. Widen the range — but note `8e-3` is not
   a free win: it improves `b` calibration at every batch size while degrading
   `log(a)` calibration at the five-item corner.

3. **Couple lr to batch by reparametrisation, not a conditional space**:
   `lr = lr_ref * (batch_size / reference_batch)`. The coupling has a known
   functional form (Smith et al. 2018, `B` proportional to `epsilon`), so a
   rectangle in decoupled coordinates is easier for TPE than a conditional
   space — TPE models parameters marginally, which is the worst case for a
   ridge.

4. **Derive `num_batches` from a simulation budget.** In online SBI
   `batch_size` is simultaneously the data-volume knob
   (`simulations = batch_size * steps`), so at most two of
   {batch, steps, simulations} can be fixed. Fixing the budget is the
   meaningful choice since it is proportional to GPU-seconds.

5. **Do not add warmup length as a tunable dimension.** It is currently
   hardcoded at 5% of total steps and untested, and the temptation to tune it
   grows once the lr range widens. Shallue et al. (2019, Sec. 5.1) report the
   opposite lesson from the study that motivates the rest of this package:
   "We are most confident in our search spaces that tuned the fewest
   metaparameters... We found it quite difficult to be confident that our
   tuning was sufficient when we searched over learning rate decay schedules."
   Sweep warmup factorially outside the HPO and fix it at a known-good value,
   rather than adding a third correlated axis to `{batch, lr}`.

Depends on: `fix/hpo-review-findings` merging first — it already rewrites
dimension types (`BoolDimension`) and removes dead dimensions in three search
spaces, so stacking this on top before that lands would compound an unreviewed
branch.

---

### Package A2: Research — Detailed Sampler Preset Defaults

The sampler presets are implemented (PR #56). This research task remains
open to optimize the defaults.

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

---

### Package A3: QMC Warm-up (remaining research)

#### Research: QMC warm-up effectiveness

Status:
- Feature implementation is complete (see Done entry: Package A3, PR #57).
- Research notebook configured for final run:
  `examples/qmc_warmup_benchmark.ipynb`
  - 10 replications per condition (N_REPLICATIONS=10)
  - 40 trials per run (N_TRIALS=40)
  - 30 epochs/batches (EPOCHS=30, NUM_BATCHES=30)
  - Checkpoints at 8, 16, 24 trials

Remaining work:
- Execute the notebook end-to-end (expected runtime: several hours)
- Record empirical results (convergence, final metrics, statistical tests) in docs
- Decide whether findings justify default changes for QMC startup settings
  and/or inclusion in the HPO benchmark paper

---

---

## Done

### Package J: FlowMatching Search-Space Expansion (2026-04-25)
Expanded `FlowMatchingSpace` to expose BayesFlow-default solver and TimeMLP
kwargs as dimensions:
- Added constants matching BayesFlow defaults for
  `fm_integrate_method`, `fm_integrate_steps`, `fm_merge`, `fm_norm`,
  `fm_residual`, `fm_spectral_normalization`, and `fm_kernel_initializer`.
- Updated `fm_time_embedding_dim` default constant from `8` to `32`
  to match local BayesFlow defaults when the dimension is not tuned.
- Wired new dimensions through `build()` into
  `integrate_kwargs` and `subnet_kwargs`.
- Added profile helpers:
  `FlowMatchingSpace.fast()`, `.balanced()`, `.quality()`, and
  `.preset(...)`.
- Updated docs (`search_spaces.md`, `defaults.md`, `api_reference.md`,
  `optimization.md`, `README.md`) and tests.

### Package I: Source-Backed Reference Details (2026-04-10)

Completed all three sub-tasks for documentation-backed implementation.

#### Sub-task 1: Tracked article summaries in `docs/references/`

All 16 reference summaries exist with extensive method details, algorithm
citations, and implementation notes.

#### Sub-task 2: Docstring citations

Added References sections to module docstrings:
- `validation/sbc_tests.py` — Talts et al. (2018), Theorem 2
- `validation/registry.py` — Talts et al. (2018), Section 4
- `optimization/pruning_strategies.py` — Schmucker et al. (2021), Emmerich & Deutz (2018), Deb et al. (2002)
- `optimization/study.py` — Sobol' (1967), Joe & Kuo (2008)
- `results/extraction.py` — Deb et al. (2002)
- `validation/c2st.py` — Linhart et al. (2023), López-Paz & Oquab (2017)

#### Sub-task 3: Inline literature comments

Added source-backed comments at key implementation points (SBC ranks,
coverage intervals, non-dominated sort, MO-SHA pruning, QMC power-of-2,
L-C2ST, global C2ST).

---

### Package I: Literature Audit (2026-04-09)

Audited all HPO-specific features against literature references.
Reorganized `docs/references.md` with topic-grouped coverage matrix
mapping each feature to its backing reference. Scope limited to features
directly implemented in bayesflow-hpo (HPO, pruning, trial selection,
QMC warm-up, SBC tests, C2ST metrics); BayesFlow core diagnostics
(ECE, contraction, z-score, log_gamma) are excluded as they wrap
`bf.diagnostics.*`.

Added 3 missing references:
- Talts et al. (2018) — SBC methodology (backs `sbc_tests.py`, coverage)
- Sobol' (1967) — Sobol sequences (backs QMC warm-up)
- Joe & Kuo (2008) — Improved Sobol direction numbers (used by
  SciPy/Optuna QMCSampler)
- Bischl et al. (2023) — HPO foundations survey

Verified implementations match reference methods: SBC rank computation,
normalization, KS/chi-squared tests, coverage intervals, QMC power-of-2
properties, pruning strategies.

### Package H: Metric Constraints & Memory Auto-Detection (2026-04-05)
Implemented layered metric constraints and memory auto-detection:
- Added hard metric constraints via `metric_constraints_hard` in
  `ObjectiveConfig` / `optimize()`, with post-validation rejection
  (`rejected_reason="metric_constraint"`).
- Added soft metric constraints via `metric_constraints_soft`, wired
  through composed sampler `constraints_func` (`_make_constraints_func`).
- Updated trial counting semantics so metric-rejected trials count
  toward non-rejected trial caps while pre-training budget rejections
  (`memory_budget`, `param_budget`, etc.) remain excluded.
- Added `max_memory_mb="auto"` plus `memory_safety_margin`, resolved via
  CUDA free-memory detection (`torch.cuda.mem_get_info()`), with
  graceful fallback to disabled memory budget when unavailable.

### Package E: C2ST Metrics (2026-04-02)
Added classifier two-sample test metrics for multivariate posterior
validation. New module `validation/c2st.py` with three components:
`lc2st()` (Linhart et al., 2023) for reference-free local posterior
diagnostics using joint samples, `global_c2st()` (López-Paz & Oquab,
2017) for standard C2ST when reference posterior samples are available,
and `make_lc2st_validate_fn()` factory returning a `ValidateFn`
compatible with `optimize(validate_fn=...)` that computes standard
per-parameter metrics and L-C2ST from a single inference pass. Added
`scikit-learn>=1.3` as optional dependency (`pip install
bayesflow-hpo[sklearn]`). 14 new tests.

### Deferred Code Quality Fixes (2026-04-01)
Three small issues found during PR #50 review, now resolved:
- **Narrowed `except TypeError`** in objective compile step: split the
  try block so `_make_cosine_decay_optimizer` errors propagate as
  compile failures instead of being silently swallowed. Only
  `_compile_for_compat` TypeError (signature mismatch) is caught.
- **Per-metric penalty values** in `_validate_metric_keys`: added
  `penalty_values` dict parameter so cost metrics use `FAILED_TRIAL_COST`
  (1e6) instead of `FAILED_TRIAL_CAL_ERROR` (1.0). Wired via new
  `_metric_penalty_map()` helper on `GenericObjective`.
- **Tightened `validation_data` type** on `PeriodicValidationCallback`
  from `Any` to `ValidationDataset`. 4 new tests.

### Package B: Trial Selection & Results (2026-03-27)
Added lexicographic-Pareto trial selection (`select_best_trial()`) to
`results/extraction.py`. Two-phase algorithm: (1) satisficing — filter
candidates by priority thresholds in order, promoting unmet priorities to
Phase 2; (2) Pareto selection over remaining study objectives with mean-rank
tiebreak (Deb et al., 2002). Priorities are tuple-based: 2-tuple
`(metric, threshold)` infers direction from `study.directions`, 3-tuple
`(metric, threshold, "below"/"above")` for user attributes. Returns
`(trial, SelectionResult)` with diagnostic metadata. Integrated into
`best_config()` via optional `priorities` parameter. Also added
`_validate_select_by()` bounds check to `trial_table()`, `best_config()`,
and `summarize_study()`. 25 new tests.

### Package G: Search Space Gaps (2026-03-26)
Added `ft_mlp_width` (constant 128), `ft_mlp_depth` (constant 2), and
`ft_bidirectional` (constant True) to `FusionTransformerSpace`, matching
peer summary spaces (SetTransformer, TimeSeriesTransformer, TimeSeriesNetwork).
Updated `build()` to pass `mlp_widths`, `mlp_depths`, and `bidirectional`
through to `bf.networks.FusionTransformer`. Added early validation in
`IntDimension.__post_init__` rejecting `log=True` + `step` (other than 1)
— Optuna's `suggest_int()` rejects this combination at runtime. 9 new tests.

### Package A3: QMC Warm-up (2026-03-26, PR #57)
Added `qmc_startup_trials` parameter to `create_study()` and `optimize()`.
`QMCWarmupSampler` composite wrapper delegates to `QMCSampler` (Sobol) for
the first N non-rejected trials, then transparently switches to the main
sampler. Composes with all 7 sampler presets, warm-start, and budget-aware
sampling. Power-of-2 warning for non-optimal Sobol counts. 34 new tests.

### Package A2: Sampler Presets (2026-03-22, PR #56)
Added 7 named sampler presets (`"tpe"`, `"gp"`, `"botorch"`, `"nsga2"`,
`"nsga3"`, `"auto"`, `"random"`) to `create_study()` and `optimize()`.
All presets auto-wire `constraints_func` for budget-aware sampling.
BoTorch and Auto use lazy imports with clear `ImportError` messages.
Added `_resolve_n_startup_trials()` for smarter pruning warmup alignment
(checks `_n_startup_trials`, `population_size`, fallback 10).
Bumped `optuna>=3.0.0` to `>=4.0.0`. 32 new tests.

### Package A1: Pruning Review & Refactor (2026-03-22, PRs #51–#54)
Four-phase rework of multi-objective pruning:
- **Phase 1** (PR #51): New `optimization/pruning_strategies.py` with three
  literature-backed strategy functions: `should_prune_dominance()` (normalized
  median AND rule, adapted from MO-ASHA; Schmucker et al., 2021),
  `should_prune_mo_sha()` (non-dominated sorting + bottom-fraction pruning;
  Schmucker et al., 2021, Algorithm 2), `should_prune_primary()` (single-metric
  median; Akiba et al., 2019). Pure-numpy `_non_dominated_sort()` (Deb et al.,
  2002). 41 tests.
- **Phase 2** (PR #52): Refactored `PeriodicValidationCallback` for pluggable
  strategies. Replaced hard-coded `["calibration_error", "nrmse"]` with
  `objective_metrics` parameter. Per-metric user attrs (`val_{metric}_step_{N}`)
  replace single composite `val_score_step_*`. Strategy dispatch via
  `_evaluate_pruning()`. 11 tests.
- **Phase 3** (PR #53): Wired `pruning_strategy` through `optimize()` →
  `ObjectiveConfig` → callback. Auto-detect `n_startup_trials` from
  `sampler.n_startup_trials` (fallback 10). `pruning_strategy="none"` skips
  callback entirely. 12 tests.
- **Phase 4** (PR #54): Added pruner string presets (`"median"`, `"hyperband"`,
  `"none"`) to `create_study()`. 10 tests. 3 new references (Schmucker et al.,
  Li et al., Emmerich & Deutz).

### Package I: Small API Fixes (2026-03-22, PR #55)
Three standalone fixes: `normalize_param_count` / `denormalize_param_count`
raise `ValueError` on degenerate bounds instead of silent `return 0.0`;
`infer_keys_from_adapter` logs at DEBUG when adapter has no `transforms`;
`make_bayesflow_infer_fn` validates `data_keys` via `available_keys` param
and removes silent skip. #5 (explicit keys in `optimize()`) dropped as YAGNI.
10 new tests.

### Package D (remaining): Validation Contract Cleanup (2026-03-22, PR #50)
Made `ObjectiveConfig.validation_data` required (non-Optional) with
`isinstance` guard. Removed dead `None`-fallback branch and `is not None`
guards. Documented `validate_fn` contract (required keys, penalty fallback,
timing caveat, intermediate pruning) in `types.py`, `api.py`, and
`objective.py`. Timing semantics difference between default and custom
`validate_fn` paths documented as a known limitation.

### Package D: `optimize()` Refactor (2026-03-21, PR #49)
Decomposed `optimize()` into 5 private helpers for readability and
testability. Fixed `_TrackingDict` false positives by overriding
`items()`/`values()`. Extracted `_register_with_aliases()` to eliminate
duplicated alias loops. No public API changes. 23 new tests.

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
