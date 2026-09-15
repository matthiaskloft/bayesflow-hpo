# Changelog

## Unreleased

### Added

- `TrainingSpace(lr_reference_batch_size=...)` reparametrizes the peak
  learning rate as `initial_lr = lr_ref * batch_size / reference_batch`, the
  linear scaling relationship of Smith et al. (2018). The sampled Optuna
  parameter is then `lr_ref`; `initial_lr` is derived. Off by default.
- `TrainingSpace(simulation_budget=...)` derives
  `num_batches = budget // (batch_size * epochs)`, keeping trials
  simulation-matched while batch size and epochs vary. Requires an `epochs`
  dimension on the space; an infeasible budget is rejected at construction.
  Off by default.
- Trials record their realized `simulations` (`batch_size * epochs *
  num_batches`) as a user attribute, and it is now a default column of
  `trials_to_dataframe()`.
- Values a search space derives rather than samples are recorded under the
  `derived_params` trial user attribute, and `best_config()`, `trial_table()`,
  `trials_to_dataframe()` and `compare_trials()` report them alongside
  `trial.params`. Without this, retraining from `best_config()` on a study
  with a reparametrized learning rate would silently use a different rate than
  the trial that was selected.
- `CheckpointPool(pruned_pool_size=...)` opts into retaining pruned trials'
  weights, which no `pool_size` could keep before: `maybe_save()` runs after
  scoring, and a pruned trial raises before reaching it. Off by default (0).
  Pruned checkpoints go to a separate pool under `pool_dir / "pruned"`, so
  they can never evict a scored trial, and past the cap retention is a
  uniform random sample of the pruned population rather than top-k --
  pruned trials stop at different rungs, so their scores are not comparable
  across rungs and top-k cannot cover the low end. Pass `seed=` for a
  reproducible sample.
- Every checkpoint now carries a `checkpoint.json` sidecar recording the
  trial number, state, objective value and -- for pruned trials -- the rung
  they stopped at, which cannot be recovered from the weights.
- `PeriodicValidationCallback.validation_step` and `.last_scores` report the
  rung the approximator's current weights were measured at, accounting for
  early stopping having restored weights from an earlier rung.
- **Joint, data-dependent metrics.** `register_joint_metric` and
  `JointMetricInputs` expose the per-condition context that
  `run_validation_pipeline` already held and then discarded, so a metric
  needing the posterior draws *and* their simulations no longer has to
  rebuild the condition loop. `make_lc2st_validate_fn` is refactored onto it
  and its duplicate loop deleted; the refactor was checked against the
  pre-refactor implementation and is bit-identical for one parameter
  (6.9e-18 for three, from float association order).
- **TARP** (Lemos et al., 2023), with the reference contract and the
  `bayesflow_hpo_joint_metric_settings` study record that keeps trials scored
  under different settings from being compared. `tarp_error` is an
  **objective** and requires supplied reference points. `tarp_error_random`
  is registered as a **diagnostic** and is rejected by
  `validate_objective_metric_kinds()` if passed in `objective_metrics`: with
  random references the statistic is blind to a posterior that ignores its
  data, so it cannot be optimized against.
- `lc2st` is a registered objective metric with a `METRIC_DIRECTIONS` entry.
  It needs the `sklearn` extra.
- Joint metrics are **off** `PeriodicValidationCallback` by default
  (`include_joint_metrics=False`) and support `max_conditions` sub-sampling.
  L-C2ST measured ~54 s per condition, so a 20-condition grid would spend
  ~18 minutes per interval deciding whether to prune; TARP is ~79 ms, so the
  choice is per-metric rather than global.
- `max_samples_per_call` (default 20,000 draws) chunks validation inference,
  threaded through `run_validation_pipeline`, `ObjectiveConfig`,
  `PeriodicValidationCallback`, `validate_once`, `default_validate_fn`,
  `make_lc2st_validate_fn` and `optimize()`, which rejects an invalid value
  up front. A batch that already fits takes the original single-call path.
  Chunking preserves the assembled array's shape, row order and draws per
  simulation, including the single-parameter trailing-axis squeeze; it does
  **not** reproduce the same draws, because each chunk is its own
  `approximator.sample()` call and BayesFlow sampling is stochastic. A
  chunked run and an unchunked one at the same seed can therefore give
  different metric values.
- `estimate_validation_memory_mb` budgets one `sample()` call, and the
  objective now checks it alongside the training estimate. Over-budget trials
  are rejected pre-training with `rejected_reason="validation_memory_budget"`,
  so they cost no trained model.

### Fixed

- **A TARP seed collision that scored a perfect posterior as maximally
  broken.** Drawing reference points from `default_rng(seed)` consumes the
  same uniform stream a caller's simulator does. Seeding a simulator and TARP
  alike -- the obvious thing for a reproducible study -- made every reference
  point an affine image of its own simulation's truth at matching shapes
  (measured correlation exactly 1.0), collapsing every `f_i` to ~0 so that a
  perfectly calibrated posterior scored `tarp_error = 0.5`, the worst value
  the statistic can take. Nothing raised. Fixed with a spawn key.
- Validation inference sampled an entire condition batch in one
  `approximator.sample()` call -- 100,000 posterior draws at once at the
  `optimize()` defaults. Neither factor is a search-space hyperparameter, so
  the pre-training budget check could not reject such a trial: it died in
  validation *after* the training run had been paid for, and recorded a
  model-quality penalty for what is a resource problem.
- Validation inference silently truncated condition values that disagreed on
  their leading dimension; that now raises. A 0-d value, or one with a
  leading dimension of 1, is correctly treated as a broadcast rather than a
  one-row batch, which would otherwise have switched chunking off for any
  condition carrying a covariate.
- CI never ran a single L-C2ST test: the C2ST module is behind
  `importorskip("sklearn")` and CI installed only `[dev]`. It now installs
  `[dev,sklearn]`.
- Documentation listed a `decay_rate` dimension on `TrainingSpace` that has
  not existed since 0.2.0.

### Documentation

- `docs/references/` is deleted. Seventeen per-paper summaries (~3,700 lines)
  backed no code path, and every one of the three ever spot-checked was
  defective. `docs/references.md` is now the single record.
- The fourteen entries that had metadata but no substance check were read
  back against full texts. Two were wrong in the same way -- a result the
  paper *uses* described as one it *introduces*: Bergstra et al. (2011) does
  not propose SMBO (§2 reviews it as prior art; TPE is its own), and Smith et
  al. (2018) does not show the gradient-noise coupling (§1 attributes it to
  Smith & Le, 2017), and carries a `B << N` condition the entry had dropped.
- Bergstra et al. (2011) has no DOI at any version, now recorded explicitly
  as a finding rather than left looking like an omission.
- `scripts/check_citations.py` (and a CI job) assert that every citation and
  locator in `src/` is stated in `docs/references.md`. It checks consistency,
  not truth.

## 0.3.0

A feature release, and unlike 0.2.0 a safe upgrade for existing studies: no
change here alters what a stored objective value *means*, so 0.2.0 studies
resume and stay comparable. The `calibration_error` computation is frozen on
purpose even though its documentation was wrong (see **Fixed**).

Four things to check before upgrading. The first two change behaviour you may
be relying on; the last two are the corrections most likely to matter.

- **`optuna` now requires `>=5.0.0,<6.0.0`**, up from `>=4.0.0`. A
  support-policy decision rather than a bug fix -- testing one optuna major
  keeps local measurements and CI results comparable -- and it means
  `bayesflow-hpo` can no longer be installed alongside an application pinned
  to optuna 4. **Changed** records the behaviour difference measured directly
  on both majors.
- **`mean_calibration_error` is now a built-in metric name.** Registering a
  custom metric under that name raises `ValueError` unless you pass
  `overwrite=True`. That is the release's only user-visible change to an
  existing API.
- **`pruning_strategy="none"` did not actually disable pruning** on a
  single-objective study: the callback consulted Optuna's default
  `MedianPruner` regardless. If you run a single-objective study with early
  stopping on, trials you expected to run to their horizon may have been
  terminated. Fixed.
- **`plot_parallel_coordinates` inverted its last axis unconditionally.**
  Correct for a cost column, wrong for anything else. Only reachable with the
  new `cost_metric=None`, so no 0.2.0 plot was affected -- but the fix adds a
  study user attribute (`bayesflow_hpo_has_cost_objective`) that older studies
  do not carry, and whose absence is read as "has a cost column".

The headline feature is `cost_metric=None`, which lets a study search over the
quality metrics alone. It changes the arity of the stored objective tuple, so
a study cannot be resumed or warm-started across a change of the setting --
the schema guard refuses the mismatch rather than mis-indexing it.

### Added

- **`cost_metric=None`.** `optimize()` and `ObjectiveConfig` now accept
  `cost_metric=None`, producing a study whose Optuna directions are the
  quality metrics alone: `len(objective_metrics)` in `"pareto"` mode, one in
  `"mean"` mode.

  This is not the same as ignoring the cost column at selection time. As an
  Optuna direction, cost shapes the search itself: the sampler models it and
  spends budget exploring the cheap-model frontier, and every cheap trial is
  non-dominated on the cost axis however mediocre its quality, so it enters
  the Pareto front that selection and warm-start read. That budget is spent
  before selection ever runs and cannot be recovered there.

  Intermediate pruning is unaffected either way. The strategies in
  `optimization/pruning_strategies.py` read `val_{metric}_step_{N}` user
  attrs written from `objective_metrics` alone, so cost has never entered a
  pruning comparison.

  The non-dominance rule behind the Pareto-front claim is Deb et al. (2002);
  the Optuna contract relied on -- objectives addressed by position, and at
  least one direction required -- is recorded against the installed version.
  Both are in [`docs/references.md`](docs/references.md).

  `objective_metrics` may not be empty when `cost_metric=None`: that would
  leave no objectives at all, which Optuna rejects. `ObjectiveConfig` now
  refuses the pair up front rather than failing later inside `create_study`.

  Cost is still *measured*: `param_count` and `inference_time_s` remain trial
  user attributes on every completed trial, which is what makes post-hoc cost
  ranking possible. `max_param_count` also still applies -- it constrains what
  gets built, independently of what gets optimized.

  **A study's stored objective tuple has a different arity depending on this
  setting**, so raw objective values are comparable only across studies run
  with the same one. Resuming or warm-starting across a change is refused by
  the existing schema guard rather than silently mis-indexed.

  `mean_objective_score()` and `warm_start_study()` take a new `has_cost`
  keyword (default `True`, the previous behavior). Passing `has_cost=False`
  is required for a `cost_metric=None` study with more than one quality
  metric: the default drops the last element as a cost score, which would
  silently omit a real metric from the checkpoint-pool and warm-start
  rankings.

- **`mean_calibration_error` metric.** `bf.diagnostics.calibration_error`
  with `aggregation=np.mean` instead of its default `np.median`, registered
  with its own `METRIC_DIRECTIONS` entry (`higher_is_better=False`,
  `worst_raw=1.0`) so it can be passed to `objective_metrics` directly. It is
  deliberately *not* in `DEFAULT_METRICS`: adding it there would change the
  columns of every stored summary.

  It is **not** called `ece`, and has no `mean_cal_error` alias. Both names
  were considered and rejected: `bf.diagnostics.expected_calibration_error`
  already exists and is a different statistic (bin-weighted, over one-hot
  model indices, for model comparison, after Naeini et al. 2015), and
  `mean_cal_error` is already an output key of the `coverage` family and a
  documented `objective_scalar` fallback. The ECE of the literature is a
  weighted mean over bins of predicted probability, not an unweighted mean
  over equally spaced nominal coverage levels, so the term is not claimed.
  `docs/references.md` records that this metric has no upstream reference:
  the aggregation and the name are this package's own.

  Prefer it over `calibration_error` for new work. A median over the 20
  nominal levels discards half the calibration curve, so a posterior whose
  central intervals behave can hide badly miscalibrated tails. On a posterior
  with its tails truncated at 1.2 SD, swept over 20 seeds, the mean-aggregated
  value lands in 0.031-0.041 while the median-aggregated one lands in
  0.001-0.017 — the median often reads as barely distinguishable from a
  perfectly calibrated posterior
  (`tests/test_validation/test_mean_vs_median_calibration_error.py`). Ratios
  against
  each metric's own calibrated baseline are not quoted here: those baselines
  are Monte Carlo noise, so the ratio swings between 3.9x and 20.4x across
  seeds while the absolute values stay put.

  It is listed in `ENCODING_UNCHANGED_AT_V2`. The metric postdates
  encoding 2, so no pre-v2 study can hold a column for it; leaving it
  out would mark it encoding-sensitive and block resuming studies over
  an encoding with no history to differ from.

- **`sampler_n_startup_trials` on `optimize()` and `create_study()`.** The
  `"tpe"` preset hardcoded `n_startup_trials=25` — 2.5x Optuna's own default of
  10 — with no way to change it, so a study smaller than ~40 trials spent most
  of its budget on uniform-random draws and reached the TPE model only at the
  very end. The new parameter overrides the count for the presets that take one
  (`"tpe"`, `"gp"`, `"botorch"`); `None` keeps the preset value, so existing
  behaviour is unchanged.

  This matters most alongside `qmc_startup_trials`. Optuna counts the study's
  `COMPLETE` and `PRUNED` trials toward the startup quota, not the sampler's own
  draws, so Sobol warm-up trials already count — but with the preset's 25 the
  model still waited for 25 trials regardless. Setting
  `sampler_n_startup_trials` at or below `qmc_startup_trials` hands over to the
  model exactly when the Sobol phase ends, which is what the warm-up was for.
  `QMCWarmupSampler.n_startup_trials` still reports
  `max(qmc_quota, main_startup)`, so pruning alignment does not inherit the
  lowered number.

  The parameter is keyword-only and appended after `show_progress_bar`:
  `optimize()` has no keyword-only separator, so inserting it among the
  existing parameters would have silently rebound the trailing positional
  arguments of any caller passing `checkpoint_pool` or `show_progress_bar`
  positionally.

  Both startup counts are now validated at the top of `optimize()` rather than
  only where they are consumed. `create_study()` checked them, but by then a
  non-resumed run had already called `optuna.delete_study()` — so a negative
  value destroyed the previous study and its trials before raising, leaving no
  replacement. This also fixes that pre-existing hole for `qmc_startup_trials`.

  The override is ignored, with a warning, when `sampler` is a sampler instance
  rather than a preset name — the same restriction `metric_constraints_soft`
  already documents, for the same reason.

  Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for
  hyper-parameter optimization. *Advances in Neural Information Processing
  Systems, 24*, 2546–2554.

### Fixed

- **`plot_parallel_coordinates` inverted the last axis unconditionally.** It
  treated the final objective column as a cost score -- log-transforming,
  negating and relabelling it `-log(...)` -- which is right only when there
  *is* a cost column. A `cost_metric=None` study ends in an ordinary quality
  metric, so a worse `nrmse` plotted higher on its axis under a label
  claiming the inversion was deliberate. No exception, no warning.

  Column names cannot settle it: `cost_metric=None` over two quality metrics
  and `cost_metric="param_count"` over one both produce two columns, and
  sniffing for the name `param_count` would misread a user metric that
  happens to share it. `create_study()` now stamps
  `bayesflow_hpo_has_cost_objective` on the study and the plot reads it.
  Studies written before the stamp existed all carried a cost column, so its
  absence defaults to `True` and their plots are unchanged.

- **`pruning_strategy="none"` did not disable pruning on a single-objective
  study.** `_evaluate_pruning` returns `False` for `"none"`, so the
  multi-objective path always honoured it, but the single-objective path
  called `trial.should_prune()` unconditionally -- handing the decision to
  whichever pruner `create_study()` installed, by default `MedianPruner`.
  `optimize()` attaches this callback whenever early stopping is on, so a run
  that never asked for pruning could still have trials terminated. The
  strategy is now checked before Optuna's pruner is consulted; intermediate
  values are still reported, since reporting alone prunes nothing.

- **Single-objective intermediate pruning reported the wrong quantity.**
  `PeriodicValidationCallback` reported `objective_metrics[0]` to Optuna's
  pruner for a one-direction study, while mean mode's actual objective is the
  *mean* of the metrics -- so pruning decisions were made on a quantity the
  study does not optimize. It now reports that mean (identical to the old
  behaviour when there is only one metric).

  Previously unreachable through `optimize()`, which always produced at least
  two directions; `cost_metric=None` reaches it in mean mode and in pareto
  mode over a single metric. In that state the multi-objective
  `pruning_strategy` cannot run either -- every strategy compares several
  objectives -- and Optuna's own pruner takes over. That was silent; a
  non-default strategy now logs a warning saying so.

- **`calibration_error` was documented as an Expected Calibration Error in six
  places; it is a median, not a mean.** The metric wraps
  `bf.diagnostics.calibration_error` with all defaults, which aggregates the
  per-level absolute coverage deviations with `aggregation=np.median`
  (`bayesflow/diagnostics/metrics/calibration_error.py:15`). An ECE is a mean.
  Corrected in `validation/registry.py` (function docstring, module docstring
  and registered description), `api.py`'s built-in metric table,
  `docs/references.md`, `docs/validation.md` and `docs/quality_report.md`.

  The conflation was inherited, not invented here: BayesFlow's own
  `basic_workflow.py` still documents this quantity as "Expected Calibration
  Error (ECE)" in `compute_default_diagnostics`. A reader cross-checking
  upstream docs will find the two descriptions disagree; this package's is the
  one that matches the code.

  The `worst_raw=1.0` comment in `objectives.py` was the load-bearing one: it
  justified the bound by calling the metric a mean of absolute deviations
  between two probabilities. The bound itself survives -- a median of values
  in [0, 1] is in [0, 1] -- but the recorded reason was wrong.

  Registering a custom metric named `mean_calibration_error` through
  `register_metric()` now
  raises `ValueError` unless `overwrite=True`, and `list_metrics()` /
  `describe_metrics()` gain a row. That is the only user-visible behaviour
  change.

  The "13 built-in validation metrics" counts in `README.md` and `CLAUDE.md`
  were already stale before this change -- the registry held 15 -- and now
  read 16, matching `len(list_metrics())`.

  **No stored value changes.** The computation of `calibration_error` is
  frozen on purpose so that records from earlier studies stay comparable; only
  the documentation is corrected, and `mean_calibration_error` is added
  alongside as the mean-aggregated variant. (issue #83)

- **Unused-hparam warning no longer advises deleting live search dimensions.**
  `check_pipeline` handed `train_fn` a plain `dict(hparams)` copy, so every read
  inside a custom train hook was invisible to tracking by construction and the
  unused set could only ever reflect `build_approximator_fn`. A run whose
  `train_fn` consumed `batch_size` — the common case, as the starting point of
  an OOM-retry loop — was told to remove it "to avoid wasting Optuna budget";
  acting on that would have pinned every trial to the default. The hook now
  receives a tracking copy and its reads count toward the report.

  The message is advisory rather than prescriptive, because the fix cannot be
  complete: `_TrackingDict` deliberately does not override `__iter__` (so that
  `dict(td)` does not falsely mark every key), which means a hook that copies
  the dict reads the copy and stays untrackable either way. The warning now
  names both hooks and says so outright instead of recommending removal.
  ([#88](https://github.com/matthiaskloft/bayesflow-hpo/issues/88))

- **`check_pipeline` no longer rejects `log_gamma`.** The pre-flight refused
  any non-finite objective value, while `METRIC_DIRECTIONS` gives `log_gamma`
  `worst_raw=-math.inf` deliberately — so the one metric the table declares
  unbounded below could not pass the pre-flight. Since the pre-flight validates
  at `n_posterior_samples=2` on a barely-trained model, where the gamma
  discrepancy underflows to `0.0` and `log_gamma` is legitimately `-inf`,
  `run_pipeline_check=True` and `log_gamma` could not be combined at all.

  An infinity is now accepted only where it **matches** the metric's registered
  `worst_raw` exactly — so `-inf` passes for `log_gamma` while `+inf` does not,
  since `log(gamma / null_quantile)` with `gamma` a probability is unbounded
  below and not above. An unregistered metric still refuses any infinity, and
  `NaN` is refused everywhere. The failure scaled *with* validation size — small
  sets lack the ranks to drive gamma to zero — so it passed on toy
  configurations and failed on real ones.
  ([#84](https://github.com/matthiaskloft/bayesflow-hpo/issues/84))

  The gamma discrepancy — the probability, under uniform ranks, of the most
  extreme point of the observed rank ECDF — is Säilynoja et al. (2022).
  Modrák et al. (2025) adopt it in Section 4.1 and define the quantity
  BayesFlow's `calibration_log_gamma` reports, `log(gamma / gamma_bar)` with
  `gamma_bar` the 5th percentile of the null distribution, so ranks extreme
  enough to drive `gamma` to `0.0` give `-inf`. Both are recorded in
  `docs/references.md`.

### Changed

- **`optuna` now requires `>=5.0.0,<6.0.0`.** It was `>=4.0.0`, so a local
  environment on 4.9.0 and a fresh CI install on 5.0.0 ran different code —
  and optuna 5.0 already changed behaviour this package depends on.
  Measurements taken locally were therefore not measurements of what CI runs.

  The behaviour change, observed directly on both released versions rather
  than taken from release notes — for a study with no completed trials, and
  for one with a single trial and no varying parameters:

  | | `optuna.importance.get_param_importances(study)` |
  |---|---|
  | 4.9.0 | raises `ValueError`: "Cannot evaluate parameter importances without completed trials." / "…with only a single trial." |
  | 5.0.0 | returns `{}` |

  `plot_param_importance` treated "did not raise" as success, so on 5.0 it
  drew an empty chart and returned a figure where its contract says `None`
  (fixed in #85).

  The #85 fix is **not** what forces the floor, and an earlier draft of this
  entry wrongly said it was. `plot_param_importance()` handles both
  signals -- the 4.x raise, via `except Exception`, and the 5.0 empty
  mapping, via the branch #85 added -- so it works unchanged on either
  major version. Both branches now have regression tests that stub
  `get_param_importances`, so neither depends on which optuna is
  installed.

  The floor is a deliberate **support-policy decision**: testing one
  optuna major rather than two keeps local measurements and CI results
  comparable, which the version split had already broken. It does mean
  `bayesflow-hpo` can no longer be installed alongside an application
  pinned to optuna 4. The range stays a range so that installing
  `bayesflow-hpo` alongside other optuna-dependent packages does not force a
  resolver conflict; the single version CI actually tests is pinned exactly in
  `.github/ci-constraints.txt`. To reproduce a CI environment locally:

  ```bash
  pip install -e ".[dev]" -c .github/ci-constraints.txt
  ```

### Documentation

- **Development setup now states the environment requirement.** `CLAUDE.md`,
  `AGENTS.md`, `README.md` and the `test-hpo`/`lint` skills previously said
  `pip install -e ".[dev]"` and `pytest tests/ -v`, which silently assumes an
  activated virtualenv that already has a Keras backend. Two failure modes went
  unrecorded: `[dev]` pulls no backend, so a venv built from it alone imports
  `pytest` and then fails at `import bayesflow`; and an editable install
  resolves to the source tree it was installed from, so a venv borrowed from
  another `git worktree` tests that worktree's source rather than the one under
  review. All five documents now require a per-checkout `.venv`, the separate
  torch install, and invocation through the venv interpreter.

- **Citation audit: six more claims corrected against full texts.** A
  systematic sweep of every implementation-backing citation in `src/`,
  `docs/references.md` and `docs/references/`, following the four errors
  found earlier in this release. Nothing here changes behaviour; all of it
  changes what the code claims its behaviour is grounded in.

  - The SBC rank-uniformity result is Talts et al. (2018) **Theorem 1**
    (Sec. 4.1, p. 6), not Theorem 2, and it states that exact posterior
    samples *imply* uniform ranks — not the equivalence that three code
    comments asserted with "iff". SBC is a necessary, not sufficient, check,
    which the paper says explicitly.
  - Median pruning is no longer attributed to Akiba et al. (2019). That
    paper's Algorithm 1 is the Successive Halving pruner and specifies no
    median rule; `MedianPruner` is documented only in the Optuna API
    reference, which is now what `"primary"` cites.
  - Hyperband's `eta = 3` default is in **Algorithm 1**'s input line.
    Section 3.6, cited previously, recommends "3 or 4" and gives the
    theoretical optimum as `e ≈ 2.718` — a different claim.
  - Emmerich & Deutz (2018) was cited for "non-dominated sorting
    (Eqs. 3--4)" and "complexity bounds (Props. 7, 9)". Neither exists as
    described; those propositions develop cone orders. Definition 5, Pareto
    dominance, was the one correct locator and is what we keep.
  - The power-of-two warning in `optimize()`'s QMC warm-up now cites the
    SciPy `qmc.Sobol` documentation, which states the property and which
    Optuna's `QMCSampler` actually wraps, rather than Sobol' (1967) — whose
    indexed copy is the Russian original and could not support the locator.
  - `validation_callback.py` still described `"dominance"` as MO-ASHA's
    promotion rule, without the correction already applied to
    `pruning_strategies.py`.

  Verified and left alone: Deb et al.'s O(MN^2) sorting, Talts's Algorithm 1,
  Linhart's Algorithms 1--2, Li et al.'s Section 6 Sobol suggestion, Joe &
  Kuo as SciPy's direction-number source, and — re-executed on 5.0.0 rather
  than assumed — that `Trial.report()` still raises `NotImplementedError` for
  multi-objective studies, which is the premise the whole pruning module
  rests on.

  `docs/references/*.md` is now marked unreliable: spot checks found
  misidentified definitions, Hyperband's Algorithm 1 labelled "Successive
  Halving" with pseudocode that is not the paper's, and a cited "ASHA (Li et
  al., 2016), JMLR 17(142)" that does not appear to exist. No code path
  depends on those summaries. `docs/references.md` records what remains
  unverified.

### Testing

- **`tests/test_end_to_end/` runs real `optimize()` studies.** Until now
  nothing in `tests/` ran one: `test_api.py` patches out `GenericObjective`,
  `create_study`, `optimize_until`, `check_pipeline` and
  `generate_validation_dataset`, and `test_direction_end_to_end.py` drives a
  real Optuna study over hand-fed metric values without building an
  approximator. Both #72 defects were integration failures in the seam neither
  covers. The new directory builds real approximators and runs the real
  validation pipeline against a tiny Gaussian model.

  These tests import BayesFlow and Keras, which
  `docs/plans/plan-testing-gaps-done.md` had ruled out for `tests/`. Both are
  already hard runtime dependencies and `tests/test_builders/test_workflow.py`
  already imports Keras, so this widens an existing precedent rather than
  adding a dependency.

  They add ~210s to the suite and are marked `endtoend`; deselect them with
  `pytest -m "not endtoend"`. They do **not** cover pruning, intermediate
  validation, open-ended stopping (unreachable at two epochs) or
  persistence/resume (studies run in memory).

## 0.2.0

Not a patch release. Re-running an unchanged configuration can produce
different scores, some 0.1.0 studies are refused rather than resumed, and five
previously accepted API usages now raise. Each is deliberate; the alternative
was continuing to rank silently wrong.

Read **Defaults that change results** even if none of the metric corrections
applies to you — the training default, the training search space, and the
learning-rate schedule all changed, and each moves scores on its own.

### Scoring corrections

- **`log_gamma` and `contraction` have recorded directions.** 0.1.0's
  higher-is-better set contained `correlation` alone, so both were minimized
  raw: a search over `log_gamma` selected the *most* miscalibrated model, and
  a search over `contraction` selected the model that learned least. The
  periodic-validation callback passed the same un-converted values to
  minimize-oriented pruning, so both rankings and pruning decisions can
  reverse. BayesFlow defines `log_gamma < 0` as rejecting rank uniformity;
  `contraction` is a variance ratio where 1 is strong learning.
- **The missing-metric penalty is a raw-space value.** In 0.1.0 a missing
  metric took the reported `calibration_error`, or a flat `1.0` when that was
  absent too, with no direction conversion. For `log_gamma` that meant a
  trial reporting nothing scored `1.0` while a trial reporting a genuinely
  good `log_gamma` of `2.0` scored `2.0` under minimization — the missing
  value won. The penalty is now the metric's own recorded worst case,
  injected in raw space and converted once. See **Known limitations** for
  where this guarantee still does not hold.
- **An unregistered metric takes `+inf`, not a finite penalty.** Nothing is
  known about its scale, so no finite constant is defensible: with `1.0`, a
  custom RMSE-like metric reporting `100.0` lost to a trial that reported
  nothing.
- **Metric aliases resolve at the `optimize()` boundary.** `cal_error` — a
  registered, documented alias — reached `check_pipeline()` un-canonicalized,
  so pre-flight compared it against the emitted `calibration_error` key and
  rejected the whole run before any trial. `optimize()` now canonicalizes
  first. `extract_objective_values` and `extract_multi_objective_values` had
  the same gap downstream, though 0.1.0's failure there was quieter than a
  flat `inf`: both fell back to the reported `calibration_error`, so
  `cal_error` came out *right by accident* while `corr` silently returned
  calibration_error's value in correlation's place, and the multi-objective
  form returned the flat `1.0` default. (The `inf` arises only once
  cross-metric substitution is removed — see the penalty entry above.) A
  directly constructed
  `PeriodicValidationCallback` failed differently rather than identically: its
  lightweight validation found no literal key, logged the miss and returned
  `None`, silently disabling that pruning step — or, for an aliased
  `("primary", ...)` target, raised `KeyError`.
- **Collision handling no longer depends on dict ordering.** A `validate_fn`
  emitting both spellings of one metric resolved to whichever came last, so a
  trial's score — and, through the periodic-validation callback, its pruning —
  turned on the insertion order of a dict the caller happened to build. All
  four boundaries that re-key a summary now keep the canonical entry:
  `check_pipeline`, `_validate_metric_keys`, the callback, and the extractors.
- **Coverage ranks are normalized as `(rank + 0.5) / (n_samples + 1)`**,
  previously `rank / (n_samples + 1)`. This changes reported `coverage_*` and
  calibration-error values for unchanged validation data, and therefore any
  constraint built on them.
- **The failed-validation fallback is restricted.** When final validation
  raises, the clamped training loss is substituted only for metrics whose
  recorded direction is lower-is-better with a unit worst case. `log_gamma`,
  `correlation`, `contraction` and `sbc_chi2` now receive their worst
  objective value instead. So do `mae` and any *registered* custom metric
  with no `METRIC_DIRECTIONS` entry, not only unregistered names: absence of
  a recorded direction refuses the substitution and yields `+inf` where 0.1.0
  gave the clamped loss. This changes stored scores and Pareto membership for
  failed trials. `rmse` and `nrmse` still accept the proxy — see **Known
  limitations**.
- **The trial-*failure* penalty is per-metric.** Distinct from the two
  penalties above: this is what a trial stores when it never reaches
  validation at all — a failed build, a rejected compile, a param-count or
  memory rejection, or the new invalid-budget rejections. 0.1.0 wrote a flat
  `1.0` into every objective slot; each slot now takes that metric's own
  worst objective value. For `log_gamma`, `mae` and `sbc_chi2` the stored
  value moves from `1.0` to `+inf`, and for `correlation` from `1.0` to
  `2.0`, so failed trials stop looking ordinary to the sampler. Default
  objectives (`calibration_error`, `nrmse`) are unaffected.
- **`denormalize_param_count()` round-trips with `normalize_param_count()`.**
  With a custom `max_count` below one million and the default `min_count`,
  0.1.0 skipped the auto-tightened lower bound that the forward direction
  applies, so the pair disagreed and the decoded raw count was wrong. The same
  stored normalized value can now decode to a different parameter count.

### Defaults that change results

- **`training_mode="fixed_budget"` is the new default.** 0.1.0 attached
  training-loss early stopping with patience 5 to every trial; the default now
  disables that and runs the schedule to its full horizon, so the same
  configuration can consume substantially more updates.
- **Fixed-budget mode adds a 5% linear warmup.** `lr_warmup_fraction` defaults
  to `0.05` before cosine decay, where 0.1.0 started at `initial_lr`
  immediately. The optimizer trajectory therefore changes even for a trial
  that would never have triggered 0.1.0's early stopping. Pass
  `lr_warmup_fraction=0` to retain the old curve.
- **`TrainingSpace()` explores a different space.** `batch_size` changed from
  the constant `256` to a tuned integer over 32–256 in steps of 32, and the
  upper bound of `initial_lr` doubled from `5e-3` to `1e-2`. This changes the
  search dimensionality and the sampled workloads independently of
  `training_mode`.
- **Registered non-default objectives are now computed.** For the built-in
  validation path, 0.1.0 computed only `DEFAULT_METRICS`, so a configuration
  optimizing `sbc_ks`, `sbc_chi2`, `mae` or `log_gamma` was either rejected by
  pre-flight or scored on a substituted penalty rather than a measurement.
  Those configurations now run and score real values.
- **Constraints on non-default metrics now bind — under the built-in
  validator.** A hard or soft constraint naming a metric or diagnostic output
  outside `DEFAULT_METRICS` — `sbc_ks`, `left_coverage_90` and similar — was
  never computed, so hard constraints silently skipped it and soft constraints
  read zero violation. The built-in pipeline is now producer-aware and
  computes it. **A caller-supplied `validate_fn` bypasses this entirely:**
  only objective keys are required of the hook, so a constrained key it omits
  is still skipped or read as zero violation. Custom hooks must return every
  constrained key themselves.
- **Aliased constraint names now match.** A constraint written as
  `metric_constraints_hard=[("cal_error", 0.05, "above")]`, or its soft
  equivalent, was compared literally against the canonical trial attribute, so
  the hard path skipped it and the soft path read zero violation — configured,
  inactive and silent. `ObjectiveConfig` canonicalizes both constraint lists
  and `optimize()` canonicalizes the soft list before it reaches
  `create_study`, so the same configuration now affects feasibility and
  sampling.
- **Sampled training budgets take precedence over the config fallbacks.**
  0.1.0 overwrote `epochs` and `num_batches` unconditionally with the
  `ObjectiveConfig` values, so a custom search space that already
  returned either had it silently discarded and every trial ran the same fixed
  budget. (`DerivedDimension` is new in 0.2.0, not a 0.1.0 facility whose
  behaviour changed; it is a beneficiary of the new precedence, letting a
  budget be computed from other sampled values.) Both are now applied with
  `setdefault()`, and the optimizer schedule and `train_fn` are built from the
  resulting values. Such
  a search space therefore moves from one fixed budget to trial-specific ones,
  changing both scores and cost; a sampled budget below 1 is now rejected as
  `invalid_training_budget` rather than silently replaced.
- **`GenericObjective` built directly resolves `pruning_n_startup_trials`.**
  Left at `None`, the default dominance pruner raised `TypeError`, which the
  trial handler converted into a failure penalty. It now resolves to 5, so
  trials that were being penalized train normally. Only affects direct
  construction, not `optimize()`.
- **`FlowMatchingSpace.quality()` reverses a categorical's choice order.**
  `fm_use_optimal_transport` moved from `CategoricalDimension(choices=[False,
  True])` to the new `BoolDimension`, which suggests over `[True, False]`.
  Optuna stores choice *order* as part of a categorical distribution and
  refuses a changed sequence as a dynamic value space, so a 0.1.0 study built
  from this profile raises as soon as a new trial is suggested — the guard
  does not catch it first, because nothing about the objective encoding
  changed. (Verified against Optuna 4.9.0:
  `CategoricalDistribution([False, True]) != CategoricalDistribution([True,
  False])`; see `docs/references.md`.) Start a new study, or pin the dimension back to
  `CategoricalDimension("fm_use_optimal_transport", choices=[False, True])`.
- **`TimeSeriesTransformerSpace` builds at every layer count.** 0.1.0 sized
  `embed_dims`, `num_heads` and `mlp_widths` to the sampled `num_layers` while
  leaving `mlp_depths` at length two, so any sampled layer count other than
  two failed to build. A `tst_mlp_depth` dimension was added and the tuple is
  now sized correctly. Searches that previously failed now run; a saved
  parameter dictionary rebuilt directly needs the new key.

### API changes that raise

- **`directions` must be all-`minimize`.** `optimize()` rejects any explicit
  `directions` list containing `"maximize"`, and the resume guard rejects a
  stored study with a non-minimize direction. 0.1.0 accepted `"maximize"` as
  the higher-is-better workaround; because penalties and conversions are now
  in minimize space, layering `"maximize"` on top inverts them a second time
  and the search prefers the worst trials. **Migration:** register the
  metric's direction (`register_metric_direction`) and leave `directions=None`.
- **Diagnostic-kind metrics are rejected as objectives.** `optimize()` raises
  for `correlation` (and its `corr` alias), `bias`, `z_score`, `coverage` (and
  its `coverage_two_sided` alias), `coverage_left`, `coverage_right`, and the
  deprecated `sbc` producer. They remain *available*, but are not all computed
  automatically: the built-in pipeline runs `DEFAULT_METRICS` plus the
  producers of your objectives and constrained keys, and only `correlation`
  and `coverage` are defaults. Correlation
  measures linear association rather than agreement (Bland & Altman, 1986;
  see `docs/references.md`), so
  it rewards a model whose estimates are perfectly correlated with the truth
  and systematically wrong; signed `bias` has its optimum at zero rather than
  at negative infinity, so minimizing it drives the search toward ever more
  severe underestimation. An old study optimizing any of these cannot be
  resumed through `optimize()`. **Migration:** optimize `nrmse` (or `mae`,
  `sbc_ks`, `calibration_error`) and read the diagnostic outputs — the
  `coverage_*` keys with `mean_cal_error`, the `left_*`/`right_*` keys with
  `left_mean_cal_error`/`right_mean_cal_error`, and `mean_z_score` /
  `mean_abs_z_score`. To get a diagnostic that is not a default —
  `bias`, `z_score`, `coverage_left`, `coverage_right` — name one of its
  output keys in a constraint so the producer is scheduled, or return it from
  your own `validate_fn`. Prefer `nrmse` over `rmse`, and see **Known
  limitations** before making either an objective.
- **`early_stopping_patience` is rejected in fixed-budget mode.** Because
  `fixed_budget` is now the default, a previously ordinary call such as
  `optimize(..., early_stopping_patience=5)` raises from `ObjectiveConfig`:
  finite-horizon cosine annealing must run to its horizon. **Migration:** drop
  the argument, or pass `training_mode="open_ended"`. Note that `open_ended`
  is *not* 0.1.0's behaviour — it is the nearest available early-stopping
  mode. 0.1.0 used cosine decay with no warmup and stopped on a per-epoch
  moving average of training loss; `open_ended` uses inverse-square-root decay
  after a one-epoch warmup and stops on periodically evaluated validation
  objectives, which can give materially different update counts, stopping
  times and restored weights.
- **Duplicate dimension names raise.** `CompositeSearchSpace` rejects a
  parameter name reused across the inference, summary and training
  components, which 0.1.0 accepted with later dictionaries silently
  overwriting earlier values. `BaseSearchSpace.dimensions` additionally
  rejects two fields of a *single* space sharing one `Dimension.name`, before
  sampling — so a standalone custom space that 0.1.0 accepted can now raise on
  its own. **Migration:** give every dimension a unique name.

- **`register_metric(kind=...)` validates its argument.** 0.1.0 stored
  whatever string it was given, so a typo such as `kind="diagnostics"` made
  the metric neither objective nor diagnostic and behaved unpredictably at
  the kind checks. Anything but `"objective"` or `"diagnostic"` now raises.

Note that the diagnostic-kind rejection above fires from `check_pipeline()`
as well as `optimize()`, so a direct pre-flight call with a diagnostic
objective raises too.

### Resume safety

`create_study` and the resume guard refuse studies whose stored values cannot
be shown to mean what this run produces, rather than silently mixing scales or
columns:

- A study whose metrics changed encoding at 0.2.0 is refused: `log_gamma`,
  `correlation`, `sbc_chi2`, `mae` and `contraction`. `contraction` was
  wrongly classed as encoding-unchanged until this release — the audit that
  derives the classification was anchored to a mid-series commit in which it
  had *already* been given its direction, so a legacy study could resume and
  mix raw values with `1 - value` in one column. The audit is now anchored to
  released 0.1.0.
- A study whose objective schema differs from the current run is refused;
  Optuna addresses objectives by position, so continuing would compare one
  metric against another in the same column.
- A populated study recording no schema is refused. On the **ordinary resume**
  path, Optuna's own persisted `metric_names` count as schema evidence where
  the `bayesflow_hpo_objective_schema` user attribute is absent. This fallback
  is *not* applied to a warm-start source: `create_study` reads only the user
  attribute there, so a source carrying Optuna labels alone is treated as
  schema-less.
- A warm-start source is validated before its trials are copied — **when
  `metric_names` is supplied**. `create_study(warm_start_from=...)` defaults
  `metric_names` to `None`, and the schema-less-source rejection is guarded on
  it, so that call can still copy trials from a source recording no schema.
  Only the **schema** is validated pre-copy, never the encoding: a
  schema-matching legacy source is copied into the target first, and
  `_guard_resumed_study` rejects it afterwards — leaving the target already
  mutated. This applies through `optimize()` too, which supplies
  `metric_names`. Warm-start from a study you know carries the current
  encoding, or into a throwaway target.

Mean-mode objective columns compare independently of **member order**, so a
study stamped by an earlier build is not refused for ordering alone. Alias
spellings are *not* normalized in schema comparison: a stored
`mean(cal_error+nrmse)` does not match a current `mean(calibration_error+nrmse)`
and is refused.

### Public constants

- **`HIGHER_IS_BETTER` holds different names.** The set is now derived from
  `METRIC_DIRECTIONS`, so it contains `correlation`, `contraction` and
  `log_gamma` where 0.1.0 held `correlation` alone. It remains the documented
  mutation point for custom metrics, and removal now has defined semantics:
  `HIGHER_IS_BETTER.discard("contraction")` suppresses the conversion rather
  than being overridden by the table. Code that iterates or copies this set
  sees three entries where it saw one.

### Result analysis

- **`select_best_trial()` restricts selection to the Pareto front.** Its final
  mean-rank tiebreak previously ran over all surviving candidates and could
  return a dominated trial. The same stored study can now yield a different
  best trial.
- **A resumed study may lose its objective column labels.** 0.1.0 stamped
  `metric_names` unconditionally; labelling is now skipped when the study
  already holds trials, because relabelling a populated study would silently
  reinterpret its stored columns. If such a study recorded its provenance in
  the `bayesflow_hpo_objective_schema` user attribute rather than in Optuna's
  own `metric_names`, result tables and plots fall back to `objective_0`,
  `objective_1`, … where 0.1.0 showed metric names. The values are unchanged;
  only the column headings are.
- **The checkpoint pool ranks by the mean of all metric objectives** excluding
  cost, previously by the first objective alone. Multi-objective runs may
  retain and evict different weights, so a bounded pool can hold different
  artifacts.

### Reliability

- **A `MemoryError` from the exact parameter-count probe now cleans up before
  propagating.** 0.1.0 re-raised immediately without calling `cleanup_trial()`,
  leaving CUDA state and cached allocations behind, so a caller that caught
  the exception and retried inherited the exhausted device.

### Known limitations

- **`rmse` and `nrmse` can still be outranked by a missing value, and still
  accept the failed-validation proxy.** Both record `worst_raw = 1.0`, so a
  trial that never reported the metric scores `1.0` and beats one reporting
  `5.0`; and because both are lower-is-better with a unit worst case, a
  clamped training loss of `0.1` is substituted for them when final validation
  raises, which can promote a failed trial over a valid one. `rmse` is in
  parameter units and unbounded; `nrmse` is range-normalized but can still
  exceed 1 when prediction error exceeds the normalization range. `nrmse` is a
  default objective, so this applies to standard configurations. Among the
  unit-worst metrics only `rmse` and `nrmse` are unbounded this way;
  `calibration_error` (a mean absolute deviation between probabilities) and
  `sbc_ks` (a supremum of a CDF difference) genuinely are bounded by `1.0`,
  which makes their *penalty* sound. It does not make the proxy safe: those
  two accept it as well, so a failed trial storing a clamped `0.1` still
  outranks a valid `0.5`. Every metric that takes the proxy can promote a
  failed trial; the bound only settles what a *missing* value is worth.
  Ensure your validation hook reliably produces every objective metric.
- **Concurrent workers can stamp conflicting schemas.** The schema read and
  the schema write are not a single transaction and there is no storage lock
  around them, so two workers starting the same empty shared-storage study
  with different objective schemas can both observe no schema, both pass their
  local checks, and then train under different positional meanings in one
  study. Stamp the study from a single process before launching parallel
  workers.
- **A legacy study optimizing `sbc_ks` may resume unsafely.** Before 0.2.0,
  final validation computed only `DEFAULT_METRICS`, which excludes `sbc_ks`,
  so those trials stored the substituted penalty rather than a measurement.
  `sbc_ks` is treated as encoding-unchanged, so such a study can pass the
  resume guard and mix old penalties with new measurements in one column.
  Start a new study if you optimized a non-default metric under 0.1.0.

### Internal

- mypy runs in CI and is blocking.
- `CanonicalMetricName`, `RawScore` and `MinimizeScore` make the two recurring
  defect classes — an un-canonicalized metric name, and a minimize-space value
  in a raw-space slot — type errors rather than silent misrankings.

## 0.1.0

Never published. This was the version carried by every commit up to the
corrections above, which is why the `bayesflow-hpo>=0.1.0` floor declared by
dependents could not distinguish fixed code from unfixed.
