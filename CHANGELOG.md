# Changelog

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
  changed. Start a new study, or pin the dimension back to
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
  measures linear association rather than agreement (Bland & Altman, 1986), so
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
  so the penalty and the proxy are sound for them. Ensure your validation hook
  reliably produces every objective metric.
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
