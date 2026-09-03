# Changelog

## 0.2.0

Not a patch release. Re-running an unchanged configuration can produce
different scores, some 0.1.0 studies are refused rather than resumed, and two
previously accepted API usages now raise. Each is deliberate; the alternative
was continuing to rank silently wrong.

Read **Defaults that change results** even if none of the metric corrections
applies to you — the training default changed, and that alone moves scores.

### Scoring corrections

- **`log_gamma` has a recorded direction.** BayesFlow defines it so that
  `log_gamma < 0` rejects rank uniformity, meaning larger is better. Without a
  recorded direction Optuna minimized it, so a search over `log_gamma`
  selected the *most* miscalibrated model in the study while every number in
  the output looked ordinary. (`contraction` and `correlation` are also
  higher-is-better; `log_gamma` was the one whose direction was missing.)
- **The missing-metric penalty is a raw-space value.** It is injected before
  direction conversion, so a minimize-space value was converted twice: a flat
  `1.0` for `log_gamma` became `-1.0`, which beat a genuinely reported `0.5`.
  A trial that failed to report the metric outranked one that reported a good
  value. See **Known limitations** for where this guarantee still does not
  hold.
- **An unregistered metric takes `+inf`, not a finite penalty.** Nothing is
  known about its scale, so no finite constant is defensible: with `1.0`, a
  custom RMSE-like metric reporting `100.0` lost to a trial that reported
  nothing.
- **Metric aliases resolve at the public extractors.** `cal_error` — a
  registered, documented alias — returned `inf` from `extract_objective_values`
  and `extract_multi_objective_values` instead of the reported value. Since
  `inf` is the unregistered-name penalty, every trial tied at the worst
  possible score and the objective went flat, with nothing logged.
  `PeriodicValidationCallback` had the same gap when constructed directly.
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
  raises, the clamped training loss is substituted only for metrics that are
  explicitly unit-scaled and lower-is-better. `log_gamma`, custom metrics and
  other incompatible scales now receive their worst objective value instead.
  This changes stored scores and Pareto membership for failed trials.

### Defaults that change results

- **`training_mode="fixed_budget"` is the new default.** 0.1.0 attached
  training-loss early stopping with patience 5 to every trial; the default now
  disables that and runs the cosine schedule to its full horizon. The same
  configuration can therefore consume substantially more updates and score
  differently even when no metric correction above applies. Pass
  `training_mode="open_ended"` for the old behaviour.
- **Constraints on non-default metrics now bind.** A hard or soft constraint
  naming a metric or diagnostic output outside `DEFAULT_METRICS` — `sbc_ks`,
  `left_coverage_90` and similar — was never computed, so hard constraints
  silently skipped it and soft constraints read zero violation. The pipeline
  is now producer-aware and computes it, which can change feasibility and
  sampling for an existing configuration.
- **`GenericObjective` built directly resolves `pruning_n_startup_trials`.**
  Left at `None`, the default dominance pruner raised `TypeError`, which the
  trial handler converted into a failure penalty. It now resolves to 5, so
  trials that were being penalized train normally. Only affects direct
  construction, not `optimize()`.

### API changes that raise

- **`directions` must be all-`minimize`.** `optimize()` rejects any explicit
  `directions` list containing `"maximize"`, and the resume guard rejects a
  stored study with a non-minimize direction. 0.1.0 accepted `"maximize"` as
  the higher-is-better workaround; because penalties and conversions are now
  in minimize space, layering `"maximize"` on top inverts them a second time
  and the search prefers the worst trials. **Migration:** register the
  metric's direction (`register_metric_direction`) and leave `directions=None`.
- **`correlation` and its `corr` alias are diagnostic-only.** `optimize()`
  rejects either in `objective_metrics`. They are still computed and reported;
  they can no longer be optimized, and an old correlation-objective study
  cannot be resumed through `optimize()`. Correlation measures linear
  association rather than agreement (Bland & Altman, 1986), so it rewards a
  model whose estimates are perfectly correlated with the truth and
  systematically wrong. **Migration:** optimize `rmse` or `nrmse` and read
  `correlation` as a diagnostic.

### Resume safety

`create_study` and the resume guard refuse studies whose stored values cannot
be shown to mean what this run produces, rather than silently mixing scales or
columns:

- A study whose metrics changed encoding at 0.2.0 is refused.
- A study whose objective schema differs from the current run is refused;
  Optuna addresses objectives by position, so continuing would compare one
  metric against another in the same column.
- A populated study recording no schema is refused. Optuna's own persisted
  `metric_names` count as schema evidence where present.
- A warm-start source is validated before its trials are copied — **when
  `metric_names` is supplied**. `create_study(warm_start_from=...)` defaults
  `metric_names` to `None`, and the schema-less-source rejection is guarded on
  it, so that call can still copy trials from a source recording no schema;
  source *encoding* is checked later, by `optimize()`, after copying.
  `optimize()` supplies `metric_names`, so its paths are validated pre-copy.

Mean-mode objective columns compare independently of **member order**, so a
study stamped by an earlier build is not refused for ordering alone. Alias
spellings are *not* normalized in schema comparison: a stored
`mean(cal_error+nrmse)` does not match a current `mean(calibration_error+nrmse)`
and is refused.

### Result analysis

- **`select_best_trial()` restricts selection to the Pareto front.** Its final
  mean-rank tiebreak previously ran over all surviving candidates and could
  return a dominated trial. The same stored study can now yield a different
  best trial.
- **The checkpoint pool ranks by the mean of all metric objectives** excluding
  cost, previously by the first objective alone. Multi-objective runs may
  retain and evict different weights, so a bounded pool can hold different
  artifacts.

### Known limitations

- **`rmse` can still be outranked by a missing value.** Its recorded
  `worst_raw` is `1.0`, so a trial that never reported `rmse` scores `1.0` and
  beats one reporting `5.0`. `rmse` is unbounded, so no finite penalty is
  correct; `nrmse` and `sbc_ks` share the `1.0` bound but are themselves
  bounded, so it is sound for them. Prefer `nrmse` as an objective.
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
