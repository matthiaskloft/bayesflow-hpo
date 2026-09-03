# Changelog

## 0.2.0

Not a patch release: metric values stored by 0.1.0 studies do not mean the
same thing here, and some 0.1.0 studies are now refused rather than resumed.
Both are deliberate — the alternative was continuing to rank silently wrong.

### Scoring corrections

- **`log_gamma` now has a recorded direction.** BayesFlow defines it so that
  `log_gamma < 0` rejects rank uniformity, meaning larger is better — the
  opposite of every other built-in metric. Without a direction Optuna
  minimized it, so a search over `log_gamma` selected the *most*
  miscalibrated model in the study while every number in the output looked
  ordinary (#72).
- **Missing metrics no longer outrank reported ones.** The substituted
  penalty is injected before direction conversion, so it must be a raw-space
  value. A flat `1.0` for `log_gamma` converted to `-1.0`, which beat a
  genuinely reported `0.5` — a trial that failed to report the metric won
  against one that reported a good value (#72).
- **An unregistered metric takes `+inf`, not a finite penalty.** Nothing is
  known about its scale, so no finite constant is defensible: with `1.0`, a
  custom RMSE-like metric reporting `100.0` lost to a trial that reported
  nothing (#72).
- **Metric aliases resolve at the public extractors.** `cal_error` — a
  registered, documented alias — returned `inf` instead of the reported
  value from `extract_objective_values` and `extract_multi_objective_values`.
  Since `inf` is the unregistered-name penalty, every trial tied at the worst
  possible score and the objective went flat, with nothing logged (#79).
- **Collision handling no longer depends on dict ordering.** A `validate_fn`
  emitting both spellings of one metric resolved to whichever came last, so a
  trial's score — and pruning, through the periodic-validation callback —
  turned on the insertion order of a dict the caller happened to build. All
  boundaries now keep the canonical entry (#79).

### Resume safety

`create_study` and the resume guard refuse studies whose stored values cannot
be shown to mean what this run produces, rather than silently mixing scales or
columns (#72, #78):

- A study whose metrics changed encoding at 0.2.0 is refused.
- A study whose objective schema differs from the current run is refused;
  Optuna addresses objectives by position, so continuing would compare one
  metric against another in the same column.
- A populated study recording no schema is refused. Optuna's own persisted
  `metric_names` count as schema evidence where present.
- A warm-start source is validated *before* its trials are copied, including
  sources that record no schema at all — copying those left the target
  populated and unstamped, a state that can never be resumed.

Mean-mode objective columns compare independently of member order, so a study
stamped by an earlier build is not refused over spelling alone.

### Other

- `PeriodicValidationCallback` canonicalizes its own metric names. It was
  protected when built through the pipeline and unprotected when constructed
  directly, which its presence in `__all__` invites (#79).
- mypy runs in CI and is blocking (#73, #74).
- `CanonicalMetricName`, `RawScore` and `MinimizeScore` make the two recurring
  defect classes — an un-canonicalized metric name, and a minimize-space value
  in a raw-space slot — type errors rather than silent misrankings (#79).

## 0.1.0

Never published. This was the version carried by every commit up to the
scoring corrections above, which is why the `bayesflow-hpo>=0.1.0` floor
declared by dependents could not distinguish fixed from unfixed code.
