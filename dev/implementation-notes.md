# Implementation notes

## Issue #118 — validation aggregation

- Main had no scalar `aggregate` option when #118 was implemented; preserve its
  historical arithmetic mean as the default. Explicit metric mappings retain
  the parameter-by-condition grid; scalar reductions retain parameter averaging.
- Geometric-domain errors must escape the objective and intermediate-validation
  catch-all handlers, or an invalid reduction silently becomes a fallback score.
- Aggregation changes score meaning, so `optimize()` records its configuration
  and checks resume/warm-start compatibility before comparing trials.

- The aggregation guard also belongs at the start of `GenericObjective`, since
  advanced callers bypass `optimize()`. Exclude only the current trial from
  the legacy-study check: Optuna creates it before invoking the objective.

## Issue #147: parallel workers

Optuna 5.0.0 QMCSampler's study-attribute counter is a non-atomic read/write.
Use the storage-assigned trial number for Sobol indexing instead, and refresh
warm-up progress from shared completed trials at each before_trial call.
The override depends on Optuna 5's private _find_sample_id method (dependency
is pinned below 6); the multiprocessing test checks the exact Sobol indices.
`constant_liar=False` is the requested public default, although Optuna 5's
own default has changed to True. Preflight has no built-in file writes;
custom hooks must isolate their output paths across processes.
