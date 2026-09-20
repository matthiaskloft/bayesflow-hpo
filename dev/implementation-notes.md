# Implementation notes

## Issue #118 — validation aggregation

- Main had no scalar `aggregate` option when #118 was implemented; preserve its
  historical arithmetic mean as the default. Explicit metric mappings retain
  the parameter-by-condition grid; scalar reductions retain parameter averaging.
- Geometric-domain errors must escape the objective and intermediate-validation
  catch-all handlers, or an invalid reduction silently becomes a fallback score.
- Aggregation changes score meaning, so `optimize()` records its configuration
  and checks resume/warm-start compatibility before comparing trials.
