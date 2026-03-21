# Spec: Package I — Small API Fixes

Three standalone fixes that improve error reporting and edge-case
correctness. No public API signature changes.

## Summary

Package I addresses three small gaps:

1. **`normalize_param_count` edge case** — contradictory bounds
   (`max_count <= min_count`) silently return `0.0` (best score),
   masking a configuration error.
2. **Silent adapter inference fallback** — when an adapter has no
   `transforms` attribute, `infer_keys_from_adapter` returns all-`None`
   with no logging, making key inference failures hard to debug.
3. **Silent data key mismatch** — `make_bayesflow_infer_fn` silently
   skips missing `data_keys` via `if k in sim_data`, producing garbage
   posterior draws without any error.

## Requirements

### R1: `normalize_param_count` — raise on invalid bounds

**File:** `objectives.py:95-96`

When `max_count <= min_count` (after auto-tightening), raise
`ValueError` with a message like:

> `"max_count ({max_count}) must be greater than min_count ({min_count})"`

**Rationale:** Contradictory bounds are always a caller mistake.
Returning `0.0` silently collapses the cost axis on the Pareto front,
making the sampler unable to learn model-size preferences. Failing
loudly is preferable to silent misoptimization.

**Edge cases to preserve:**
- `param_count <= 0` → still returns `1.0` (worst score, broken model)
- `min_count <= 0` → still auto-corrected to `1`
- Auto-tightening (`min_count == MIN_PARAM_COUNT and max_count < MAX_PARAM_COUNT`) runs before the check

### R2: Debug logging in `infer_keys_from_adapter`

**File:** `api.py:63-65`

Add a single `logger.debug(...)` message when `transforms is None`:

```
"Adapter has no 'transforms' attribute; skipping key inference"
```

Minimal scope — one log line on the fallback path only. No logging
for the happy path or per-transform matches.

### R3: Validate `data_keys` exist in `make_bayesflow_infer_fn`

**File:** `validation/inference.py:20-32`

At the top of `make_bayesflow_infer_fn`, accept an optional
`available_keys: set[str] | None` parameter. When provided, check
that every key in `data_keys` is in `available_keys`. Raise
`KeyError` with a clear message listing the missing keys and
available keys.

The caller (`run_validation_pipeline` at `validation/pipeline.py:52`)
passes `available_keys` derived from the first simulation dict's keys:

```python
available_keys = set(validation_data.simulations[0].keys())
```

Inside the `infer_fn` closure, remove the `if k in sim_data` guard
and use direct dict access:

```python
conditions = {k: sim_data[k] for k in data_keys}
```

This turns a silent data-loss bug into an immediate `KeyError`.

**Backwards compatibility:** The `available_keys` parameter defaults
to `None` (no check), so direct callers of `make_bayesflow_infer_fn`
who don't pass it continue to work — but the silent skip is still
removed from the closure itself (direct `dict[key]` access raises
`KeyError` naturally).

## Design Decisions

### D1: #5 (explicit `param_keys`/`data_keys` in `optimize()`) — dropped

Dropped from Package I. The feature would need to also support
`inference_conditions` as a third key group (models with both a
summary network and direct conditioning), which makes it a design-level
change rather than a small fix. Deferred to Package D's `optimize()`
refactor, which already touches this code.

### D2: `ValueError` over neutral return for `normalize_param_count`

Considered returning `0.5` (neutral) as a graceful fallback. Rejected
because contradictory bounds are always a caller mistake, and silent
fallbacks mask misconfigurations that affect optimization quality.

### D3: Minimal logging scope for `infer_keys_from_adapter`

Considered verbose logging (per-transform matches, final result dict).
Rejected to keep noise low — the fallback path is the only case where
debugging is needed. The happy path is self-evident from the returned
keys.

## Scope

### In scope

- `normalize_param_count` guard + `ValueError`
- One `logger.debug()` line in `infer_keys_from_adapter`
- `available_keys` validation in `make_bayesflow_infer_fn`
- Remove silent `if k in sim_data` guard from `infer_fn` closure
- Unit tests for all three changes
- Update `denormalize_param_count` docstring if affected

### Out of scope

- Explicit `param_keys`/`data_keys` parameters on `optimize()`
  (deferred to Package D)
- Three-key support (`inference_conditions` as separate concept)
- Changes to `generate_validation_dataset`
- Any public API signature changes to `optimize()`

## Open Questions

None — all design decisions resolved during brainstorming.
