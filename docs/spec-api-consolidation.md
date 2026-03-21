# Spec: Package C — API Consolidation & Search Space Simplification

## Summary

Align bayesflow-hpo's public API naming with BayesFlow 2.x conventions and
simplify the search space dimension system by replacing the `enabled` /
`include_optional` mechanism with a `constant` field on dimensions.

**Motivation:**

1. The `batches_per_epoch` parameter is misaligned with BayesFlow 2.x, which
   uses `num_batches` in `build_dataset()` and `steps_per_epoch` in Keras.
   The current `default_train_fn` passes `batches_per_epoch` to
   `approximator.fit()`, which silently drops it — `num_batches` (required,
   no default) is never set, causing a runtime crash on BF 2.0.8.

2. Five Optuna dimension names use abbreviated forms that diverge from
   BayesFlow's actual constructor parameter names, making it harder for
   users to map trial results back to BayesFlow docs.

3. The search space system has overlapping mechanisms for fixing
   hyperparameters (`enabled`, `include_optional`, single-point ranges,
   constructor overrides, custom hooks), creating confusion about the
   idiomatic approach.

## Requirements

### R1: Rename `batches_per_epoch` → `num_batches`

Clean break (no deprecation shim). Rename in all locations:

| Location | File | Lines |
|----------|------|-------|
| `optimize()` signature | `api.py` | 115 |
| `ObjectiveConfig` dataclass | `optimization/objective.py` | 228 |
| `default_train_fn()` | `optimization/objective.py` | 71, 82, 90 |
| `check_pipeline()` | `pipeline.py` | 144, 219, 292 |
| LR decay computation | `optimization/objective.py` | 512 |
| Consistency model step calc | `search_spaces/inference/consistency.py` | 25 |
| Builder docstrings | `builders/workflow.py` | 6, 35 |
| Documentation | `docs/optimization.md` | 17, 45, 52 |
| All test fixtures | `tests/` | multiple |

**Bug fix included:** `default_train_fn` must pass `num_batches=` (not
`batches_per_epoch=`) to `approximator.fit()` so it works on BF 2.0.8+
without a custom `train_fn`. Remove the CLAUDE.md gotcha about 2.0.8
compatibility being opt-in.

### R2: Expand abbreviated Optuna dimension names

Align the post-prefix portion with BayesFlow's actual parameter names:

| Current | New | BayesFlow kwarg | File |
|---------|-----|-----------------|------|
| `cf_actnorm` | `cf_use_actnorm` | `use_actnorm` | `coupling_flow.py` |
| `fm_use_ot` | `fm_use_optimal_transport` | `use_optimal_transport` | `flow_matching.py` |
| `fm_time_alpha` | `fm_time_power_law_alpha` | `time_power_law_alpha` | `flow_matching.py` |
| `ds_spectral_norm` | `ds_spectral_normalization` | `spectral_normalization` | `deep_set.py` |
| `st_num_inducing` | `st_num_inducing_points` | `num_inducing_points` | `set_transformer.py` |
| `tst_time_embed` | `tst_time_embedding` | `time_embedding` | `time_series_transformer.py` |

Prefixes (`cf_`, `fm_`, `ds_`, `st_`, `tst_`) are **required** — removing
them would cause silent overwrites when inference + summary params merge
via `dict.update()` in `CompositeSearchSpace.sample()` (e.g., both spaces
define `dropout`).

### R3: Replace `enabled` / `include_optional` with `constant` field

**Remove:**
- `enabled: bool` field on `IntDimension`, `FloatDimension`, `CategoricalDimension`
- `include_optional: bool` field on `BaseSearchSpace` and all 6 subclasses
  that declare it (TrainingSpace, ConsistencyModelSpace, FlowMatchingSpace,
  StableConsistencyModelSpace, FusionTransformerSpace)
- Skip logic in `BaseSearchSpace.sample()`:
  `if not dim.enabled and not self.include_optional: continue`
- Conditional `if key in params` checks in all `build()` methods (~26 sites)
- `_validate()` filtering on `enabled=True` only
- `TrainingSpace.defaults()` method + merge logic in
  `CompositeSearchSpace.sample()`

**Add `constant` field to all Dimension types:**

```python
@dataclass
class FloatDimension:
    name: str
    low: float | None = None
    high: float | None = None
    log: bool = False
    constant: float | None = None  # NEW
```

**Semantics:**
- `constant` and `low`/`high` (or `choices`) are **mutually exclusive** —
  raise `ValueError` if both are set.
- When `constant` is set, `sample()` injects the value directly into the
  params dict **without calling Optuna's `suggest_*()`**. This is more
  efficient (no DB write, no sampler overhead, no dashboard clutter).
- Constants do **not** appear in `trial.params` or `best_config()` output.
  Checkpoints are the source of truth for full model reproduction.

**Migration of currently-optional dimensions:**

All 26 dimensions currently using `enabled=False` become `constant=<BayesFlow default>`.
Users widen the range to tune them by passing a dimension with `low`/`high`
instead of `constant` via the dataclass constructor.

Example:

```python
# Before: optional, skipped by default
cf_actnorm: CategoricalDimension = field(
    default_factory=lambda: CategoricalDimension(
        "cf_actnorm", choices=[True, False], enabled=False
    )
)

# After: fixed at BayesFlow default, user widens to tune
use_actnorm: CategoricalDimension = field(
    default_factory=lambda: CategoricalDimension(
        "cf_use_actnorm", constant=False,
    )
)
# User opts in to tuning:
CouplingFlowSpace(
    use_actnorm=CategoricalDimension("cf_use_actnorm", choices=[True, False])
)
```

### R4: Add `.constants` property on search spaces

```python
@property
def constants(self) -> dict[str, Any]:
    """Return {name: value} for all fixed dimensions."""
    return {d.name: d.constant for d in self.dimensions if d.constant is not None}
```

`CompositeSearchSpace.constants` merges across inference + summary +
training sub-spaces. Enables users or results code to retrieve the full
fixed config without touching Optuna.

### R5: Simplify `build()` methods

With all dimensions always present in the params dict (either sampled or
constant), the conditional `if key in params` pattern in `build()` methods
is no longer needed. All `build()` methods unconditionally read from params.

## Design Decisions

### Clean break vs deprecation shim for `batches_per_epoch`

**Decision:** Clean break.
**Alternatives considered:** Deprecation shim accepting both names.
**Rationale:** Pre-1.0 package with no external users. A shim adds
complexity with no benefit.

### Prefixes on dimension names

**Decision:** Keep prefixes, expand abbreviated portions.
**Alternatives considered:** Drop prefixes entirely; keep abbreviations.
**Rationale:** Prefixes are load-bearing — `CompositeSearchSpace.sample()`
merges via `dict.update()`, so unprefixed names like `dropout` would cause
silent overwrites between inference and summary spaces. Expanding the
post-prefix portion (e.g., `cf_actnorm` → `cf_use_actnorm`) makes the
mapping to BayesFlow kwargs self-documenting.

### `constant` field vs `enabled` / `include_optional`

**Decision:** Replace with `constant`.
**Alternatives considered:** Keep `enabled`/`include_optional` and add docs;
add a `fix()` helper method.
**Rationale:** The `enabled`/`include_optional` two-level system creates
confusion. `constant` is a single, direct mechanism: either a dimension
has a fixed value or it has a range. Constructor overrides let users
switch between the two.

### Constants skip Optuna suggest

**Decision:** Inject directly, skip `trial.suggest_*()`.
**Alternatives considered:** Go through Optuna with `low==high`.
**Rationale:** More efficient (no DB write, no sampler overhead). Constants
don't need to appear in `trial.params` because checkpoints are the source
of truth for reproduction, and `search_space.constants` provides
programmatic access to fixed values.

## Scope

### In scope

- `batches_per_epoch` → `num_batches` rename (all files)
- Fix `default_train_fn` to pass `num_batches` to BF 2.0.8+ `fit()`
- 6 dimension name expansions
- `constant` field on all Dimension types
- Remove `enabled`, `include_optional`, `defaults()`, skip logic
- Simplify all `build()` methods (remove conditional param checks)
- `.constants` property on `BaseSearchSpace` and `CompositeSearchSpace`
- Update all tests
- Update docs (`optimization.md`, `search_spaces.md`, CLAUDE.md gotcha)

### Out of scope

- Other Package C items (#3 normalize_param_count, #4 debug logging,
  #5 explicit param_keys/data_keys) — tracked separately in TODO.md
- Sampler/pruner presets (Package A)
- `optimize()` refactor (Package D)

## Architecture Overview

### Dimension types (base.py)

```
IntDimension    → name, low?, high?, step?, log?, constant?
FloatDimension  → name, low?, high?, log?, constant?
CategoricalDimension → name, choices?, constant?
```

Validation in `__post_init__`: either `constant` is set, or `low`/`high`
(or `choices`) are set. Never both.

### Sample flow

```
CompositeSearchSpace.sample(trial)
  → inference_space.sample(trial)
     → for dim in dimensions:
          if dim.constant is not None:
              params[dim.name] = dim.constant   # skip Optuna
          else:
              params[dim.name] = dim.suggest(trial)  # via Optuna
  → summary_space.sample(trial)   # same logic, merged via update()
  → training_space.sample(trial)  # same logic, merged via update()
  → return params   # all dims present, no gaps
```

### Build flow

```
CouplingFlowSpace.build(params)
  → _validate(params)   # all dimensions are required (no enabled filter)
  → reads ALL dimensions unconditionally from params
  → returns CouplingFlow(...)
```

## Constraints

- Must work with BayesFlow 2.0.8+ (`num_batches` in `build_dataset()`)
- No backwards-compatibility shims (pre-1.0 package)
- All existing tests must be updated and pass
- Ruff lint must pass

## Open Questions

None — all design decisions resolved during brainstorm.
