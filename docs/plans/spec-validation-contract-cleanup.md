# Spec: Validation Contract Cleanup (Package D remaining)

## Summary

Three cleanup items left over from Package D's `optimize()` refactor
(PR #49). Two are documentation-only; one is a small code change to
tighten types. Together they close the gap between the implicit
`validate_fn` contract and what's documented/enforced at the type level.

## Requirements

### 1. Document `validate_fn` return contract

**Decision: docstring-only, no code changes.**

Add contract details to three locations:

- **`ValidateFn` type alias** (`types.py:25-26`): document that the
  returned `dict[str, float]` must contain all keys listed in
  `objective_metrics`. Missing keys → penalty substitution. NaN/Inf →
  penalty substitution. Extra keys are preserved as `trial.user_attrs`.
- **`optimize()` docstring** (`api.py:169-172`): expand the
  `validate_fn` parameter docs with the same contract.
- **`ObjectiveConfig` docstring** (`objective.py:219-221`): same.

Note: intermediate pruning currently hard-codes
`["calibration_error", "nrmse"]` regardless of `objective_metrics`.
This coupling is documented as a known limitation; Package A1 will
make it configurable.

### 2. Document timing semantics limitation

**Decision: document the limitation, no code changes.**

Add a note to the `validate_fn` parameter docs in `optimize()` and
`ValidateFn`:

> When using a custom `validate_fn`, the `inference_time` cost metric
> measures wall-clock time for the entire `validate_fn` call (inference
> + metric computation). The default path measures pure inference time
> only. This makes `cost_metric="inference_time"` not directly
> comparable between default and custom validation. Consider using
> `cost_metric="param_count"` when comparing studies that mix both
> paths.

### 3. Make `validation_data` non-Optional in `ObjectiveConfig`

**Decision: tighten the type, remove dead code.**

Changes:
- `ObjectiveConfig.validation_data`: change type from
  `ValidationDataset | None` to `ValidationDataset` (remove default
  `None`)
- Remove the `else` branch in `GenericObjective.__call__` that
  substitutes penalty values when `validation_data is None`
  (objective.py:660-666)
- The `if config.validation_data is not None` guard around
  `PeriodicValidationCallback` injection (objective.py:573) becomes
  always-true — remove the guard (always inject the callback)
- The `if config.validation_data is not None` guard in the validation
  step (objective.py:616) also becomes always-true — remove the guard

Rationale: `optimize()` always generates validation data before
constructing `ObjectiveConfig`. No external code constructs
`ObjectiveConfig` directly. The `None` path was dead code.

## Design Decisions

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| validate_fn contract | Docstring-only | Docstring + runtime warning on hook registration |
| Timing semantics | Document limitation | Change ValidateFn signature to return timing; drop inference_time for custom path |
| validation_data optionality | Make non-Optional | Keep Optional + document; expose validate=False in optimize() |

## Scope

**In scope:**
- Docstring updates to `ValidateFn`, `optimize()`, `ObjectiveConfig`
- Type change for `ObjectiveConfig.validation_data`
- Remove dead `None` fallback code in `GenericObjective.__call__`
- Remove now-redundant `is not None` guards

**Out of scope:**
- Making intermediate validation metrics configurable (Package A1)
- Adding `validate=False` to `optimize()` (future work if needed)
- Changing `ValidateFn` signature (would be a breaking change)

## Architecture Overview

Small, localized changes across 3 files:

```
types.py          — docstring update to ValidateFn
api.py            — docstring update to optimize()
objective.py      — docstring update to ObjectiveConfig
                  — type change: ValidationDataset | None → ValidationDataset
                  — remove None fallback branch
                  — remove is-not-None guards
```

No new modules, no new dependencies, no behavioral changes.

## Constraints

- No breaking changes to `optimize()` public API
- `ObjectiveConfig` type tightening is internal (no external users)
- Must pass existing test suite without modification (unless tests
  explicitly construct `ObjectiveConfig(validation_data=None)`)

## Open Questions

None — all decisions resolved during brainstorm.
