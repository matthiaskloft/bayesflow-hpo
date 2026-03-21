# Spec: Package D — `optimize()` Refactor (Code Items)

## Summary

Refactor `optimize()` into a fully decomposed orchestrator, fix
`_TrackingDict` false positives, and deduplicate builder registration.
These are internal quality improvements with no public API changes.

## Requirements

1. **Decompose `optimize()` into named helpers** — extract 4 private
   functions so `optimize()` reads as a linear sequence of named steps.
2. **Fix `_TrackingDict` false positives** — override `items()` and
   `values()` so custom builders iterating via these methods don't
   trigger spurious unused-key warnings.
3. **Deduplicate builder registration** — extract
   `_register_with_aliases()` to eliminate the repeated alias loop.
4. ~~Export `default_validate_fn`~~ — already done (line 63 + `__all__`
   lines 148-149 in `__init__.py`). **Dropped.**

## Design Decisions

### 1. `optimize()` decomposition

**Decision:** Extract 4 helpers, making `optimize()` a pure orchestrator.

**Current state:** `optimize()` is ~130 lines (post-Package C). It's
readable but mixes orchestration with inline logic for validation data
setup, objective construction, direction derivation, and study
creation/execution.

**Alternatives considered:**
- *Drop — already clean enough.* Rejected: even small helpers improve
  testability and make the orchestrator self-documenting.
- *Extract only 2 helpers.* Rejected in favor of full decomposition.

**Default assignments** (`objective_metrics`, `report_frequency`
validation) stay in `optimize()` before any helper call — they are
input normalization, not a distinct step.

**Extracted helpers (all private, in `api.py`):**

| Helper | Lines | Responsibility |
|--------|-------|---------------|
| `_infer_and_validate_keys(adapter) -> tuple[list[str], list[str]]` | 314-333 | Call `infer_keys_from_adapter`, validate results, apply condition-only fallback. Returns `(param_keys, data_keys)`. The raw `inference_conditions` list is not returned — it is only used as a fallback source for `data_keys`. |
| `_setup_validation_data(simulator, validation_simulator, param_keys, data_keys, validation_conditions, sims_per_condition) -> ValidationDataset` | 335-343 | Pick the right simulator, call `generate_validation_dataset` |
| `_build_objective(**config_kwargs) -> GenericObjective` | 358-380 | Construct `ObjectiveConfig` + `GenericObjective` |
| `_derive_directions(objective, directions, objective_metrics, objective_mode, cost_metric) -> tuple[list[str], list[str]]` | 382-400 | Validate/auto-derive `directions`, build `metric_names` list. Returns `(directions, metric_names)` |
| `_create_and_run_study(objective, study_kwargs, n_trials, max_total_trials, show_progress_bar) -> Study` | 402-429 | Delete-if-needed, `create_study`, `optimize_until`. `study_kwargs` is a dict bundling `{study_name, directions, metric_names, storage, resume, warm_start_from, warm_start_top_k}` to keep the parameter count manageable |

After extraction, `optimize()` becomes:

```python
def optimize(...) -> optuna.Study:
    if objective_metrics is None:
        objective_metrics = ["calibration_error", "nrmse"]
    # ... report_frequency validation ...

    # Step 1: Infer keys
    param_keys, data_keys = _infer_and_validate_keys(adapter)
    # Step 2: Validation data
    validation_data = _setup_validation_data(...)
    # Step 3: Pre-flight check
    check_pipeline(...)
    # Step 4: Build objective
    objective = _build_objective(...)
    # Step 5: Derive directions
    directions, metric_names = _derive_directions(objective, ...)
    # Step 6: Run study
    return _create_and_run_study(...)
```

### 2. `_TrackingDict` fix

**Decision:** Override `items()` and `values()` to mark keys as
accessed. Leave `__iter__` untracked.

**Rationale:** `dict(tracking_dict)` calls `__iter__` internally, which
would falsely mark all keys as accessed. `items()` and `values()` are
the common iteration patterns in custom builders (`for k, v in
hparams.items()`). Raw `for k in hparams:` without value access is
rare and arguably a code smell anyway.

**Implementation:**

```python
def items(self):
    self.accessed_keys.update(self.keys())
    return super().items()

def values(self):
    self.accessed_keys.update(self.keys())
    return super().values()
```

**Alternatives considered:**
- *Track everything + explicit copy method.* Rejected: changes the
  dict copy pattern used by `build_continuous_approximator`.
- *Document the limitation.* Rejected: users would get confusing
  warnings with no workaround.

### 3. Registration dedup

**Decision:** Extract `_register_with_aliases()` helper.

**Current duplication:** Both `register_custom_inference_network` and
`register_custom_summary_network` repeat:

```python
if builder is not None:
    register_*_builder(name=name, builder=builder, overwrite=overwrite)
    for alias in aliases or []:
        register_*_builder(name=alias, builder=builder, overwrite=overwrite)
```

**Extracted helper:**

```python
def _register_with_aliases(
    register_fn: Callable[..., None],
    name: str,
    builder: Callable[[dict[str, Any]], Any],
    aliases: list[str] | None,
    overwrite: bool,
) -> None:
    """Register a builder under *name* and each alias."""
    register_fn(name=name, builder=builder, overwrite=overwrite)
    for alias in aliases or []:
        register_fn(name=alias, builder=builder, overwrite=overwrite)
```

Note: `register_fn` is called with keyword arguments to match
the existing `register_inference_builder` / `register_summary_builder`
signatures (`name: str, builder: Callable, overwrite: bool`).

## Scope

### In scope
- Extract 4 private helpers from `optimize()` in `api.py`
- Override `items()` and `values()` on `_TrackingDict` in `pipeline.py`
- Extract `_register_with_aliases()` in `registration.py`
- Tests for each change

### Out of scope
- Public API changes (all helpers are private)
- The 3 decision-needed items from Package D (timing semantics,
  validation_data optionality, validate_fn contract docs)
- Export `default_validate_fn` (already done)

## Constraints

- No public API changes — `optimize()` signature stays identical
- No behavioral changes — all helpers must preserve exact current logic
- Tests must pass with `KERAS_BACKEND=torch`
