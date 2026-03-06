# TODOs

Issues identified during the multi-objective pruning quality audit (2026-03-06).
Items marked **(pre-existing)** were not introduced by the pruning PR.

---

## Error Handling

### Broad `except Exception` in `_run_lightweight_validation` (pre-existing)

**File:** `optimization/validation_callback.py:188-194`

The catch block swallows all exceptions and logs at `DEBUG` level. If the validation pipeline has a bug, pruning silently never activates for the entire study with no visible indication. Should log at `WARNING` level and/or track consecutive failures.

### Final validation not wrapped in try-except (pre-existing)

**File:** `optimization/objective.py:299-317`

If the final `run_validation_pipeline` call fails (OOM, numerical error), the trial crashes after a full training run. Could fall back to training loss instead of losing the trial entirely.

### `get_param_count` returns -1 on error (pre-existing)

**File:** `objectives.py:27-34`

When `count_params()` raises `ValueError`, the function returns `-1`, which `normalize_param_count` maps to `0.0` (best possible param score). A broken model silently appears as having 0 parameters on the Pareto front.

### `api.py` delete-study catches only `KeyError` (pre-existing)

**File:** `api.py:201-204`

`optuna.delete_study` can also raise `OperationalError` (locked/corrupted SQLite) or `RuntimeError`. These propagate as unhandled exceptions before any trials run.

---

## Reporting

### `optimize_until` warning message doesn't mention pruning

**File:** `optimization/study.py:324-329`

When `max_total_trials` is exhausted, the warning says "Consider raising max_param_count or tightening the search space" — wrong advice when the cause is heavy pruning. Should differentiate budget rejection from pruning in the guidance.

---

## Pruning Robustness

### `OptunaReportCallback` stores per-epoch user attrs on every trial (pre-existing)

**File:** `optimization/callbacks.py:40`

Stores `epoch_{N}_loss` on every epoch for every trial. With 200 epochs and 100+ trials, this bloats the SQLite database. Consider reducing frequency or making it configurable.

### `MedianPruner` docstring in `create_study` is misleading

**File:** `optimization/study.py:79-82`

The `pruner` docstring doesn't mention that the `MedianPruner` is only used for single-objective studies. Multi-objective studies use the custom median strategy in `PeriodicValidationCallback`.
