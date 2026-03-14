# TODOs

Issues identified during the multi-objective pruning quality audit (2026-03-06).
Items marked **(pre-existing)** were not introduced by the pruning PR.

---

## Error Handling

### Broad `except Exception` in `_run_lightweight_validation` (pre-existing) — PARTIALLY FIXED

**File:** `optimization/validation_callback.py:186-215`

**Original issue:** Catch block swallowed all exceptions at `DEBUG` level.

**Status:** Now logs at `WARNING` level (line 210) and tracks consecutive failures with a warning after 3 (lines 157-161). Still returns `None` on failure rather than raising, but this is intentional — intermediate validation failures should not crash the trial.

### ~~Final validation not wrapped in try-except (pre-existing)~~ — RESOLVED

**File:** `optimization/objective.py` (step 8 in `__call__`)

Fixed: When final validation fails (OOM, numerical error), the trial now falls back to training-loss-based objective values instead of returning worst-case penalty values. The best moving-average training loss (from `MovingAverageEarlyStopping`) is clamped to [0, 1] and used as a proxy for each metric objective, paired with the real param-count-based cost score. If no training loss is available, falls back to full penalty values. The trial is flagged with `validation_fallback=training_loss` in user attrs.

### ~~`get_param_count` returns -1 on error~~ — RESOLVED

**File:** `objectives.py:43-61`

Fixed: `get_param_count` now raises `ValueError` (model not built) or `TypeError` (unsupported type) instead of returning `-1`. Callers handle exceptions appropriately.

### ~~`api.py` delete-study catches only `KeyError`~~ — RESOLVED

**File:** `api.py:320-329`

Fixed: Now catches `KeyError` (study not found) and generic `Exception` with a `logger.warning` and `exc_info=True`, so corrupted/locked storage doesn't crash before any trials run.

---

## Reporting

### ~~`optimize_until` warning message doesn't mention pruning~~ — RESOLVED

**File:** `optimization/study.py:370-421`

Fixed: Warning now includes a failure breakdown with pruned count, per-reason counts, and a dominant-failure-reason warning. Guidance mentions `max_total_trials`, `max_param_count`, and adjusting the search space.

---

## Pruning Robustness

### `OptunaReportCallback` stores per-epoch user attrs on every trial (pre-existing) — OPEN

**File:** `optimization/callbacks.py`

Stores `epoch_{N}_loss` for every trial at `report_frequency` intervals. With 200 epochs and 100+ trials, this bloats the SQLite database. The `report_frequency` parameter exists but defaults to `10` and is not configurable from `optimize()`.

### `MedianPruner` docstring in `create_study` is misleading — OPEN

**File:** `optimization/study.py`

The `pruner` docstring doesn't mention that `MedianPruner` is only used for single-objective studies. Multi-objective studies use the custom median strategy in `PeriodicValidationCallback`.

---

## ~~Extensibility~~ — RESOLVED

### ~~`optimize()` assumes `BasicWorkflow`~~ — RESOLVED

Resolved by the custom approximator hooks implementation (2026-03-14). `optimize()` now accepts `build_approximator_fn`, `train_fn`, and `validate_fn` hooks. The trial lifecycle operates on raw `Approximator` objects, not `BasicWorkflow`.

### ~~`run_validation_pipeline` assumes flat posterior shape~~ — RESOLVED

Resolved by the `validate_fn` hook. Users with non-standard posterior shapes (e.g. per-item `(n_sims, n_samples, I)`) can supply a custom `validate_fn` that handles their format.
