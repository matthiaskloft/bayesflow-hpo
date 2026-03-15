# bayesflow-hpo — Project TODOs

Tracked items for ongoing development. Updated by contributors and Claude Code sessions.

## Open

### Trial counting docs & reporting
The trial counting mechanism (what counts as a "trial" towards `n_trials` / `optimize_until` cap) is unclear to users. Needs:
- Better inline documentation explaining which trials count (trained vs budget-rejected vs failed)
- Clearer console output showing trial progress (e.g., "Trial 5/20 (3 trained, 2 rejected)")
- Document the distinction between `_non_rejected_now()` and `_total_now()` in user-facing output

## Done

### Quickstart runnable example (2026-03-15)
Fixed `examples/quickstart.ipynb` to run end-to-end from a fresh clone:
- Removed stale kwargs (`param_keys`, `data_keys`, `validation_data`) that no longer exist in `optimize()` API
- Changed `n_trials=0, resume=True` → `n_trials=5, storage=None`
- Updated markdown to reflect that key inference and validation data generation happen inside `optimize()`

### Review CI checks (2026-03-15)
PR #9 (stale revert) was already closed. CI passes on main (lint + test 3.11/3.12/3.13). No action needed.

### Enhance code docs (2026-03-12)
Added/enhanced module-level docstrings on all 42 .py files, all `build()` methods, private helpers, and design-decision comments. All 233 tests pass, ruff clean.
