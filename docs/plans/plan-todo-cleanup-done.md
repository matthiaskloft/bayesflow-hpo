# Plan: Clean Up Open TODOs

**Created**: 2026-03-15
**Author**: Claude

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-15 | |
| Phase 1: Mark stale TODO done + fix color assertion | IMPLEMENTED | 2026-03-15 | |
| Ship | TODO | | |

## Summary

**Motivation**: The project has two open TODOs in `dev/TODO.md`. On inspection, TODO #1 (unify metric auto-detection) is already resolved — `plot_metric_panels` already calls `_get_metric_user_attrs()` at line 369. TODO #2 (fragile color assertion) is a real issue where `line.get_color()` may return hex strings depending on matplotlib version.

**Outcome**: Both TODOs moved to Done, with the color assertion fixed using `matplotlib.colors.to_hex()`.

## Phase 1: Mark stale TODO done + fix color assertion

**Scope**: Single phase since both changes are trivial and related (TODO cleanup).

### Changes

1. **`tests/test_visualization.py`** (lines 291-294):
   - Add `from matplotlib.colors import to_hex` to imports
   - Replace `line.get_color() in ("grey", "gray")` with `to_hex(line.get_color()) == to_hex("gray")`

2. **`dev/TODO.md`**:
   - Move all completed items to Done section
   - Mark "Unify metric auto-detection" as already resolved (no code change)
   - Mark "Fix fragile iso-line color assertion" as done
