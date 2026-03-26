# Plan: Package G — Search Space Gaps

| Stage | Status |
|-------|--------|
| Plan  | DONE   |
| Phase 1: FusionTransformer dimensions | DONE |
| Phase 2: IntDimension log+step validation | DONE |
| Ship  | TODO   |

## Motivation

Two consistency gaps in the search space module:

1. **FusionTransformerSpace** is missing `mlp_width`, `mlp_depth`, and
   `bidirectional` — all three are accepted by `bf.networks.FusionTransformer`
   and exposed by peer summary spaces (SetTransformer, TimeSeriesTransformer,
   TimeSeriesNetwork).
2. **IntDimension** silently allows `log=True` + `step` (other than 1),
   which Optuna's `trial.suggest_int()` rejects at runtime with a
   `ValueError`. Early validation avoids confusing mid-trial errors.

## Phase 1: FusionTransformerSpace dimensions

### Changes

**`src/bayesflow_hpo/search_spaces/summary/fusion_transformer.py`**

Add three new constant-by-default fields matching BayesFlow defaults:

| Field | Dimension name | Type | Default | BF default |
|-------|---------------|------|---------|------------|
| `mlp_width` | `ft_mlp_width` | `IntDimension` | `constant=128` | `(128, 128)` |
| `mlp_depth` | `ft_mlp_depth` | `IntDimension` | `constant=2` | `(2, 2)` |
| `bidirectional` | `ft_bidirectional` | `CategoricalDimension` | `constant=True` | `True` |

Update `build()` to:
- Read `ft_mlp_width`, `ft_mlp_depth`, `ft_bidirectional` from params
- Pass `mlp_widths=tuple([mlp_width] * num_layers)` and
  `mlp_depths=tuple([mlp_depth] * num_layers)` (same pattern as SetTransformer)
- Pass `bidirectional=bool(params["ft_bidirectional"])`

Update docstring's "Fixed dimensions" section.

**`tests/test_search_spaces/test_phase2_spaces.py`**

- Add constant-injection tests for `ft_mlp_width=128`, `ft_mlp_depth=2`,
  `ft_bidirectional=True`
- Update `FusionTransformerSpace` `.constants` test to include new keys

**`docs/search_spaces.md`**

Add new rows to the FusionTransformerSpace table.

## Phase 2: IntDimension log+step validation

### Changes

**`src/bayesflow_hpo/search_spaces/base.py`**

In `IntDimension.__post_init__`, after existing validation, add:

```python
if self.log and self.step is not None and self.step != 1:
    raise ValueError(
        f"IntDimension({self.name!r}): log=True is incompatible "
        f"with step={self.step}. Optuna requires step=1 (or None) "
        f"for log-scale integer sampling."
    )
```

**Tests** (new file or extend existing):

- `IntDimension("x", low=1, high=100, log=True, step=4)` → `ValueError`
- `IntDimension("x", low=1, high=100, log=True, step=1)` → OK
- `IntDimension("x", low=1, high=100, log=True)` → OK (step=None)
- `IntDimension("x", low=1, high=100, step=4)` → OK (log=False)

## Risks

None — both changes are additive/defensive. No public API signatures change.
New FusionTransformer fields default to `constant`, so existing code is
unaffected.
