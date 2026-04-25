# Plan: FlowMatching Search-Space Expansion

**Created**: 2026-04-25
**Author**: Codex + Matze

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Spec | DONE | 2026-04-25 | Generic FlowMatching solver and TimeMLP kwargs belong in this package |
| Plan | DONE | 2026-04-25 | Expansion with BayesFlow-default constants for untuned dimensions |
<<<<<<< HEAD
| Phase 1: Add Missing BayesFlow-Default Dimensions | DONE | 2026-04-25 | Added solver and TimeMLP dimensions with BayesFlow-default constants; `fm_time_embedding_dim` default set to 32 |
| Phase 2: Add Presets Or Tunable Profiles | DONE | 2026-04-25 | Added `fast`, `balanced`, `quality`, and `preset(...)` profile helpers |
| Phase 3: Documentation And Examples | DONE | 2026-04-25 | Updated docs tables/snippets and README usage notes |
=======
| Phase 1: Add Missing BayesFlow-Default Dimensions | DONE | 2026-04-25 | Added solver and TimeMLP dimensions with BayesFlow-default constants; `fm_time_embedding_dim` default set to 32 |
| Phase 2: Add Presets Or Tunable Profiles | DONE | 2026-04-25 | Added `fast`, `balanced`, `quality`, and `preset(...)` profile helpers |
| Phase 3: Documentation And Examples | DONE | 2026-04-25 | Updated docs tables/snippets and README usage notes |
>>>>>>> 603a782... Expand FlowMatching search space defaults and profiles
| Ship | TODO | | |

## Spec

_Design decisions and requirements - the "what and why". Written directly
from downstream IRT inference-speed use cases._

### Summary

**Motivation**: `FlowMatchingSpace` currently tunes TimeMLP width, depth,
and dropout, but several relevant `bf.networks.FlowMatching` and TimeMLP
kwargs are absent or fixed without documentation. For ODE-based posterior
sampling, integration method and integration steps dominate inference
latency. Downstream packages such as `bayesflow_irt` should not need to
fork generic FlowMatching HPO logic to tune these settings.

**Outcome**: `bayesflow_hpo.FlowMatchingSpace` exposes the relevant
FlowMatching and TimeMLP knobs as dimensions. Any dimension that is not
optimized must default to the corresponding BayesFlow default, so an
untuned HPO dimension does not silently change model behavior. The package
may also provide ergonomic fast/balanced/quality profiles for
speed-sensitive workflows.

### Requirements

- **R1**: Any FlowMatching or TimeMLP dimension that is not optimized must
  be represented as a constant matching the current BayesFlow default.
- **R2**: Add dimensions for FlowMatching integration kwargs:
  `fm_integrate_method` and `fm_integrate_steps`.
- **R3**: Add dimensions for TimeMLP kwargs that materially affect speed,
  memory, or regularization: `fm_merge`, `fm_norm`, `fm_residual`,
  `fm_spectral_normalization`, and optionally `fm_kernel_initializer`.
- **R4**: Keep existing dimension names stable:
  `fm_subnet_width`, `fm_subnet_depth`, `fm_dropout`, `fm_activation`,
  `fm_use_optimal_transport`, `fm_time_power_law_alpha`,
  `fm_time_embedding_dim`.
- **R4a**: Correct any existing fixed/default dimensions that do not match
  BayesFlow defaults unless they are intentionally optimized. In particular,
  local BayesFlow `TimeMLP` defaults `time_embedding_dim=32`, while the
  current `FlowMatchingSpace` uses `constant=8`.
- **R5**: `build()` must pass new TimeMLP dimensions through
  `subnet_kwargs` and new solver dimensions through `integrate_kwargs`.
- **R6**: Tests must assert both default constants and widened/tuned
  dimensions are propagated to `bf.networks.FlowMatching`.
- **R7**: Docs must explain that `fm_integrate_steps` usually has the
  strongest inference-time effect because it multiplies velocity-network
  evaluations during ODE sampling.
- **R8**: If named profiles are implemented, they must be opt-in and must
  not make `FlowMatchingSpace()` silently faster but less accurate.

### Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Ownership | Downstream packages / `bayesflow_hpo` | `bayesflow_hpo` | The kwargs are generic to BayesFlow `FlowMatching`, not IRT-specific. |
| Compatibility | Preserve old constants / Match BayesFlow defaults / Tune everything | Match BayesFlow defaults | Untuned HPO dimensions should be semantically neutral. If users want the legacy `time_embedding_dim=8`, they can pass an explicit constant override. |
| Solver exposure | Only `steps` / `method` and `steps` / Full integrate kwargs | `method` and `steps` first | These are high-impact and fit existing dimension types. Tolerances can be added later if needed. |
| Profiles | Separate class / constructor arg / no profiles | Constructor arg or classmethod | A profile can widen dimensions consistently without adding many new public classes. |
| Null values | Use `None` categorical values / string sentinel | Prefer actual `None` if dimensions support it | `norm=None` is a valid BayesFlow TimeMLP value; tests should verify sampling/build path handles it. |

### Scope

#### In Scope

- Extend `src/bayesflow_hpo/search_spaces/inference/flow_matching.py`.
- Add or update tests in `tests/test_search_spaces/test_flow_matching_space.py`.
- Update docs: `docs/search_spaces.md`, `docs/defaults.md`,
  `docs/api_reference.md` if needed, and `docs/TODO.md` if this closes an
  existing search-space gap.
- Add examples/snippets showing fast-inference override usage.

#### Out of Scope

- Changing BayesFlow `FlowMatching` itself.
- Benchmarking every downstream domain.
- Adding arbitrary nested dict dimensions for all `integrate()` tolerances.
- Changing `CouplingFlowSpace`, `DiffusionModelSpace`, or consistency
  spaces except for cross-doc consistency.

### Architecture Overview

`FlowMatchingSpace` remains a dataclass `BaseSearchSpace` with dimension
fields. `BaseSearchSpace.sample()` emits a flat `dict[str, Any]`; `build()`
maps the flat keys into the nested constructor structure expected by
BayesFlow:

```text
fm_* flat params
  -> subnet_kwargs={widths, dropout, activation, time_embedding_dim,
                    merge, norm, residual, spectral_normalization, ...}
  -> integrate_kwargs={method, steps}
  -> bf.networks.FlowMatching(...)
```

Downstream packages continue to pass this search space through
`CompositeSearchSpace(inference_space=...)`.

### Constraints

- Keras backend must be set before importing BayesFlow in tests that
  instantiate real networks.
- Dimension names use the `fm_` prefix to avoid collisions inside
  `CompositeSearchSpace`.
- `steps="adaptive"` is valid for BayesFlow, but `IntDimension` cannot
  represent string values. The initial plan should either keep adaptive as a
  constant outside the tuned profile or use a categorical dimension if mixed
  fixed/adaptive search is required.
- `method="euler"` is incompatible with adaptive steps in BayesFlow
  integration; tests and profile choices must avoid invalid combinations.

### Open Questions

- Should `fm_integrate_steps` use `CategoricalDimension(constant="adaptive")`
  to represent the BayesFlow default, or should the dimension system gain
  first-class support for mixed string/integer solver-step choices?
- Should named profiles be constructor strings such as
  `FlowMatchingSpace(profile="fast")`, classmethods such as
  `FlowMatchingSpace.fast()`, or separate helper functions?
- Do we want to support conditional search spaces to prevent invalid
  combinations such as `method="euler"` with `steps="adaptive"`?

## Implementation Plan

### Phase 1: Add Missing BayesFlow-Default Dimensions

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/search_spaces/inference/flow_matching.py`
- `tests/test_search_spaces/test_flow_matching_space.py`
- `docs/search_spaces.md`
- `docs/defaults.md`
- `docs/api_reference.md` if it lists FlowMatching dimensions

**Steps:**
1. Add dataclass fields with constants matching local BayesFlow defaults:
   - `integrate_method = CategoricalDimension("fm_integrate_method", constant="tsit5")`
   - `integrate_steps = CategoricalDimension("fm_integrate_steps", constant="adaptive")`
   - `merge = CategoricalDimension("fm_merge", constant="concat")`
   - `norm = CategoricalDimension("fm_norm", constant="layer")`
   - `residual = CategoricalDimension("fm_residual", constant=True)`
   - `spectral_normalization = CategoricalDimension("fm_spectral_normalization", constant=False)`
2. Consider whether `kernel_initializer` should be added now as
   `constant="he_normal"`; if it is exposed as a dimension, the untuned
   constant must match BayesFlow.
3. Change `fm_time_embedding_dim` from `constant=8` to `constant=32` unless
   the implementation deliberately makes it tuned by default. If preserving
   legacy behavior is necessary, document the legacy override:
   `time_embedding_dim=IntDimension("fm_time_embedding_dim", constant=8)`.
4. Include new fields in `dimensions`.
5. Update `build()` to include these keys in `subnet_kwargs` and
   `integrate_kwargs`.
6. Keep old dimension names unchanged, but update constants to BayesFlow
   defaults where they differ.
7. Update tests to verify default constants are sampled and passed through.

**Depends on:** None

### Phase 2: Add Presets Or Tunable Profiles

**Files to create:**
- None, unless profile docs are split into a new page

**Files to modify:**
- `src/bayesflow_hpo/search_spaces/inference/flow_matching.py`
- `tests/test_search_spaces/test_flow_matching_space.py`
- `docs/search_spaces.md`
- `docs/defaults.md`

**Steps:**
1. Choose a profile API after resolving the open question:
   constructor argument, classmethod, or helper function.
2. Define at least three profiles:
   - `default`: mirrors BayesFlow constructor defaults for any dimension that
     is fixed rather than optimized.
   - `fast`: small widths/depths, low/no dropout, fixed-step Euler/Tsit5,
     fewer steps, lean TimeMLP kwargs.
   - `quality`: wider/deeper TimeMLP, adaptive or higher fixed steps.
3. Ensure profiles only configure dimensions; they should not bypass the
   `BaseSearchSpace` validation/sampling path.
4. Add tests for profile defaults and build propagation.
5. Document that users can manually override any profile dimension by
   passing custom `Dimension` objects to the dataclass.

**Depends on:** Phase 1

### Phase 3: Documentation And Examples

**Files to create:**
- Optional example under `examples/` if existing examples are insufficient

**Files to modify:**
- `README.md`
- `docs/search_spaces.md`
- `docs/defaults.md`
- `docs/optimization.md` if it discusses search-space composition
- `docs/plans/plan-flowmatching-search-space-expansion.md`

**Steps:**
1. Add a FlowMatching table covering the new dimensions and defaults.
2. Add a short "speed-sensitive FlowMatching" snippet:

   ```python
   inference_space = hpo.FlowMatchingSpace(
       subnet_width=hpo.IntDimension("fm_subnet_width", 32, 128, step=32),
       subnet_depth=hpo.IntDimension("fm_subnet_depth", 1, 3),
       integrate_method=hpo.CategoricalDimension("fm_integrate_method", ["euler", "tsit5"]),
       integrate_steps=hpo.CategoricalDimension("fm_integrate_steps", [16, 24, 32]),
       merge=hpo.CategoricalDimension("fm_merge", ["add", "concat"]),
       norm=hpo.CategoricalDimension("fm_norm", [None, "layer"]),
   )
   ```

3. Cross-link downstream usage: packages can pass the space as
   `CompositeSearchSpace(inference_space=inference_space, ...)`.
4. Note solver tradeoffs: fewer fixed steps are faster but require
   validation against posterior diagnostics.

**Depends on:** Phase 2, or Phase 1 if profiles are deferred.

## Verification & Validation

- **Automated**:
  - `pytest tests/test_search_spaces/test_flow_matching_space.py -q`
  - `pytest tests/test_search_spaces -q`
  - Run any docs-snippet or API-reference tests if available.
- **Manual**:
  - Monkeypatch `bf.networks.FlowMatching` and verify captured
    `subnet_kwargs` and `integrate_kwargs`.
  - Build a real `FlowMatchingSpace` sample under `KERAS_BACKEND=torch` and
    instantiate the network.
  - For a fast profile, run one small posterior-sampling smoke test in a
    downstream package such as `bayesflow_irt` if available.

## Dependencies

- BayesFlow `bf.networks.FlowMatching` constructor.
- BayesFlow TimeMLP kwargs:
  `widths`, `activation`, `kernel_initializer`, `residual`, `dropout`,
  `spectral_normalization`, `time_embedding_dim`, `merge`, `norm`.
- BayesFlow `integrate()` kwargs:
  `method`, `steps`, and possibly `min_steps`, `max_steps`, `atol`, `rtol`
  for future expansion.

## Notes

- The current local BayesFlow default is
  `integrate_kwargs={"method": "tsit5", "steps": "adaptive"}`.
- The current local TimeMLP default is:
  `widths=(256, 256, 256, 256, 256)`, `activation="mish"`,
  `dropout=0.05`, `time_embedding_dim=32`, `merge="concat"`,
  `norm="layer"`, `residual=True`, `spectral_normalization=False`.
- The existing `FlowMatchingSpace` sets `fm_time_embedding_dim` to
  `constant=8`, which differs from the local BayesFlow TimeMLP default of
  32. This plan intentionally changes the untuned constant to 32 so fixed
  HPO dimensions match BayesFlow defaults. Users who prefer the leaner legacy
  value can pass an explicit `IntDimension("fm_time_embedding_dim",
  constant=8)`.
- `docs/search_spaces.md` currently appears to describe some FlowMatching
  dimensions differently from the implementation. This plan should reconcile
  docs with code as part of Phase 1.

## Review Feedback

- Not independently reviewed by a sub-agent in this pass; the plan was
  cross-checked against
  `src/bayesflow_hpo/search_spaces/inference/flow_matching.py`,
  `tests/test_search_spaces/test_flow_matching_space.py`, and the local
  BayesFlow `FlowMatching` / TimeMLP constructors.
