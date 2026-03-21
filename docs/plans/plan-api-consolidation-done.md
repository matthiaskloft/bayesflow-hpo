# Plan: Package C — API Consolidation & Search Space Simplification

**Created**: 2026-03-21
**Author**: Claude
**Spec**: [`docs/spec-api-consolidation.md`](../spec-api-consolidation.md)

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-21 | |
| Phase 1: Dimension `constant` field | MERGED | 2026-03-21 | PR #48 |
| Phase 2: Search space migration | MERGED | 2026-03-21 | PR #48 |
| Phase 3: `num_batches` rename + dimension name expansion | MERGED | 2026-03-21 | PR #48 |
| Ship | MERGED | 2026-03-21 | PR #48 |

## Summary

**Motivation**: bayesflow-hpo's `default_train_fn` passes `batches_per_epoch`
to `approximator.fit()`, but BayesFlow 2.0.8 requires `num_batches` in
`build_dataset()` — the current name is silently dropped, making the default
training path broken. Additionally, the search space system has two overlapping
optionality mechanisms (`enabled` + `include_optional`) that should be replaced
with a simpler `constant` field on dimensions.

**Outcome**: The public API uses BayesFlow-aligned names (`num_batches`),
`default_train_fn` works on BF 2.0.8+ out of the box, Optuna dimension names
match BayesFlow constructor kwargs, and search spaces use a single `constant`
mechanism instead of the `enabled`/`include_optional` two-level system.

## Assumptions

- BayesFlow 2.0.8 is the minimum supported version (uses `num_batches` in
  `build_dataset()`)
- No external users depend on the current API (pre-1.0 package)
- All 29 currently-optional dimensions (`enabled=False`) have a BayesFlow
  default value that can be used as the `constant`. Defaults must be
  verified against BayesFlow source before coding Phase 2 (see pre-work).
- `st_mlp_width` has a parametric default (`2 * embed_dim`) — needs
  special handling (computed in `build()`, not a scalar constant)
- `st_num_inducing` defaults to `None` in BayesFlow — requires a sentinel
  to distinguish "no constant set" from "constant is explicitly None"

## Design Decisions

All design decisions were resolved in the spec. Summary:

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Migration strategy for `batches_per_epoch` | Clean break vs. deprecation shim | Clean break | Pre-1.0, no external users |
| Dimension name prefixes | Keep / drop / expand post-prefix | Keep prefixes, expand abbreviations | Prefixes prevent silent overwrites in `dict.update()` merge |
| Optionality mechanism | `enabled`/`include_optional` vs. `constant` field | `constant` field | Single mechanism, no two-level system |
| Constant sampling | Via Optuna `suggest_*` vs. direct injection | Direct injection | More efficient, checkpoints are source of truth |
| `constant` vs `low/high` | Mutually exclusive vs. precedence | Mutually exclusive (`ValueError`) | Forces explicit intent |
| Sentinel for absent constant | `None` vs. `_UNSET` sentinel | `_UNSET = object()` sentinel | `None` must be a valid constant value (e.g., `st_num_inducing` defaults to `None` in BayesFlow). Use `_UNSET` internally to mean "no constant set". |
| Parametric defaults | Scalar constant vs. computed in `build()` | Keep as range dim, compute in `build()` | `st_mlp_width` defaults to `2 * embed_dim` — cannot be a scalar `constant`. Keep it as a standard range dim and compute the default in `build()` when the value equals a sentinel. |

## Scope

### In Scope
- `constant` field on `IntDimension`, `FloatDimension`, `CategoricalDimension`
- Remove `enabled`, `include_optional`, `defaults()`, skip logic
- `.constants` property on `BaseSearchSpace` and `CompositeSearchSpace`
- Simplify all `build()` methods (remove conditional param checks)
- `batches_per_epoch` → `num_batches` rename across all files
- 6 Optuna dimension name expansions
- Fix `default_train_fn` for BF 2.0.8+
- Update all tests, docs, CLAUDE.md

### Out of Scope
- Package I items (#3, #4, #5, #18) — tracked in TODO.md
- Sampler/pruner presets (Package A)
- `optimize()` refactor (Package D)

## Implementation Plan

### Phase 1: Dimension `constant` field

Add the `constant` field to the dimension infrastructure and update
`BaseSearchSpace` sampling/validation logic. This phase changes the
foundation without touching any concrete search spaces yet.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/search_spaces/base.py` — Add `constant` field to
  `IntDimension`, `FloatDimension`, `CategoricalDimension` with
  `__post_init__` validation (mutually exclusive with `low`/`high`/`choices`).
  Update `BaseSearchSpace.sample()` to inject constants directly. Add
  `.constants` property. Remove `enabled` field from all dimension types.
  Remove `include_optional` field and skip logic from `BaseSearchSpace`.
  Update `_validate()` to require all dimensions (no `enabled` filter).
- `src/bayesflow_hpo/search_spaces/composite.py` — Add `.constants`
  property that merges sub-space constants. Remove `defaults()` merge
  logic from `sample()`.
- `src/bayesflow_hpo/search_spaces/training.py` — Remove `include_optional`
  field and `defaults()` method. Convert `batch_size` from `enabled=False`
  to `constant=256`.
- `tests/test_search_spaces/test_composite_space.py` — Update tests for
  new constant behavior, remove `include_optional` tests, add `.constants`
  property tests.

**Steps:**
1. Add `_UNSET = object()` sentinel in `base.py`. Add `constant` field
   using `field(default=_UNSET)` (bare `= _UNSET` is invalid for
   dataclasses) to `IntDimension`, `FloatDimension`,
   `CategoricalDimension` with `__post_init__` validation: raise
   `ValueError` if both `constant is not _UNSET` and `low`/`high`/`choices`
   are set. `constant=None` is valid (used by `st_num_inducing`).
   **Note:** `_UNSET` is never present in `params` passed to `build()` —
   it is resolved to the actual value during `sample()`.
2. Remove `enabled` field from all three dimension types
3. Update `BaseSearchSpace.sample()`: inject constants directly (check
   `dim.constant is not _UNSET`), remove `include_optional` skip logic
4. Update `BaseSearchSpace._validate()`: require all dimensions
5. Add `.constants` property to `BaseSearchSpace`:
   `{d.name: d.constant for d in self.dimensions if d.constant is not _UNSET}`
6. Remove `include_optional` from `BaseSearchSpace`. **Note:** this
   simultaneously breaks the constructor of ALL 10 concrete spaces (not
   just the 6 that re-declare it) since they inherit from `BaseSearchSpace`.
   This is intentional (pre-1.0 clean break).
7. Update `CompositeSearchSpace`: add `.constants` (merging sub-spaces),
   remove `defaults()` merge logic. **Must be atomic with step 8** —
   `CompositeSearchSpace.sample()` calls `self.training_space.defaults()`
   which is removed in step 8.
8. Update `TrainingSpace`: remove `include_optional`, `defaults()`, convert
   `batch_size` to `constant=256`
9. Update composite/training tests
10. Run `ruff check src/ tests/` and `pytest tests/` to verify

**Depends on:** None

### Phase 2: Search space migration

Migrate all 10 concrete search spaces (5 inference + 5 summary) from
`enabled=False` to `constant=<BF default>`, expand the 6 abbreviated
dimension names, and simplify all `build()` methods.

**Pre-work (before coding):** Read BayesFlow 2.0.8 source to verify the
default value for every `enabled=False` dimension. Per CLAUDE.md's
source-backed implementation rule, constants must be verified against
actual BayesFlow constructors. Update `docs/defaults.md` with any
missing defaults. The dimensions needing BF source verification:

- `dm_noise_schedule`, `dm_prediction_type` (DiffusionModel)
- `cm_max_time`, `cm_sigma2`, `cm_s0`, `cm_s1` (ConsistencyModel)
- `scm_sigma` (StableConsistencyModel)
- `tsn_recurrent_type`, `tsn_bidirectional`, `tsn_skip_steps` (TimeSeriesNetwork)
- `tst_mlp_width`, `tst_time_embed` (TimeSeriesTransformer)
- `ft_template_type` (FusionTransformer)
- `fm_time_embedding_dim` (FlowMatching — missing from spec dim rename
  table but is an `enabled=False` dim that needs migration)
- `st_mlp_depth` (SetTransformer — must also be verified)

**Special cases:**
- `st_mlp_width`: BayesFlow default is `2 * embed_dim` (parametric). Keep
  as a standard range dim, not `constant`. In `build()`, if the sampled
  value is needed as a default, compute from `st_embed_dim`.
- `st_num_inducing`: BayesFlow default is `None`. Use `constant=None`
  (valid with the `_UNSET` sentinel design from Phase 1).
- `st_mlp_depth`: Verify BF default. Also referenced in
  `optimization/constraints.py` (line 99) via `params.get("st_mlp_depth", 2)`
  — this file must be updated since the key will now always be present
  in params (injected as constant).

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/search_spaces/inference/coupling_flow.py` — Convert
  4 `enabled=False` dims to constants, rename `cf_actnorm` →
  `cf_use_actnorm`, simplify `build()` to unconditionally read all params
- `src/bayesflow_hpo/search_spaces/inference/flow_matching.py` — Convert
  4 dims (including `fm_time_embedding_dim`), rename `fm_use_ot` →
  `fm_use_optimal_transport`, `fm_time_alpha` → `fm_time_power_law_alpha`,
  simplify `build()`
- `src/bayesflow_hpo/search_spaces/inference/diffusion.py` — Convert
  2 dims, simplify `build()`
- `src/bayesflow_hpo/search_spaces/inference/consistency.py` — Convert
  4 dims, simplify `build()`, remove `include_optional`
- `src/bayesflow_hpo/search_spaces/inference/stable_consistency.py` —
  Convert 1 dim, simplify `build()`, remove `include_optional`
- `src/bayesflow_hpo/search_spaces/summary/deep_set.py` — Convert 4 dims,
  rename `ds_spectral_norm` → `ds_spectral_normalization`, simplify
  `build()`
- `src/bayesflow_hpo/search_spaces/summary/set_transformer.py` — Convert
  2 dims (keep `st_mlp_width` as range), rename `st_num_inducing` →
  `st_num_inducing_points` (constant=None), simplify `build()`
- `src/bayesflow_hpo/search_spaces/summary/fusion_transformer.py` —
  Convert 1 dim, simplify `build()`, remove `include_optional`
- `src/bayesflow_hpo/search_spaces/summary/time_series_network.py` —
  Convert 3 dims, simplify `build()`
- `src/bayesflow_hpo/search_spaces/summary/time_series_transformer.py` —
  Convert 2 dims, rename `tst_time_embed` → `tst_time_embedding`,
  simplify `build()`
- `tests/test_search_spaces/test_phase2_spaces.py` — **Rewrite** test
  strategy: replace `test_default_sampling_skips_optional` /
  `test_optional_sampling_includes_optional` parametrized groups with
  tests that verify constants appear in params with correct values.
  Remove all `include_optional=True/False` constructor calls. Update
  all dimension name references.
- `tests/test_search_spaces/test_coupling_flow_space.py` — Update dim
  name references, remove `include_optional=True` constructor calls
- `tests/test_search_spaces/test_flow_matching_space.py` — Replace
  `test_default_sampling_skips_optional_dimensions` (line 9) and
  `test_optional_sampling_includes_optional_dimensions` (line 23) with
  constant-value tests. Remove `assert dim.enabled is False` (line 145).
- `tests/test_optimization/test_sampling.py` — Update `include_optional`
  references
- `src/bayesflow_hpo/optimization/constraints.py` — Update
  `params.get("st_mlp_depth", 2)` (constant is now always present)
- `docs/api_reference.md` — Remove `include_optional` from all
  constructor examples, replace "Optional dimensions" with "Constants"
- `docs/search_spaces.md` — Remove `include_optional` references
- `docs/defaults.md` — Update with verified BF defaults from pre-work
- `docs/architecture.md` — Update optionality description (currently
  says `include_optional=True` pattern)
- `.claude/agents/search-space-reviewer.md` — Update pattern descriptions
  to reflect `constant` instead of `enabled`/`include_optional`

**Steps:**
1. Pre-work: read BF source for all undocumented defaults, update
   `docs/defaults.md`
2. For each of the 5 inference spaces: convert `enabled=False` dims to
   `constant=<BF default>`, apply any dimension renames, remove
   `include_optional`, simplify `build()` to unconditionally read params
3. For each of the 5 summary spaces: same conversion, with special
   handling for `st_mlp_width` (keep as range) and `st_num_inducing`
   (constant=None)
4. Rewrite search space tests: replace optional-skip test strategy with
   constant-value verification. Remove all `enabled`/`include_optional`
   assertions.
5. Update search-space-reviewer agent config
6. Run `ruff check src/ tests/` and `pytest tests/` to verify

**Depends on:** Phase 1 (dimension `constant` infrastructure)

### Phase 3: `num_batches` rename + dimension name expansion in non-search-space files

Rename `batches_per_epoch` → `num_batches` across the API, objective,
pipeline, builders, docs, and remaining tests. Update docs and CLAUDE.md.

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_hpo/api.py` — Rename parameter in `optimize()` signature
  and all internal references
- `src/bayesflow_hpo/optimization/objective.py` — Rename in
  `ObjectiveConfig`, `default_train_fn()`, LR decay computation
- `src/bayesflow_hpo/pipeline.py` — Rename in `check_pipeline()`
- `src/bayesflow_hpo/builders/workflow.py` — Update docstrings
- `tests/test_optimization/test_objective.py` — Update test fixtures
- `tests/test_search_spaces/test_phase2_spaces.py` — Update any remaining
  `batches_per_epoch` references
- `docs/optimization.md` — Update parameter docs and examples
- `docs/search_spaces.md` — Update dimension name references and examples
- `docs/defaults.md` — Update if it references old names
- `docs/api_reference.md` — Update parameter references
- `docs/architecture.md` — Update if it references old names
- `CLAUDE.md` — Remove BF 2.0.8 compatibility gotcha, update any
  `batches_per_epoch` references
- `examples/getting_started.ipynb` — Update example code
- `examples/two_moons_optimization.ipynb` — Update example code
- `examples/custom_summary_network.ipynb` — Update example code

**Steps:**
1. Rename `batches_per_epoch` → `num_batches` in `api.py` (signature +
   body)
2. Rename in `ObjectiveConfig` dataclass and `default_train_fn()` in
   `optimization/objective.py`. **Two edits on line 90:** both the
   hparams key lookup (`hparams["batches_per_epoch"]` → `hparams["num_batches"]`)
   AND the kwarg passed to `.fit()` (`batches_per_epoch=` → `num_batches=`)
3. Rename in LR decay computation (`objective.py`, `decay_steps` calc)
4. Rename in `pipeline.py` (`check_pipeline()`)
5. Update `builders/workflow.py` docstrings
6. Update `consistency.py` step calculation: `params.get("batches_per_epoch", 50)`
   → `params.get("num_batches", 50)`
7. Update all test files (24 occurrences in `test_objective.py`,
   2 in `test_phase2_spaces.py` lines 139/286)
8. Update docs: `optimization.md`, `search_spaces.md`, `defaults.md`,
   `api_reference.md`, `architecture.md`
9. Update CLAUDE.md: remove BF 2.0.8 compatibility gotcha
10. Update example notebooks
11. Run `ruff check src/ tests/` and `pytest tests/` to verify

**Depends on:** None (can run in parallel with Phase 2; the
`consistency.py` fallback reads runtime params, not search space fields).
**Note:** `test_phase2_spaces.py` lines 139/286 (`batches_per_epoch` in
test params) should be absorbed into Phase 2's rewrite of that file
rather than applied separately in Phase 3. If Phase 3 runs first, the
edits will be overwritten by Phase 2's rewrite.
**Note:** The `two_moons_optimization.ipynb` update is a **pattern removal**
(the compatibility `train_fn` workaround becomes unnecessary), not just
a key rename.

## Verification & Validation

- **Automated**: Full test suite (`pytest tests/ -v`) must pass after each
  phase. Ruff lint (`ruff check src/ tests/`) must pass.
- **Manual**:
  - After Phase 1: Verify `FloatDimension("x", constant=0.5)` injects
    without Optuna; verify `FloatDimension("x", low=0.1, high=0.5, constant=0.3)`
    raises `ValueError`
  - After Phase 2: Verify `CouplingFlowSpace().constants` returns the
    expected fixed values; verify `CouplingFlowSpace(use_actnorm=CategoricalDimension("cf_use_actnorm", choices=[True, False])).constants` excludes `cf_use_actnorm`
  - After Phase 3: Verify `default_train_fn` passes `num_batches` to
    `approximator.fit()` (check the call matches BF 2.0.8 `build_dataset`
    signature)
- **Search space reviewer agent**: Run the `search-space-reviewer` agent
  after Phase 2 to verify pattern compliance

## Dependencies

- BayesFlow >= 2.0.8 (for `num_batches` in `build_dataset()`)
- No new external dependencies

## Notes

_Living section — updated during implementation._

## Review Feedback

Reviewed in 2 iterations. Iteration 1: 14 findings (3 blockers, 9 warnings,
2 suggestions). Iteration 2: 9 findings (2 blockers, 5 warnings, 2 suggestions).

**Blockers addressed:**
1. BF defaults for 15+ dimensions unverified → Added pre-work step to
   Phase 2 requiring BF source verification before coding.
2. Removing `include_optional` from `BaseSearchSpace` breaks all 10 space
   constructors → Added explicit note in Phase 1 step 6.
3. `CompositeSearchSpace.sample()` and `TrainingSpace.defaults()` removal
   are co-dependent → Added atomicity note in Phase 1 steps 7-8.

**Warnings addressed:**
- Actual `enabled=False` count is 29, not 26 (added `fm_time_embedding_dim`)
- `st_mlp_width` parametric default → Added to Design Decisions + Phase 2
  special cases (keep as range dim)
- `st_num_inducing` default `None` → Added `_UNSET` sentinel to Design
  Decisions + Phase 1
- `test_phase2_spaces.py` needs strategy rewrite → Updated Phase 2 to say
  "Rewrite" not "Update"
- `test_flow_matching_space.py:145` `dim.enabled` assertion → Added to
  Phase 2 file list
- Phase 3 dependency on Phase 2 was incorrect → Fixed to "None"
- `default_train_fn` two-part fix → Explicit in Phase 3 step 2

**Suggestions noted:**
- `_TrackingDict` confirmed no changes needed
- `.claude/agents/search-space-reviewer.md` added to Phase 2 file list

**Iteration 2 blockers addressed:**
4. `_UNSET` sentinel requires `field(default=_UNSET)` for dataclasses →
   Made explicit in Phase 1 Step 1
5. `st_mlp_depth` missing from pre-work list + `constraints.py` not in
   file list → Added both

**Iteration 2 warnings addressed:**
- `test_flow_matching_space.py` needs full function replacement (lines 9-29)
  → Updated Phase 2 file description
- Phase 2 missing doc-update step → Added `docs/api_reference.md`,
  `docs/search_spaces.md`, `docs/defaults.md`, `docs/architecture.md`
- `test_phase2_spaces.py` shared between Phase 2 and 3 → Added note
  that Phase 3 edits should be absorbed into Phase 2's rewrite
- `two_moons_optimization.ipynb` is a pattern removal → Added explicit note
