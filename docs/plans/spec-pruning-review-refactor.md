# Spec: Pruning Review & Refactor (Package A1)

## Summary

The current multi-objective pruning implementation in bayesflow-hpo uses a
custom median-based strategy with hard-coded metrics and a geometric mean
composite score. A literature review reveals that (1) Optuna has no built-in
multi-objective pruning (Issue #3450, open since April 2022), (2) the
geometric mean composite is scale-sensitive and poorly referenced, and (3)
the startup trial count is misaligned with the sampler's exploration phase.

This package refactors the pruning system into a pluggable architecture with
four strategies backed by literature, fixes the metric alignment bug, and
adds string-based pruner presets for single-objective studies.

## Requirements

### R1: Pluggable multi-objective pruning strategies

The system must support four named strategies selectable via a string parameter
in `optimize()`:

| Strategy | Behavior | Reference |
|----------|----------|-----------|
| `"none"` | Disable intermediate pruning entirely | — |
| `"dominance"` | Per-objective median check; prune if worse than median on ALL objectives (AND rule) | Simplified adaptation of MO-ASHA dominance-based promotion (Schmucker et al., 2021) |
| `"mo-sha"` | Non-dominated sorting (NSGA-II style) at each validation step; prune if trial falls in the bottom 1/η fraction | Schmucker et al. (2021), MO-ASHA Algorithm 1 |
| `"primary"` | Single-metric median pruning on a user-specified objective | MedianPruner design (Akiba et al., 2019) |

**Default:** `"dominance"` for multi-objective studies.

For single-objective studies, Optuna's built-in pruner is always used
regardless of this setting.

### R2: Strategy selection syntax

```python
# Simple strategies
optimize(..., pruning_strategy="dominance")
optimize(..., pruning_strategy="mo-sha")
optimize(..., pruning_strategy="none")

# Primary strategy with metric specification (tuple syntax)
optimize(..., pruning_strategy=("primary", "calibration_error"))

# Primary with default metric (first entry in objective_metrics)
optimize(..., pruning_strategy="primary")
```

The `pruning_strategy` parameter accepts `str | tuple[str, str]`.

### R3: Intermediate metrics auto-align with objective_metrics

Remove the hard-coded `_INTERMEDIATE_METRICS = ["calibration_error", "nrmse"]`.
Intermediate validation always uses the same metrics as `objective_metrics`.
No separate `intermediate_metrics` parameter.

### R4: Auto-detect n_startup_trials from sampler

Default `pruning_n_startup_trials` to the sampler's `n_startup_trials`
attribute (25 for TPE, 10 for BoTorch/GP). Fall back to 10 if the sampler
doesn't expose the attribute. User can still override explicitly via the
existing `pruning_n_startup_trials` on `ObjectiveConfig` (not surfaced on
`optimize()` — power users access it through the config, consistent with
other advanced knobs like `intermediate_validation_interval`).

### R5: Normalize metrics before comparison

For the `"dominance"` strategy, normalize each objective to [0, 1] using
the range of completed trials' scores at the same step before comparing
against the median. This eliminates scale sensitivity — the main failure
mode of the current geometric mean approach. Emmerich & Deutz (2018,
Proposition 9) show that linear scalarization can only find solutions on
convex Pareto fronts; non-convex regions are unreachable. Schmucker
et al. (2021, Section 6, NAS experiments) confirmed empirically that
scalarization "tends to penalize one objective heavier than the other"
while "globally informed techniques are more robust towards objectives
of different magnitude."

### R6: Single-objective pruner string presets in create_study()

Add string-based pruner selection alongside the existing `pruner=` object
parameter:

| Preset | Pruner | Notes |
|--------|--------|-------|
| `"median"` | `MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)` | Current default |
| `"hyperband"` | `HyperbandPruner(min_resource=1, reduction_factor=3)` | Outperforms MedianPruner with TPE (Optuna benchmarks; Li et al., 2018) |
| `"none"` | `NopPruner()` | Disable pruning |

Accept both strings and objects: `pruner="hyperband"` or
`pruner=optuna.pruners.MedianPruner(...)`. The type annotation changes from
`pruner: Any | None` to `pruner: str | optuna.pruners.BasePruner | None`.

Note: R6 applies to `create_study()` only. `optimize()` does not currently
forward a `pruner` parameter to `create_study()` and this spec does not
change that — users who want non-default single-objective pruners call
`create_study()` directly and pass the study to `optimize()` via `resume`.

## Design Decisions

### D1: Four strategies rather than one configurable strategy

**Decision:** Separate named strategies rather than a single parameterized
strategy.

**Alternatives considered:**
- Single strategy with knobs (scalarization method, threshold type, etc.) —
  too many interacting parameters, hard to document
- Strategy protocol/ABC with custom implementations — over-engineered for
  4 strategies; can be added later if needed

**Rationale:** Named strategies are discoverable, documentable, and
reference-backable. Each maps to a distinct algorithmic approach from the
literature.

### D2: Keep custom multi-objective pruning (don't wait for Optuna)

**Decision:** Maintain our own multi-objective pruning implementation.

**Alternatives considered:**
- Wait for Optuna to add multi-objective `trial.report()` — Issue #3450
  has been open since April 2022 with no merged PR
- Use only single-objective pruning — loses a key benefit of cutting
  bad trials early in the default multi-objective setting

**Rationale:** Optuna's multi-objective pruning gap is unlikely to close
soon. Our strategies are well-referenced and provide real value.

### D3: "dominance" as default over "mo-sha"

**Decision:** Default to the simpler `"dominance"` strategy.

**Alternatives considered:**
- Default to `"mo-sha"` — more principled but needs more completed trials
  to produce stable rankings, which can be wasteful in small studies
- Default to `"none"` — safest but loses the pruning benefit

**Rationale:** `"dominance"` is conservative (only prunes trials that are
worse than median on ALL objectives), works well with few completed trials,
and is easy to reason about. Users with larger trial budgets can switch
to `"mo-sha"`.

### D4: Tuple syntax for "primary" strategy metric

**Decision:** `pruning_strategy=("primary", "calibration_error")`.

**Alternatives considered:**
- Encode in string `"primary:calibration_error"` — less Pythonic, needs
  string parsing
- Separate parameter `primary_pruning_metric=` — not obviously tied to
  the "primary" strategy

**Rationale:** Tuple syntax is Python-native, type-checkable, and makes
the association between strategy and metric visually obvious.

### D5: Auto-align intermediate metrics with objective_metrics

**Decision:** Remove independent intermediate_metrics configuration.

**Alternatives considered:**
- Default to objective_metrics but allow override — added complexity for
  a niche use case (expensive metrics mid-training)
- Keep separate — perpetuates the current disconnect

**Rationale:** The hard-coded `["calibration_error", "nrmse"]` is a design
bug. Pruning on different metrics than what the study optimizes can prune
trials that would have been good on the actual objectives. If a user needs
cheaper intermediate metrics, they can use `validate_fn` to control what
gets computed.

### D6: Auto-detect startup trials from sampler

**Decision:** Read `sampler.n_startup_trials` with fallback to 10.

**Alternatives considered:**
- Fixed default of 10 — simpler but doesn't adapt to sampler choice
- Keep at 5 — too low for TPE's 25 startup trials, produces volatile
  median baselines

**Rationale:** Pruning against a baseline of random (non-guided) trials is
unreliable. Waiting until the sampler has completed its exploration phase
ensures pruning decisions are based on a representative sample.

## Scope

### In scope

- Pluggable pruning strategy selection via `pruning_strategy` in `optimize()`
- Four strategies: `"none"`, `"dominance"`, `"mo-sha"`, `"primary"`
- Remove hard-coded `_INTERMEDIATE_METRICS`; use `objective_metrics`
- Normalize metrics in `"dominance"` strategy
- Auto-detect `n_startup_trials` from sampler
- Single-objective pruner string presets in `create_study()`
- Update references.md with MO-ASHA, Hyperband, and related citations

### Out of scope

- Strategy protocol/ABC for user-defined strategies (future work)
- Configurable quantile threshold for `"dominance"` (start with median;
  can parameterize later if needed)
- Changes to `MovingAverageEarlyStopping` (orthogonal mechanism)
- Sampler presets (Package A2, depends on this package)
- QMC warm-up (Package A3, independent)

## Architecture Overview

### Modified files

```
src/bayesflow_hpo/
├── api.py                          # Add pruning_strategy parameter
├── optimization/
│   ├── objective.py                # Pass pruning_strategy to callback; auto-detect startup
│   ├── validation_callback.py      # Major refactor: strategy dispatch, remove hard-coded metrics
│   ├── study.py                    # Add pruner string presets to create_study()
│   └── pruning_strategies.py       # NEW: strategy implementations
└── __init__.py                     # Re-export pruning_strategy type
```

### New module: `optimization/pruning_strategies.py`

Contains the four strategy implementations as plain functions (not classes):

```python
def should_prune_dominance(
    trial: optuna.Trial,
    scores: dict[str, float],   # {metric_name: value}
    step: int,
    n_startup_trials: int,
) -> bool:
    """Per-objective median check with normalization (AND rule)."""

def should_prune_mo_sha(
    trial: optuna.Trial,
    scores: dict[str, float],
    step: int,
    n_startup_trials: int,
    reduction_factor: int = 3,
) -> bool:
    """Non-dominated sorting + bottom-fraction pruning (MO-ASHA)."""

def should_prune_primary(
    trial: optuna.Trial,
    score: float,
    step: int,
    n_startup_trials: int,
) -> bool:
    """Single-metric median pruning."""
```

### PeriodicValidationCallback changes

- Constructor receives `pruning_strategy: str | tuple[str, str]` and
  `objective_metrics: list[str]` instead of hard-coded metrics
- `_run_lightweight_validation()` computes all `objective_metrics`
  (not just calibration_error + nrmse)
- `on_epoch_end()` dispatches to the appropriate strategy function
- For `"none"`: do not create the `PeriodicValidationCallback` at all
  (skip both intermediate validation and pruning). Users who want
  intermediate metrics without pruning can use `OptunaReportCallback`'s
  `report_frequency` for training loss tracking.

### Data flow

```
optimize()
  → pruning_strategy="dominance" (default)
  → ObjectiveConfig.pruning_strategy = "dominance"
  → ObjectiveConfig.pruning_n_startup_trials = auto from sampler (25 for TPE)
  → GenericObjective.__call__()
    → PeriodicValidationCallback(
        pruning_strategy="dominance",
        objective_metrics=["calibration_error", "nrmse"],
        n_startup_trials=25,
      )
    → on_epoch_end()
      → _run_lightweight_validation() → {cal_err: 0.12, nrmse: 0.85}
      → store per-metric user attrs: val_cal_err_step_3=0.12, val_nrmse_step_3=0.85
      → should_prune_dominance(trial, scores, step=3, n_startup=25)
        → gather completed trial scores at step 3
        → normalize each metric to [0,1] using observed range
        → check: is trial worse than median on ALL metrics?
        → return True/False
```

### User attribute schema migration

The current implementation stores a single composite scalar per step:
`val_score_step_{N}`. The new implementation stores per-metric values:
`val_{metric_name}_step_{N}` (e.g., `val_calibration_error_step_3`).

**Migration concern:** Resumed/warm-started studies mixing old and new
trials will have different attribute schemas. Strategy functions must
handle missing per-metric attributes gracefully — if a completed trial
has only the old `val_score_step_*` format, skip it when gathering
reference scores (it won't contribute to the pruning baseline, but
this is conservative and correct). Document this in the migration notes.

### Single-objective detection

Single-objective is determined by `len(study.directions) == 1`.
Studies using `objective_mode="mean"` produce 2 directions (mean + cost)
and are treated as multi-objective for pruning purposes. This is
intentional: `"mean"` mode still optimizes two competing objectives.

### validate_fn contract change

When `validate_fn` is provided, it must return all metrics listed in
`objective_metrics` (not just `calibration_error` and `nrmse` as
before). Document this in the `validate_fn` parameter docstring and
raise a clear error if required metrics are missing from the returned
dict.

## Constraints

- **Backwards compatibility:** `pruning_strategy` defaults to `"dominance"`,
  which is behaviorally similar to the current approach (both use median
  comparison). The main difference is metric normalization and alignment
  with `objective_metrics`.
- **Performance:** `"mo-sha"` requires non-dominated sorting which is
  O(M * N^2) where M = number of objectives and N = number of reference
  trials. With typical HPO budgets (< 200 trials), this is negligible.
- **Dependencies:** No new dependencies. Non-dominated sorting can be
  implemented in ~30 lines with numpy (no need for pymoo or similar).

## Resolved Questions

1. **`"none"` skips intermediate validation entirely.** Do not create the
   `PeriodicValidationCallback` at all. Users who want intermediate metrics
   without pruning can use `OptunaReportCallback`'s `report_frequency`.

2. **`"mo-sha"` hard-codes η=3.** Standard in Hyperband/SHA literature
   (Li et al., 2018, Section 3.6: "in practice we suggest taking η to be
   equal to 3 or 4"; theoretical optimum η=e≈2.718). Add configurability
   later if users request it.

## Open Questions

1. **Pruned trial counting with aggressive strategies.** `"mo-sha"` may
   prune more aggressively than the current approach. The default
   `max_total_trials = 3 * n_trials` may need adjustment. Monitor this
   during testing and consider documenting strategy-specific guidance.

## References

All references below have been read in full text (or verified from source
code) per the project's source-backed implementation mandate.

- Schmucker, R., Donini, M., Zafar, M. B., Salinas, D., & Archambeau, C. (2021). Multi-objective asynchronous successive halving. *arXiv preprint*. https://doi.org/10.48550/arxiv.2106.12639
  *Read: full text (19 pages). Algorithms 1–2, experimental results (Sections 5–6).*

- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research*, *18*(185), 1–52.
  *Read: full text (all 52 pages). Algorithm 1, Sections 3.1–3.6 (SHA, η, R), Section 5 (theoretical guarantees), Section 6 (extensions incl. quasi-random sampling suggestion for Package A3).*

- Emmerich, M. T. M., & Deutz, A. H. (2018). A tutorial on multiobjective optimization: Fundamentals and evolutionary methods. *Natural Computing*, *17*(3), 585–609. https://doi.org/10.1007/s11047-018-9685-y
  *Read: full text (25 pages). Definition 5 (Pareto dominance), Equations 3–4 (non-dominated sorting), Proposition 7 (Θ(n²) complexity), Proposition 9 (scalarization limited to convex Pareto fronts).*

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, *6*(2), 182–197. https://doi.org/10.1109/4235.996017
  *Already in references.md. Non-dominated sorting confirmed via MO-ASHA Algorithm 1 lines 1–6.*

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631). https://doi.org/10.1145/3292500.3330701
  *Already in references.md. MedianPruner verified from Optuna source code.*
