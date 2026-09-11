# Plan: End-to-End `optimize()` Tests

**Created**: 2026-09-11
**Revised**: 2026-09-11 (v2, after adversarial review — see "Review history")
**Author**: Claude Code (Claude Opus 5)
**Issue**: [#76](https://github.com/matthiaskloft/bayesflow-hpo/issues/76)

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan v1 | SUPERSEDED | 2026-09-11 | Phase 3 was unworkable; smoke test was not load-bearing |
| Plan v2 | DONE | 2026-09-11 | |
| Phase 1: isolated fixtures + observed training | DONE | 2026-09-11 | 3 tests |
| Phase 2: metric-path integration regressions | DONE | 2026-09-11 | 11 tests |
| Phase 3: hook + selection tests | DONE | 2026-09-11 | 5 tests |
| Phase 4: CI measurement, docs, close-out | DONE | 2026-09-11 | measured 210s; see below |
| Mutation verification | DONE | 2026-09-11 | 6/6 regressions caught |

## Summary

**Motivation**: nothing in `tests/` runs a real `optimize()`. `test_api.py`
patches out `GenericObjective`, `create_study`, `optimize_until`,
`check_pipeline` and `generate_validation_dataset` — it asserts argument
forwarding. `test_optimization/test_direction_end_to_end.py` drives real
Optuna but hand-feeds synthetic metric values: no approximator is built and no
validation pipeline runs. The seam
`optimize()` → `check_pipeline()` → build → train → `run_validation_pipeline()`
→ objective extraction is exercised by no test, and #72 found two defects that
lived exactly there.

**Outcome**: a `tests/test_end_to_end/` suite that runs real `optimize()`
studies against a tiny simulator, real approximators and the real validation
pipeline, asserting structure and differential properties rather than values.

**What this suite does not cover** (stated up front, because v1 implied
otherwise): at 2 epochs no trial reaches the intermediate-validation warmup
(`optimization/validation_callback.py:236` returns early while
`epoch < self.warmup`), so **pruning, intermediate validation and open-ended
stopping are not protected by this suite.** Nor is persistence/resume, since
studies run in memory.

## Measurements taken before writing this plan

Run on this branch (`main` @ 60055c7), local Windows, CPU, torch 2.x /
bayesflow 2.0.8 / keras 3.12.1:

| Measurement | Result |
|---|---|
| Real 2-trial `optimize()`, tiny nets, 2 epochs × 4 batches | **17.9 s** |
| Same with `objective_metrics=["log_gamma"]` + a multi-output constraint | **14.6 s** |
| Existing suite, `pytest tests/ -q` | 891 passed, **106 s** |

### Measured after implementation (Phase 4)

The pre-implementation estimate of “~90–120 s, under four minutes total” was
**wrong**, and this is the corrected number. Measured under CI's own command
(`pytest tests/ --cov=bayesflow_hpo`), locally on Windows:

| Run | Result |
|---|---|
| `-m "not endtoend"` (baseline) | 891 passed, 19 deselected, **54 s** |
| Full suite | 910 passed, **290 s** |
| `tests/test_end_to_end/` alone | 19 passed, **210 s** |

So the suite costs **~210 s**, roughly 2× the original estimate. The dominant
cost is not training (~0.7 s a trial) but `check_pipeline()`, which trains and
validates an entire extra approximator before every study — pre-flight roughly
doubles the cost of each of the ~10 studies.

Reducing trial counts where no assertion was cross-trial (a review finding)
took it from ~236 s to 210 s. Cutting further would mean shrinking posterior
draws to the point where metric finiteness stops being reliable, which trades
the suite's determinism for seconds.

**Decision: keep it in CI on all three Python versions.** The matrix runs in
parallel, so the wall-clock cost is +3.5 min per job, and version drift in
BayesFlow/Keras is exactly what this suite exists to notice. The `endtoend`
marker is the documented lever if that judgement changes
(`pytest -m "not endtoend"`), and it is verified to work — 19 deselected.

Caveat retained: CI runs Ubuntu only (`.github/workflows/ci.yml:38`); these are
Windows timings.

**Re-measured on optuna 5.0.0.** These numbers were originally taken on 4.9.0
while CI, under the old `optuna>=4.0.0`, would have installed 5.0.0 — so they
were not measurements of what CI runs. The requirement is now
`>=5.0.0,<6.0.0` with the exact CI version pinned in
`.github/ci-constraints.txt`; reproduce that environment with
`pip install -e ".[dev]" -c .github/ci-constraints.txt`. On 5.0.0: full suite
919 passed in 169 s without coverage, no behavioural difference in this suite.

Record torch/bayesflow/keras versions alongside any re-measurement: torch is
unpinned and bayesflow carries only a lower bound (`pyproject.toml:23`).

## Findings from the probes

I ran the two repairs #76 lists under its item 2 rather than reading them:

- `objective_metrics=["log_gamma"]` completes end-to-end; the stored user attr
  is `-3.454782` and the reported objective `3.45478` — the raw
  higher-is-better value and its minimize-form, consistent.
- A hard constraint on `left_coverage_90` pulls `coverage_left` into the
  pipeline via `producer_for_key()`, and `left_coverage_90/95/98/99` plus
  `left_mean_cal_error` appear in the trial's user attrs.

Both were fixed in #78/#79, after #76 was filed. So Phase 2 is regression
protection for repairs already made, not bug discovery.

**Correction to v1**: v1 claimed "the fixes are currently held in place by
nothing." That is false. `tests/test_optimization/test_metric_name_inventory.py`
covers non-default validator threading (`:219`), constraint metric inclusion
(`:254`), alias direction registration (`:290`) and output-key producers
(`:322`). What is missing is *integration* coverage — those tests assert the
helper functions, not that a real run reaches them.

One thing I checked and found *not* to be a bug: a hard constraint
`("left_coverage_90", 0.99, "below")` did not reject a trial reporting `1.0`.
`_check_hard_constraints` defines `"below"` as *violated when the value is
below the threshold* (`optimization/objective.py:1022`). Correct behaviour;
my first reading was not.

## Design decisions

### 1. `COMPLETE` does not mean the run worked

This is the decision the whole plan turns on. A trial that raises during
training returns a penalty tuple rather than propagating
(`optimization/objective.py:1277`); a validation exception likewise falls back
(`objective.py:1362`). **Both produce `TrialState.COMPLETE`.** And missing
objective keys are inserted by `_validate_metric_keys()` before storage
(`objective.py:1335`, `:1346`), so "the key is present" does not mean anything
computed it.

A success assertion must therefore establish, together:

- exactly the requested number of trials, all `COMPLETE`;
- **no** `rejected_reason` / training-error / validation-error / fallback user
  attrs;
- every raw metric and every objective value **finite**;
- every direction is `MINIMIZE` — not merely that there are three of them;
- each objective equals the independently specified transform of its raw
  metric, within tolerance (see decision 3).

Even that does not prove training ran: a no-op `train_fn` leaves an
initialized model that validates fine and satisfies all of the above. So
Phase 1 wraps the real `default_train_fn` in a spy that counts invocations and
asserts optimizer steps actually occurred. That asserts *execution*, never
model quality.

### 2. Assert structure and differential properties, never metric values

Two epochs on 128 simulations is noise; `calibration_error < 0.1` would be a
flake generator. Permitted: structural assertions, relational assertions
(objective == documented transform of raw), and differential assertions of the
form #76 endorses. Excluded, per #76: recorded golden values, which bake in
whatever is currently wrong.

### 3. Tolerance, not equality, when comparing objectives to user attrs

User attrs are rounded to six decimals (`objective.py:1347`) while objective
extraction reads the unrounded summary (`objective.py:1357`, `:1408`). Literal
`==` is wrong. Compare with an absolute tolerance that covers the rounding,
and **require finiteness first** — see the false-positive in Phase 2.1.

### 4. Every study gets its own `CheckpointPool` under `tmp_path`

`GenericObjective` creates a default pool when none is passed
(`objective.py:914`), rooted at the relative path `checkpoints`
(`optimization/checkpoint_pool.py:41`), writing `trial_0000/weights.weights.h5`
(`:72`). Every study restarts trial numbering, so studies overwrite each
other's files, and save failures are swallowed (`:76`) — the suite would stay
green while colliding.

This is not hypothetical: **my own probe runs created `checkpoints/trial_0000`
and `trial_0001` in the worktree.** It is gitignored (`.gitignore:161`) so it
does not dirty the repo, but the collision risk between tests is real.

Pass an explicit `CheckpointPool(pool_dir=tmp_path / ...)` per study, including
for multiple studies inside one test.

### 5. Reverses a stated convention, narrowly

`plan-testing-gaps-done.md` records: *"Tests must not require BayesFlow or
Keras."* That convention is the direct cause of the gap — a test that cannot
build an approximator cannot walk the seam.

**Correction to v1**: v1 said every current test honours it. False —
`tests/test_builders/test_workflow.py:3` already imports keras and exercises
real optimizer schedules at `:67`. So this is a narrower reversal than v1
claimed: the new convention is that `tests/test_end_to_end/` may import
BayesFlow and Keras, extending a precedent rather than breaking new ground.

### 6. Do not seed by passing a sampler instance

`optimize()` **silently skips soft constraints** when given a user-supplied
sampler instance (`api.py:478`, logged as a warning). A generic "seed the
sampler" fixture would therefore disable exactly the thing a soft-constraint
test asserts. Seed NumPy; pass sampler *presets* by name; enumerate trial
configurations explicitly rather than relying on sampler determinism.

## Phases

### Phase 1 — Isolated fixtures, observed training, smoke test

`tests/test_end_to_end/conftest.py`:

- `tiny_simulator` — `bf.simulators.make_simulator([prior_fn, likelihood_fn])`,
  scalar `theta ~ N(0,1)`, `x` of shape `(4, 1)`.
- `tiny_adapter` — `.as_set(["x"]).rename("theta", "inference_variables")
  .concatenate(["x"], into="summary_variables", axis=-1)`.
- `tiny_search_space` — **every dimension spelled out as a `Dimension`
  object**, e.g.
  `FlowMatchingSpace(subnet_width=IntDimension("fm_subnet_width", constant=16),
  subnet_depth=IntDimension("fm_subnet_depth", constant=1),
  dropout=FloatDimension("fm_dropout", constant=0.0))`, and the same shape for
  `DeepSetSpace` (`summary_dim=4, depth=1, width=16`, **dropout pinned**) and
  `TrainingSpace` (`batch_size=32, initial_lr=1e-3`).

  v1 wrote `FlowMatchingSpace(subnet_width=16)`. That is not merely shorthand:
  dimension discovery filters on
  `isinstance(getattr(self, f.name), _DIMENSION_TYPES)`
  (`search_spaces/base.py:276`), so a bare int is **silently dropped** and the
  builder loses the key. Pinning dropout matters too — left unpinned
  (`inference/flow_matching.py:64`, `summary/deep_set.py:61`) it is sampled,
  and trials stop being identical when a test needs them to be.

  Pinned this way the approximator is ~5.3K params, far under any budget, so
  no trial is budget-rejected and `n_trials` means what it says.
- `checkpoint_pool` — per-test pool under `tmp_path` (decision 4).
- `training_spy` — wraps and calls the real `default_train_fn`, counting
  invocations (decision 1).
- `run_study(**overrides)` — wraps `optimize()` with the fast defaults
  (`n_trials=2, epochs=2, num_batches=4, sims_per_condition=32,
  n_posterior_samples=50, storage=None, show_progress_bar=False`).
- Module-level `pytestmark = pytest.mark.endtoend`. **Registering the marker in
  `pyproject.toml` does not apply it** — v1 conflated the two, and without the
  `pytestmark` line `-m "not endtoend"` would deselect nothing.
- Set the matplotlib `Agg` backend before imports if anything here imports
  `results.visualization`; `tests/test_visualization.py:14` does this today,
  and that side effect is not inherited when only this directory runs.

`pyproject.toml` → `[tool.pytest.ini_options]`:

```toml
markers = ["endtoend: runs a real optimize() with a real approximator (~15s each)"]
```

`test_optimize_smoke.py::test_default_objectives_complete_successfully`:
the full success assertion of decision 1, plus `trials_to_dataframe(study)`
carrying the objective columns, plus the training spy showing real steps.

**Must fail if**: training is replaced by a no-op; any trial silently takes the
training-error or validation-error fallback; a direction is flipped.

**Explicitly does not guard** the #72 defects — see Phase 2.

### Phase 2 — Metric-path integration regressions

v1's central claim was that the Phase 1 smoke test "would have caught both #72
integration failures." **That was wrong**, and it is the most important thing
this revision fixes. The smoke test requests `calibration_error` and `nrmse`,
both already in `DEFAULT_METRICS`
(`validation/registry.py:734` = `calibration_error, nrmse, correlation,
coverage, rmse, contraction`). Delete objective-metric threading from
pre-flight, or constraint producers from final validation, and that test still
passes. It never requests a non-default metric.

So the load-bearing cases live here, and they are the point of the suite:

1. `test_registered_non_default_metric_is_optimizable` —
   `objective_metrics=["log_gamma"]` (registered, **not** default). Assert
   every value **finite first**, then `value ≈ -user_attrs["log_gamma"]`
   within tolerance.

   The finiteness guard is not decoration. If final validation stopped
   requesting `log_gamma`, sanitization inserts the raw penalty `-inf`
   (`objectives.py:291`, stored at `objective.py:1335`), conversion yields
   `+inf`, and `value == -raw` **still holds**. Without the finiteness check
   this test passes with the defect present.

   **Must fail if**: `objective_metrics` stops being threaded to
   `default_validate_fn`; the direction transform is reversed.

2. `test_multi_output_constraint_key_is_computed` — a hard constraint on
   `left_coverage_90` puts that key in `user_attrs`.
   **Must fail if**: `producer_for_key()` is dropped from `_pipeline_metrics()`.
   **Does not guard** constraint *enforcement* — hence:

3. `test_unattainable_hard_constraint_rejects_deterministically` —
   `("left_coverage_90", 1.1, "below")`. Coverage is a mean of booleans
   (`validation/registry.py:690`), so it is unattainable regardless of model
   quality, making the rejection deterministic. Assert
   `rejected_reason == "metric_constraint"` and the penalty tuple.
   Hard-rejected trials are excluded from the trained count
   (`optimization/study.py:780`), so set an explicit small `max_total_trials`
   or the study will not terminate. Pair with a permissive threshold asserting
   acceptance.
   **Must fail if**: the `_check_hard_constraints()` call at `objective.py:1351`
   is removed — which test 2 alone would not notice.

4. `test_direction_registered_under_alias_is_honoured` — register under the
   alias `"cal_error"`, then optimize. Assert against the **independent
   formula** `1 - raw` (the registration default, `objectives.py:436`), not
   against whatever the implementation computes. Additionally pass the *alias*
   to `optimize()` as the objective name: registering under an alias while
   optimizing the canonical name does not test the canonicalization boundary
   in the direction the bug ran.
   Requires a fixture that snapshots and restores `METRIC_DIRECTIONS` and
   `HIGHER_IS_BETTER` **in place** (mutate the existing objects, do not rebind)
   — existing tests import those objects directly, so rebinding the module
   attribute would leave them holding the old dict.

5. `test_soft_constraint_reaches_the_sampler` — one integration assertion for
   the soft path, which has separate API/sampler wiring. Do **not** pass a
   sampler instance (decision 6).

### Phase 3 — Hook and selection tests

**v1's Phase 3 did not work.** `optimize()` calls `check_pipeline()` before the
study runs (`api.py:460`), and pre-flight trains its own approximator and calls
the *same* custom `validate_fn` (`pipeline.py:331`), raising `PipelineError` on
missing or non-finite required metrics (`pipeline.py:365`, `:372`). A
two-response iterator has three consumers — pre-flight, trial 0, trial 1 — so
it would shift trial identities, exhaust, or reject before optimization began.

The hook receives no trial argument (`api.py:198`), so the fixture must define
the contract explicitly:

- the hook returns a **valid** response for the pre-flight call (identified by
  call ordering, since pre-flight is always first), and controlled responses
  thereafter;
- unexpected extra calls fail loudly rather than returning a default;
- the test asserts the observed call count matches the contract.

Do **not** mock away pre-flight — that removes part of the seam under test.

`test_optimize_selection.py`, with `cost_metric="param_count"` and identical
pinned architectures so **both trials carry the same cost coordinate**. v1
used the default `cost_metric="inference_time"` (`api.py:130`), which with a
custom validator measures the hook's own execution time
(`objective.py:1297`) — a fast, metric-omitting trial can be non-dominated on
cost, so both trials land on the Pareto front and the assertion passes even
with the direction inverted.

- `test_omitted_metric_ranks_strictly_worse_than_a_bad_reported_value` —
  v1 stated this requirement backwards in prose (reporting catastrophic
  "must rank worse than" omission) and then weakened it to "does not
  outrank", which a broken implementation assigning every trial the same
  penalty would pass. The correct requirement, matching
  `tests/test_optimization/test_direction_end_to_end.py:185`:

  > a missing metric receives a **strictly worse** metric objective than any
  > finite reported value.

- `test_better_metric_wins` — enumerate the two configurations deliberately
  and assert **exact** Pareto membership, as the existing synthetic suite does
  (`test_direction_end_to_end.py:111`, `:116`, `:139`), not merely that the
  better trial is contained in `best_trials`.

### Phase 4 — CI measurement, docs, close-out

- Run the suite **under CI's actual command** (`pytest tests/ --cov=...`) and
  record the delta. Only then claim a budget.
- Worker configuration: bayesflow defaults `use_multiprocessing=False` and
  passes `workers` through `build_dataset`, so the logged
  "Using 16 data loading workers" is a Keras **thread** pool, not 16 torch
  worker processes — v1 implied the latter. `default_train_fn` forwards no
  worker option (`objective.py:102`), so pinning it needs a real mechanism
  covering both pre-flight and trials. Decide from the measured CI run; do not
  pre-empt.
- `docs/TODO.md`: record the suite and the narrowed convention.
- `CHANGELOG.md`: note that `tests/test_end_to_end/` runs real approximators.
  v1 proposed announcing a "new Keras test dependency" — **cut**, it is not
  new (`test_builders/test_workflow.py:3`) and not a dependency change.
- Comment on #76: item 2's bugs were already fixed and already unit-covered;
  what this adds is integration coverage; the slow-marker question is answered
  by measurement.

## Out of scope

- **#77 (schema stamping race)** — needs a storage-level atomic operation and a
  genuinely racing test. This suite gives it somewhere to live.
- **#81 (`cost_metric=None`)** — its test list asks for "one end-to-end
  `optimize()` run, per the gap #76 describes"; Phase 1's fixtures are that
  run. Deliberately sequenced after.
- **Pruning, intermediate validation, open-ended stopping** — unreachable at
  2 epochs (`validation_callback.py:236`). Raising epochs to reach them would
  multiply the runtime; a separate targeted test is the better trade.
- **Persistence / resume / serialization** — in-memory studies by design.
- **Golden/reference studies** — argued against in #76.
- **`examples/getting_started.ipynb:126`** carries a `train_fn` labelled
  "BayesFlow 2.0.8 expects `num_batches` in fit()". `default_train_fn` already
  passes `num_batches` (`objective.py:106`) and my probes ran without the hook,
  so the workaround is stale. Separate cleanup issue.

## Risks

| Risk | Mitigation |
|---|---|
| Assertions pass while the guarded defect is present | Every test states the deliberate regression that must fail it. Verify by actually introducing each one before merging. |
| CI slower than local; budget claim unverified | Phase 4 measures under CI's own command before any claim. |
| Checkpoint collisions between studies | Per-test `CheckpointPool` under `tmp_path` (decision 4). |
| Global direction registry leaking | Snapshot/restore **in place**, Phase 2.4. |
| Seeding via a sampler instance silently disables soft constraints | Decision 6: seed NumPy, use presets by name. |
| BayesFlow API drift breaking fixtures | That is the point — this suite is the only thing that would notice. |

## Mutation verification (done)

The plan requires each test to name the regression that must fail it. Each was
introduced into the source, the targeted test run, and the source restored.
All six were caught:

| Regression introduced | Test that failed | Guards |
|---|---|---|
| `default_train_fn` returns before `fit()` | `test_default_objectives_complete_successfully` | training actually runs |
| `_pipeline_metrics([])` — objectives not threaded | `test_registered_non_default_metric_is_optimizable` | the #72 headline defect |
| `producer_for_key()` replaced by a registered-name check | `test_multi_output_constraint_key_is_computed` | the #72 constraint defect |
| `_check_hard_constraints()` call disabled | `test_unattainable_hard_constraint_rejects_deterministically` | constraint enforcement |
| `register_metric_direction` stops canonicalizing | `test_direction_registered_under_an_alias_is_honoured` | the alias boundary |
| `log_gamma` `to_minimize` inverted to identity | all three of `test_the_better_trial_is_exactly_the_pareto_front`, `test_registered_non_default_metric_is_optimizable`, `test_omitted_metric_ranks_strictly_worse_than_a_bad_reported_value` | the #72 defect class |

The last row is the one that matters most: the direction-inversion class #72
was about is now caught by three independent tests, none of which needs anyone
to have thought of that particular path.

## Review history

**v1 → v2**, after an adversarial review by `gpt-6-astra` (low effort) via
codex CLI. I verified each finding against the source before accepting it; all
of the following were confirmed:

| Finding | Verified at | Effect on plan |
|---|---|---|
| Smoke test uses only `DEFAULT_METRICS`, so it guards neither #72 defect | `validation/registry.py:734` | Central claim of v1 deleted; regressions moved to Phase 2 |
| Pre-flight calls the custom `validate_fn` before the study | `api.py:460`, `pipeline.py:331` | v1's Phase 3 rewritten; it could not have run |
| `COMPLETE` is reachable via training/validation fallbacks | `objective.py:1277`, `:1362` | Decision 1 added |
| `value == -raw` passes when both are infinite | `objectives.py:291`, `objective.py:1335` | Finiteness guard added |
| User attrs rounded to 6dp; objectives are not | `objective.py:1347` vs `:1357` | Tolerance instead of equality |
| Default `CheckpointPool` writes to relative `checkpoints/` | `checkpoint_pool.py:41`, `:72` | Decision 4; confirmed my probes created it |
| Constraint-key test does not guard enforcement | `objective.py:1351` | Phase 2.3 added |
| Cost = hook execution time defeats the selection test | `api.py:130`, `objective.py:1297` | `cost_metric="param_count"` |
| "Does not outrank" is weaker than the real requirement | `test_direction_end_to_end.py:185` | Strict inequality; prose direction fixed |
| Sampler instance silently disables soft constraints | `api.py:478` | Decision 6 |
| Bare ints are silently dropped from dimension discovery | `search_spaces/base.py:276` | Fixtures spelled out as `Dimension` objects |
| Registering a marker ≠ applying it | — | `pytestmark` added |
| Keras already imported in tests | `test_builders/test_workflow.py:3` | Decision 5 narrowed |
| "Held in place by nothing" is false | `test_metric_name_inventory.py:219`, `:254`, `:290`, `:322` | Corrected to "no *integration* coverage" |
| `"below"` comparison is at `:1022`, not `:1018` | `objective.py:1022` | Citation fixed |
| 2 epochs never reach validation warmup | `validation_callback.py:236` | Non-coverage stated up front |

Not accepted: nothing. Every checked claim held.

## Estimate

6–8 hours (up from v1's 4–6). The increase is Phase 3's pre-flight contract and
the per-test mutation verification, both of which v1 omitted rather than
sized.
