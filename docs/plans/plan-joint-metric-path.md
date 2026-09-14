# Plan: a joint, data-dependent metric path

Design session for [issue #82](https://github.com/matthiaskloft/bayesflow-hpo/issues/82),
which asks for a validation path that can carry a *joint*, *data-dependent*
metric with TARP as its first consumer, and for
[issue #75](https://github.com/matthiaskloft/bayesflow-hpo/issues/75), which
asks for a joint *log-density* metric (`coverage_error` from
`bayesflow-calibration-loss`). Both issues are scoping-only and both end on the
same question — whether they are one capability. This document answers that and
the remaining open questions, and specifies the contract. **No production code
changes are made by this plan.**

Every claim about this repository below was read from the source at the cited
line. Every claim about TARP is from Lemos et al. (2023) as recorded in
[`references.md`](../references.md) (OpenAlex `W4319453761`), reached through
`bayesflow_irt.sbc.compute_tarp_coverage` and verified against full text; §6
records the tracing.

This document was revised after an independent review that verified it against
source. The review confirmed §1's two findings and §2's contract shape, and
refuted three integration decisions — the pinning mechanism, the summary merge,
and the draws shape. Those are now §D7, §D3 and §D3 respectively, rewritten.
§7 records what the first version got wrong, because an implementer who read
only the first version would have built all three. A later pass measured the
per-trial cost (§D9) and verified the two outstanding paper claims (§6); both
changed conclusions, so neither is left as an assumption.

---

## 1. The finding that settles the design

**This package has already built the joint, data-dependent metric path — twice,
by duplication, outside the pipeline.**

`make_lc2st_validate_fn` ([`validation/c2st.py:511`](../../src/bayesflow_hpo/validation/c2st.py))
is a `ValidateFn` factory that re-implements the body of
`run_validation_pipeline` ([`validation/pipeline.py:23`](../../src/bayesflow_hpo/validation/pipeline.py)):
it builds its own `infer_fn` (duplicating `pipeline.py:57-62` argument for
argument), runs its own condition loop, calls `compute_condition_metrics` per
parameter, and *additionally* assembles

- joint draws — `draws_3d`, shape `(n_sims, n_samples, n_params)`
  (`c2st.py:630`), and
- the **data** — `obs`, concatenated from `validation_data.data_keys`
  (`c2st.py:627`),

and passes both to `lc2st(...)`. L-C2ST is precisely a joint, data-dependent
metric. The only reason it does not appear in the registry is that
`register_metric` ([`validation/registry.py:74`](../../src/bayesflow_hpo/validation/registry.py))
cannot express it — the same wall #82 and #75 both hit, confirmed by
`_reshape_for_bf` (`registry.py:423`) and its comment at `registry.py:420-421`:
"Since validation runs per-parameter, the last axis is always 1."

The second copy is `bayesflow_irt.hpo.make_irt_hooks`' validation closure, cited
in #82's constraint 2. TARP would be the third.

So open question 5 — *"does it belong in `bayesflow-hpo` at all, or should the
joint metric be computed by the dependent package and passed in as a scalar?"* —
is already answered by the repository's own history: the scalar-from-outside
route was taken for L-C2ST, and it produced a ~140-line duplicate of the
pipeline that drifted. The duplicate has no `timing`, no `per_parameter`
result, no `cleanup_trial()` call, and returns a bare `dict` rather than a
`ValidationResult`. It has also drifted *numerically*: it pools every
parameter's rows into one flat list with a `param_key` column (`c2st.py:607`)
and aggregates them together, where `run_validation_pipeline` aggregates per
parameter and then `nanmean`s the per-parameter summaries
(`pipeline.py:128-134`). Those disagree whenever conditions are unbalanced
across parameters. **The capability belongs here.**

It also produced a silent defect, which is the second finding:

**`lc2st` has no entry in `METRIC_DIRECTIONS`.** Confirmed by inspection of
[`objectives.py:265-350`](../../src/bayesflow_hpo/objectives.py): the table
holds `correlation`, `contraction`, `log_gamma`, `calibration_error`,
`mean_calibration_error`, `nrmse`, `rmse`, `sbc_ks`, `sbc_chi2` — and nothing
else. The full path, traced end to end: `"lc2st"` is in neither
`METRIC_DIRECTIONS` nor `_ALIASES`, so `canonical_metric_name` returns it
unchanged; `validate_objective_metric_kinds` (`registry.py:265`) deliberately
passes unknown names, since "a custom `validate_fn` may return objective values
that are not registered"; `_direction_for` (`objectives.py:506`) returns `None`;
`_metric_to_minimize` passes the value through; and `worst_raw_value` returns
`math.inf` (`objectives.py:573`). That `inf` reaches trials at
`optimization/objective.py:197`, `:199`, `:988` and `:1010`.

The configuration is supported, not hypothetical: `make_lc2st_validate_fn` is
exported from `__init__.py`, and its summary key is literally `"lc2st"`
(`c2st.py:649`), so `validate_fn=make_lc2st_validate_fn(),
objective_metrics=["lc2st"]` runs today. The *direction* is accidentally
correct — the statistic is `mean((p - 0.5)**2)` (`c2st.py:346`), and
lower-is-better values pass through unchanged. The *worst case* is not:
`probs_class0` are classifier probabilities, so the statistic is bounded above
by 0.25, and a trial that failed to report it is scored infinitely bad rather
than 0.25-bad. Defensible in isolation; wrong as a Pareto coordinate.

A joint metric path that did not also carry a direction registration would
reproduce this for `tarp_error`.

---

## 2. Decisions

### D1 — One capability, not two (open question 1)

#75 needs joint **log-densities**, which requires the *approximator*. #82 needs
joint **draws plus an x-derived reference**, which requires the *data*. Both
already live in the same scope: the condition loop of `run_validation_pipeline`
holds `approximator`, `sim_batch` (which carries param keys and data keys alike
— `pipeline.py:52-56` reads `validation_data.simulations[0].keys()` for exactly
this reason), and `draws` in its joint `(n_sims, n_samples, n_params)` form
*before* it is sliced per parameter at `pipeline.py:92`.

The capability is therefore not "log-densities" or "reference points". It is:
**stop discarding what the loop already has.** One contract, carrying the union.

### D2 — The contract (open question 2)

A single frozen-dataclass argument rather than positional arrays:

```python
@dataclass(frozen=True)
class JointMetricInputs:
    draws: np.ndarray                     # (n_sims, n_samples, n_params), always 3-D
    true_values: np.ndarray               # (n_sims, n_params)
    param_keys: tuple[str, ...]           # column order of the two above
    sim_batch: Mapping[str, np.ndarray]   # the whole condition batch
    data_keys: tuple[str, ...]            # which of its keys are data
    approximator: Any                     # for #75's log-densities
    cond_id: int

JointMetricFn = Callable[[JointMetricInputs], dict[str, float]]
```

Every field is load-bearing, and the existing L-C2ST code is the proof: it
assembles `sim_batch` + `data_keys` by hand at `c2st.py:617-627`, reads the
column order from `validation_data.param_keys` (nothing else records it —
`pipeline.py:90`), and seeds per condition as `seed + cond_id` (`c2st.py:643`).
`approximator` covers #75. `n_posterior_samples` is deliberately *not* a field:
a floor-aware metric reads it off `draws.shape[1]`, which cannot disagree with
the array it is describing.

Open question 2 asked whether the metric path gets the whole validation batch
including `x`, or a declared subset, and flagged that the former "leaks the data
into every metric". **It gets the whole batch, and nothing leaks**, because
joint metrics are dispatched separately: a `MetricFn` registered through
`register_metric` keeps its `(draws[n, s], true_values[n]) -> dict` signature
untouched and never sees `JointMetricInputs`. The dataclass is also what lets
`approximator` be added for #75 without breaking a signature TARP already
depends on.

### D3 — Where it runs, and the two shape hazards

Dispatch happens inside the existing loop in `run_validation_pipeline` — not in
a second pass. Two reasons, both read from the source: inference is the
expensive step (`timing["inference"]`, accumulated at `pipeline.py:80`) and a
second pass doubles it; and `cleanup_trial()` runs at the end of each iteration
(`pipeline.py:106`), so a joint metric evaluated outside the loop would face
released state.

Two hazards in that loop must be specified, or the feature ships broken and
quiet.

**(a) `draws` is rebound to 2-D in the single-parameter branch.**
`pipeline.py:96-97` does `draws = np.squeeze(draws, axis=-1)` when there is one
parameter — an in-loop rebinding of the loop's own variable. A joint dispatch
placed after the per-parameter branch would therefore receive a 2-D array in
single-parameter studies, violating D2's shape guarantee outright;
`compute_tarp_coverage` raises on `posterior_draws.ndim != 3`. The existing
L-C2ST duplicate already works around exactly this (`c2st.py:629-631`, "Ensure
draws are 3D for lc2st").

**That is not the only squeeze, and moving the dispatch is not sufficient.**
`make_bayesflow_infer_fn` *itself* collapses the trailing axis before
`pipeline.py` ever sees the array (`validation/inference.py:57-61`):

```python
if len(param_keys) == 1:
    draws = np.asarray(post_draws[param_keys[0]])
    if draws.ndim == 3 and draws.shape[-1] == 1:
        draws = np.squeeze(draws, axis=-1)
    return draws
```

So in an ordinary scalar-parameter study the joint path receives a 2-D array no
matter where in the loop it is dispatched, `compute_tarp_coverage` raises on
`ndim != 3`, and **every condition takes D8's failure path** — a metric that
appears to be configured and silently never produces a value. An earlier
revision of this plan claimed that reordering the dispatch established the 3-D
invariant; it does not.

**Decision, in two parts:**

1. Dispatch joint metrics before the per-parameter branch, and bind the
   pipeline's squeezed array to a new name rather than rebinding `draws` — this
   removes the `pipeline.py:96-97` hazard.
2. **Normalize explicitly at the joint boundary**: the dispatch re-expands a
   2-D result with `draws[..., None]` before constructing `JointMetricInputs`,
   so D2's 3-D guarantee is established by the joint path itself rather than
   assumed from upstream. Nothing in `infer_fn` or the marginal path changes,
   so marginal behaviour is untouched.

**Acceptance test:** a single scalar-parameter study driven through the *real*
`make_bayesflow_infer_fn`, asserting the joint metric receives
`(n_sims, n_samples, 1)`. A mock that returns 3-D draws passes trivially and
would not have caught this, which is why the test must use the real closure.

**(b) The multi-parameter top-level summary is built *from* the per-parameter
summaries.** `pipeline.py:128-134` takes its key set from the first parameter's
summary and `nanmean`s across parameters. A joint key deliberately absent from
`per_parameter` is therefore absent from `overall_summary` too — the objective
would find no `tarp_error`, substitute `worst_raw_value("tarp_error")` for
*every* trial, and the study would silently optimize a constant. That is the
exact failure mode §1 criticizes.

**Decision:** joint outputs are aggregated separately across conditions and
merged into the top-level summary as
`overall_summary = {**per_parameter_mean_summary, **joint_summary}`. They stay
out of `per_parameter` — a joint metric has no per-parameter value, and writing
one would be a lie the `nanmean` would then average. In `condition_metrics`,
whose multi-parameter form is one row per (condition, parameter), joint values
are **not** written: they would have to be duplicated across the parameter rows
of a condition, and a duplicated value is one that someone will later average.
They appear in the summary only.

`aggregate_condition_rows` (`metrics.py:54`) itself needs no change — it derives
numeric keys from the first row, skips `id_cond`/`n_sims`, and `nanmean`s, which
is correct for a list of per-condition joint scalars. The change is in the
pipeline's assembly, not in that function. The first version of this plan
conflated the two.

`make_lc2st_validate_fn` is refactored onto this path as the proof, which
deletes the duplicate loop. That refactor is the acceptance test for the
contract: if L-C2ST does not fit it, the contract is wrong.

### D4 — The routing surface, which decides whether the metric runs at all

A joint registry is not enough. The name has to survive four lookups that
currently consult the marginal registry only, and every one of them fails
*silently*:

- `producer_for_key` (`registry.py:214`) reads `_REGISTRY` and `_OUTPUTS`.
- `_metric_names_for_pipeline` (`optimization/objective.py:775-787`) builds the
  `metrics=` list passed to `run_validation_pipeline` by calling
  `producer_for_key`, and **drops names it returns `None` for** — so
  `objective_metrics=["tarp_error"]` would request nothing, compute nothing, and
  take the penalty. That is the same silent-inactivity bug the comment block at
  `:777-783` exists to document.
- `resolve_metrics` raises on a name it cannot resolve, so a joint name must not
  simply be passed through to it either.
- `validate_objective_metric_kinds` (`registry.py:265`) reads `_KINDS`, so
  D6's `kind="diagnostic"` guard on `tarp_error_random` does not fire unless
  joint kinds are visible there.

**Decision: one registry, with a marker on the entry, not a parallel registry.**
`register_joint_metric(name, fn, ...)` writes into the same `_REGISTRY` /
`_KINDS` / `_OUTPUTS` / `_DESCRIPTIONS` / `_REQUIRES` tables plus a
`_JOINT: set[str]`, so `producer_for_key`, `_metric_names_for_pipeline`,
`validate_objective_metric_kinds` and `describe_metrics` keep working untouched,
and only two sites branch: `resolve_metrics` (which must return joint callables
separately) and the dispatch in `run_validation_pipeline`.

This reverses the first version's choice of a parallel registry. The separation
that matters is the *callable signature*, which the marker preserves, not the
storage. A parallel registry buys nothing and costs four edits that are silent
when forgotten.

### D5 — Directions for the new metrics (open question 3)

```python
register_metric_direction("tarp_error", higher_is_better=False, worst_raw=1.0)
register_metric_direction("lc2st", higher_is_better=False, worst_raw=0.25)
```

`tarp_error` is the median over credibility levels of `|ECP - level|`
(`compute_tarp_coverage` docstring). Both terms lie in [0, 1], so the deviation
is bounded by 1 and the median of bounded values is bounded. Lower is better.
Unlike `log_gamma` it needs no infinite penalty — #82's own reading, confirmed.

`lc2st` closes §1's defect: `(p - 0.5)**2` for a probability `p` is bounded by
0.25. This is worth landing on its own, independently of TARP.

Registering a lower-is-better metric changes no conversion; what it buys is the
tighter `worst_raw` and an explicit scale, which `objectives.py:294-299`
documents as the thing that distinguishes a known [0, 1] metric from an unknown
one for the training-loss fallback.

### D6 — The reference-mode contract (#82 constraint 3)

The measured table in #82 is two different metrics sharing a name: 0.0103 under
a random reference and 0.1678 under a classical one, for the *same* degenerate
estimator. A stored trial value without its reference mode is uninterpretable.

`compute_tarp_coverage` already refuses to guess — it reports
`reference_mode="provided"`, not `"data_dependent"`, whenever the caller
supplies the array (`bayesflow-irt`, `src/bayesflow_irt/sbc.py` at commit
`ffc68d5`), because a supplied reference may still have been generated
independently of the data. `bayesflow-hpo` inherits that refusal and adds:

**Different keys, not one key with a mode field.** A random-reference run emits
`tarp_error_random` and is registered `kind="diagnostic"`; only a provided
reference emits `tarp_error`, registered `kind="objective"`. Two numbers that
cannot be compared must not be comparable by name, and this is the only form
that survives a `trials_to_dataframe()` read six months later.

**The reference itself** is supplied as a `Callable[[JointMetricInputs],
np.ndarray]` returning `(n_sims, n_params)` for that condition, or as an
array-per-condition sequence of the same shape indexed by `cond_id`. A single
`(n_sims, n_params)` array reused across conditions is **rejected**: the truths
differ per condition, so a shared reference is a different metric on each one.
The first version of this plan said "an array" without saying which, and the
array form is the one a user reaches for first.

**The callable receives `approximator`**, since it receives the whole
`JointMetricInputs`. A reference provider that samples the posterior it is
auditing produces a reference that looks data-dependent and is not — the exact
blind spot the two-key split exists to prevent. This must be a documented
warning on the provider parameter, because nothing can detect it.

`bayesflow-hpo` does not compute reference points. That is what keeps #82
constraint 4 out of this package: whether `classical_item_statistics` covers
1PL/2PL but not PCM/GPCM is a `bayesflow-irt` question about *which* reference
is available, not a question about the contract. The companion `bayesflow-irt`
issue owns it.

### D7 — Pinning the nuisance parameters (#82 constraint 5)

`compute_tarp_coverage(..., resolution=20, metric="euclidean",
standardize=True, seed=None)` moves with every one of those, with the number of
validation datasets, and with the reference draw. Its docstring also records a
**floor that `tarp_error` cannot go below**, because `f_i` is supported on
`{0, 1/n_draws, ..., 1}` and its ECDF is a staircase: at `n_draws = 5` the floor
is exactly `1 / (resolution + 1)` and dominates everything else; by
`n_draws = 100` it has fallen below the Monte Carlo noise from a few thousand
simulations, so beyond that the residual is set by `n_simulations` rather than
by discreteness. The same docstring warns that any quoted floor figure is
meaningful only alongside the `n_simulations` and `resolution` it was measured
at. This plan therefore quotes the relation and not a table of numbers.

That is the same defect shape as #75's constraint 2 and the #72 findings, and
here it is worse than #75's, because `n_posterior_samples` is a *validation*
setting a caller can change between studies without touching the search space at
all.

**The pin does not go in the objective schema.** The first version of this plan
said it did, and that is impossible: `bayesflow_hpo_objective_schema` is a
positional list of objective column names — `api.py:875-876` discards anything
that is not a `list`/`tuple`, and `schema_matches` (`objectives.py:806`) rejects
on length mismatch first. Appending settings to it would change its length,
break resumption for every existing stamped study, and misrepresent the number
of objective columns. A `tarp_error_random` registered as a *diagnostic* has no
column there at all, so its mode could not be pinned by that mechanism even in
principle.

**Decision:** a separate, JSON-serializable user attribute
`bayesflow_hpo_joint_metric_settings` holding `resolution`, `metric`,
`standardize`, `seed`, `n_posterior_samples`, the condition count, the reference
mode, and the condition sub-sample size if D9 introduces one — with its own
comparison function and its own refusal message, checked alongside the schema in
`_check_study_compatibility`. Three things the implementation must settle,
which the schema mechanism does not settle for it:

- **A populated study carrying no such attribute.** Stamping happens only on a
  study with no trials (`api.py:983`), so a study started before this feature
  can never acquire one. Decide explicitly: refuse, or accept-and-stamp.
- **What a "provided" pin can actually prove.** A `Callable` reference provider
  is not serializable, so the pin records that *a* provider was supplied, never
  which one. The two-key split mitigates this and does not close it. Say so in
  the docs rather than implying the pin is complete.
- **The concurrent stamp.** Two workers racing on a fresh shared-storage study
  can stamp different settings, since the guard is stamp-only-when-empty. This
  pre-exists for the objective schema; joint settings widen the window.

`seed` must not default to `None` — the factory takes an explicit `int`. The
same docstring notes that with `seed=None` two identical calls return different
numbers.

### D8 — What happens when a joint metric raises

`run_validation_pipeline` has no per-metric `try`/`except`. A joint metric that
raises on one condition aborts the entire validation, so the trial loses its
*marginal* metrics too and drops to `_training_loss_fallback`. That blast radius
is materially worse than today's, because joint metrics are precisely the ones
with optional dependencies and numerical preconditions: L-C2ST fits an sklearn
classifier per fold, and `compute_tarp_coverage` with `standardize=True` raises
on a constant dimension.

**Decision:** joint metrics are dispatched under a per-metric guard that records
the failure and lets the marginal metrics complete. The guard must not swallow
the reason; a failing joint metric should be visible in the trial's user attrs,
not inferable only from a penalty value.

**A per-condition guard is not enough on its own, because `worst_raw` may never
be reached.** `aggregate_condition_rows` (`metrics.py:54`) derives its key set
from `condition_rows[0]` and `nanmean`s later rows, skipping NaN. Combine that
with a guard that simply omits the failed condition's key and the result depends
on *which* condition failed:

- condition 0 succeeds, condition 1 raises → the key exists in row 0, so the
  summary reports the metric **averaged over the conditions that happened to
  succeed**. The objective sees a present, finite, flattering value and never
  applies `worst_raw`.
- condition 0 raises → the key is absent from row 0, so it is absent from the
  key set and **every later success is discarded**.

Scores would therefore depend on failure order, and a model could *benefit* from
failing on the conditions it finds hardest — which is the same class of defect
as the `worst_raw` inversions D5 exists to prevent.

**Decision: any required condition failing invalidates that metric for the whole
trial.** The metric's key is omitted from the summary entirely, so the objective
substitutes its registered `worst_raw` exactly once, and a partially computed
joint metric is never averaged. Per-condition `worst_raw` before aggregation is
the alternative and is rejected: it still yields a finite blend of real and
penalty values, which reads as a mediocre model rather than a broken
measurement.

**Acceptance tests:** failure in the first condition, failure in the last
condition, and total failure must all produce the same outcome — the metric
absent, `worst_raw` applied once, marginal metrics intact.

### D9 — Cost per trial (open question 4) — **measured**

#82 asked for this to be measured before TARP becomes an axis in a search whose
trials are already GPU-bound. It has been, on synthetic arrays of the shapes the
validation pipeline would hand each metric (pure CPU, AMD Ryzen, numpy 2.4.2;
harness in the session scratchpad, reproducible from `compute_tarp_coverage` at
`bayesflow-irt` commit `ffc68d5`). Times are per **condition**; multiply by the
study's condition count for the per-trial cost.

| n_sims | n_draws | n_params | resolution | TARP ms/condition |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 100 | 2 | 20 | 0.5 |
| 100 | 1000 | 2 | 20 | 4.1 |
| 500 | 1000 | 2 | 20 | 18.9 |
| 100 | 1000 | 15 | 20 | 16.8 |
| 500 | 1000 | 15 | 20 | 78.8 |
| 500 | 1000 | 60 | 20 | 308.9 |
| 1000 | 1000 | 60 | 20 | 587.6 |
| 500 | 1000 | 15 | **100** | 75.2 |

Three results, and two of them were not what the design assumed:

**1. TARP is cheap.** At a realistic 500 simulations × 1000 draws × 15
parameters it is 79 ms per condition — under a second per trial for a ten-
condition validation set, against training that is measured in minutes. It does
not need a condition sub-sample, and it is not a reason to keep TARP off an
axis.

**2. `resolution` is free, contradicting the cost model in this plan's own
earlier draft.** Raising it from 20 to 100 changed nothing (78.8 → 75.2 ms, i.e.
within noise). The credibility levels are thresholds applied to an already-
computed vector of `f_i`, not a repeated distance computation, so the
`× resolution` factor the first draft wrote into the cost model does not exist.
Scaling is linear in `n_draws` (×13.9 across a ×16 range), in `n_sims` and in
`n_params` — i.e. exactly `O(n_sims · n_draws · n_params)` and nothing more.
**`resolution` is therefore free to raise for statistical reasons**, which
matters because it sets the `1/(resolution + 1)` floor discussed in D7.

**3. L-C2ST — the metric this package already ships — is the one that cannot
afford to be an axis.** Measured through the repository's own
`validation.c2st.lc2st` on the same shapes:

| n_sims | n_params | n_obs | L-C2ST ms/condition |
| ---: | ---: | ---: | ---: |
| 100 | 2 | 5 | 431 |
| 500 | 15 | 50 | 55,667 |
| 500 | 60 | 50 | 79,814 |

An independent second run replicated the 500 × 15 figure at 53,633 ms (a 4%
spread) with an identical statistic of 0.0654, confirming the measurement is
deterministic and not an artefact of machine load. The small-shape row is that
second run's figure: the first run's 768 ms for it included one-time sklearn
warm-up, which the replication separates out. Neither correction touches the
conclusion.

At matched shapes (500 × 15) that is **~700× TARP**: 56 seconds per condition,
so roughly nine minutes for a ten-condition validation, per trial. It fits an
`MLPClassifier` per fold, five folds per condition, and that dominates
everything else. `n_draws` does not enter — `lc2st` uses draw index 0 only
(`c2st.py:330`).

**Consequences for the design, which are not cosmetic:**

- The sub-sampling escape hatch and the pinned sub-sample size belong to
  **L-C2ST**, not TARP. D7's settings list keeps the field; TARP will not use it.
- **Joint metrics must not run under `PeriodicValidationCallback` by default** —
  a mid-training pruning decision costing nine minutes per interval is not a
  pruning decision. But *simply excluding them is not implementable as stated*,
  and this is the subtlest consequence of the measurement.
  `_run_lightweight_validation` requires **every** `objective_metrics` key and
  returns `None` when any is missing
  (`optimization/validation_callback.py:421-431`); the caller then bails at
  `if raw_scores is None: ... return` (`:258-267`), which skips pruning **and**
  `_update_early_stopping`. So for `objective_metrics=["nrmse", "tarp_error"]`,
  omitting the joint metric would silently disable marginal pruning and
  validation-based early stopping as well — a cost optimization that turns off
  stopping is a regression, not a saving.

  **Decision:** the intermediate metric set becomes explicit rather than implied
  by `objective_metrics`. Joint metrics are excluded from it by default; the
  callback validates against *that* set, so a missing joint key is expected
  rather than a fault. Three configurations must be specified, not discovered:
  `objective_mean` (whose members would silently change meaning if one is
  dropped mid-training), a study whose designated primary metric is the joint
  one, and a joint-only study — where there is no intermediate signal at all and
  the right behaviour is to **reject the configuration up front** or require an
  explicit opt-in, never to degrade into a study that cannot stop early.

  **Acceptance tests:** a mixed-objective study must still prune and still stop
  early with the joint metric excluded; a joint-only study must be rejected or
  opted into explicitly. Without both, this default can silently disable
  stopping.
- `lc2st`'s existing status as a usable objective deserves a documented warning
  with these numbers next to it. Nothing in the package currently tells a user
  that `objective_metrics=["lc2st"]` adds minutes per trial.

**The fraction of `timing["inference"]` — measured on GPU.** #82 phrased the
cost question as a ratio against inference, so it was measured on an RTX 5090
(driver 615.71.09, torch 2.11.0+cu128, bayesflow 2.0.8) through the real
`make_bayesflow_infer_fn` closure — the same one `run_validation_pipeline`
builds — with TARP timed on the same machine at the *same shapes*, so no
cross-machine or cross-shape extrapolation enters the ratio. Harness:
[`bench_inference_ratio.py`](bench_inference_ratio.py).

| n_sims | n_draws | inference ms | TARP ms | TARP as % of inference |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 200 | 1964.1 | 1.1 | 0.06% |
| 200 | 200 | 2004.2 | 2.5 | 0.12% |
| 100 | 400 | 2052.0 | 2.4 | 0.12% |

**TARP is 0.06–0.12% of inference.** The question is closed: it is not a cost
consideration at any shape, and L-C2ST at 53.6 s is roughly *25× the entire
inference pass*, which is the finding that matters.

Two things this run turned up that the design did not anticipate:

**Inference cost is nearly flat in `n_sims` and `n_draws`.** Doubling either
moved it by 2–4% (1964 → 2004 → 2052 ms). `FlowMatching._inverse` integrates an
adaptive ODE (`integrate_adaptive`, Tsit5), so wall time is set by the number of
integration steps, and the GPU parallelizes across samples. A bigger validation
set is therefore close to free on GPU while TARP grows linearly from a tiny
base — the ratio *improves* with scale rather than degrading. Any cost model
that assumes inference is linear in the sample count is wrong on GPU.

**`make_bayesflow_infer_fn` samples the whole condition batch in one call, with
no chunking** (`validation/inference.py:53`). At `n_sims=500, n_draws=1000` that
materializes 500,000 posterior samples at once and needed more than 20 GiB —
it OOMed on a 32 GiB card that had another job on it. This is a property of the
*existing* pipeline, not of the joint metric path, and it bounds how large a
validation set can be on a given card. It qualifies D10: memory is a non-issue
for joint metrics holding one condition's draws, and is emphatically not a
non-issue for the inference step feeding them. Tracked as
[issue #101](https://github.com/matthiaskloft/bayesflow-hpo/issues/101).

Caveat on the absolute numbers: the card was shared with a live study
throughout, and the benchmark ran under a deliberate
`set_per_process_memory_fraction` self-cap so that an overrun would fail the
benchmark rather than the other job. The inference figures are therefore an
upper bound on a contended device. The *ratio* is unaffected — both sides were
measured in the same process on the same device.

### D10 — Memory

**For the joint metrics themselves, a non-issue**, recorded so it is not
re-litigated. They run inside the loop on one condition's `draws`, which the
loop already holds; nothing new is retained across conditions, and
`cleanup_trial()` still runs per iteration.

**For the inference step that feeds them, emphatically not a non-issue**, which
D9's GPU run established after this section was first written.
`make_bayesflow_infer_fn` samples the whole condition batch in one call with no
chunking, so peak memory scales as `n_sims × n_posterior_samples` — 500 × 1000
needed more than 20 GiB and OOMed a 32 GiB card. The package's own defaults
already ask for 100,000 samples in a single forward pass, and
`estimate_peak_memory_mb` covers *training* only, so `max_memory_mb` passes a
trial that then dies after training is paid for. Tracked separately as
[issue #101](https://github.com/matthiaskloft/bayesflow-hpo/issues/101); it is a
property of the existing pipeline and blocks nothing in this plan, but it bounds
how large a validation set the joint metrics can be run on.

---

## 3. `requires=` is documentation-only today

#75 calls `requires="sklearn"` a precedent and asks whether it is sufficient to
keep the dependency optional. **It is not, yet.** `grep -rn 'requires=' src/`
returns zero call sites: `_REQUIRES` (`registry.py:61`) is populated from the
parameter default and read only by `describe_metrics` (`registry.py:413`) for
display. Nothing enforces or gates on it, and no metric declares it.

So `requires="calibration-loss"` on `coverage_error` documents an intent and
gates nothing. Either the import guard is written explicitly (the way
`_require_sklearn` at `c2st.py:52` already does for C2ST, which is the real
precedent), or `requires=` is given enforcement. #75's open question is answered
"no, not as it stands."

---

## 4. What this does *not* decide

- **#75's floor treatment.** Whether `coverage_error` fixes `B` and `n_samples`
  at validation time or reports excess over the floor, and whether the floor
  correction holds empirically, remain #75's open questions. This contract
  carries `approximator`, which is all #75 needed from *this* issue.
- **`mode` for `coverage_error`** (#75 constraint 3). Exploitable at the package
  default; a metric-side decision, not a contract one.
- **The `bayesflow-irt` consumer change.** #82 constraint 2 is explicit that
  `make_irt_hooks`' closure would not reach TARP even through a fixed pipeline.
  A companion issue there is required and is out of this repository's scope.
- **PCM/GPCM reference coordinates** (#82 constraint 4) — see D6.
- **TARP's cost as a fraction of real inference time** — D9 measures TARP in
  absolute terms and settles the design questions that depended on it, but the
  ratio against `timing["inference"]` needs a trained approximator.

## 5. Suggested order for the implementation session

1. Register `lc2st`'s direction (`worst_raw=0.25`). Standalone defect fix, no
   dependency on anything below.
2. `JointMetricInputs`, `register_joint_metric` with the `_JOINT` marker (D4),
   the `resolve_metrics` branch, and the dispatch in `run_validation_pipeline`
   — including D3's two hazards and D8's guard.
3. Refactor `make_lc2st_validate_fn` onto it and delete the duplicate loop. This
   is the acceptance test for the contract.
4. Wire the measured costs in (D9): joint metrics off `PeriodicValidationCallback`
   by default, the condition sub-sample available for L-C2ST, and a documented
   cost warning on `lc2st` as an objective.
5. TARP: the metric, the reference-provider contract (D6), the two-key mode
   split, `bayesflow_hpo_joint_metric_settings` (D7), `tarp_error`'s direction
   entry. Expand the Lemos entry in [`references.md`](../references.md) — it
   currently reads "not currently implemented" and carries none of the
   section-level detail this design leans on.
6. #75's `coverage_error` on the same contract, with an explicit import guard
   per §3.

Steps 1 and 3 are worth landing alone: together they remove a duplicated
pipeline and close a live `worst_raw` defect, independently of whether TARP ever
ships.

## 6. Reference tracing

Traceability was checked against full text rather than assumed, per the
project's source-backing rule. **Both claims flagged as untraced in the first
revision have since been verified** against the PDFs in the project Zotero
library, and [`references.md`](../references.md) has been updated with the
section-level detail.

**Lemos et al. (2023), Sec. 3.1** — verified. The section is titled "High
posterior density coverage testing" and works the case
`p_hat(theta|x) = p(theta)` explicitly: because the HPD generator is then
independent of `x`, so that `H(p_hat, alpha, x) = H(p_hat, alpha)`, the paper
concludes this estimator "has perfect HPD ECP in this case". The same section
states the HPD region generator "is not a positionable credible region
generator", which is why Theorem 3 does not reach it. Sec. 3.2 defines the TARP
generator as the positionable one. The section number carried between
repositories without a check turns out to be correct.

**Modrák et al. (2025), Sec. 4.3** — verified, and **sharper than this plan had
claimed**. Case study 2 is "an incorrect posterior that equals the prior", and
Figure 4 splits the parameter rank distribution by the average value of the
corresponding data elements, reporting that "the distributions for the two cases
exactly compensate to make the overall distribution uniform". So marginal ranks
under a data-ignoring posterior are *exactly* uniform, not merely
indistinguishable from uniform — which is a stronger statement of the problem
than "blind" and makes the marginal metrics' failure structural rather than a
matter of power. The paper's own remedy is a test quantity involving both data
and parameters, recommending the joint log-likelihood as "a useful default" —
which is, independently, what #75 proposes to add. Sec. 4.4 (case study 5) adds
the companion case: correct marginals with wrong correlation structure passes
SBC on the univariate parameters while likelihood-based quantities fail.

**Lemos, Algorithm 2 / Theorem 3 / Secs. 4.2 and 4.3**, and **Linhart et al.
(2023), Theorem 3.1**, remain traced as before — the former through
`compute_tarp_coverage`'s docstring (which carries a direct quotation from the
paper for Sec. 4.3), the latter through `c2st.py:345`.

## 7. What the first version of this document got wrong

Recorded because an implementer who read only the first version would have built
all three, and each fails quietly.

1. **The pin.** It said the nuisance settings and reference mode go into
   `bayesflow_hpo_objective_schema`, "refused the way that attribute already
   refuses". That attribute is a positional list of objective column names and
   cannot carry them; the attempt would have broken resumption for every
   existing study. Now D7, on its own attribute.
2. **The summary.** It said joint outputs "aggregate with no change" and belong
   in the top-level summary only. True of `aggregate_condition_rows`, false of
   the pipeline: the multi-parameter top-level summary is built *from* the
   per-parameter summaries, so a summary-only joint key would have been dropped
   and every trial would have taken the penalty for a metric it computed. Now
   D3(b).
3. **The shape.** It placed the dispatch after the per-parameter branch, where
   `draws` has already been rebound to 2-D for single-parameter studies,
   violating the contract's own shape guarantee. Now D3(a).

It also chose a parallel registry without weighing the four silent lookup
failures that choice creates (now D4), quoted a floor table from a superseded
docstring revision whose successor exists specifically to say bare floor figures
are not meaningful (now D7), and paraphrased `_reshape_for_bf`'s comment inside
quotation marks (now quoted verbatim in §1).

**A third pass, from PR review, corrected three more — all of the same kind, in
that each produces a metric that looks configured and silently never reports:**

4. **The shape fix was incomplete.** D3(a) had claimed that dispatching before
   the per-parameter branch established the 3-D invariant. It does not:
   `make_bayesflow_infer_fn` squeezes the trailing axis itself for
   single-parameter studies, so ordinary scalar studies would have hit the
   failure path on every condition. D3(a) now normalizes explicitly at the joint
   boundary, and requires an acceptance test through the real closure rather
   than a 3-D-returning mock.
5. **D8 had no partial-failure policy.** Reusing `aggregate_condition_rows`
   unchanged makes the reported score depend on *which* condition failed, and
   lets a model benefit from failing on its hardest conditions. Now: any
   required condition failing invalidates the metric for the trial.
6. **Excluding joint metrics from `PeriodicValidationCallback` would have
   disabled marginal pruning and early stopping too**, because the callback
   requires every `objective_metrics` key and no-ops when one is missing. D9 now
   specifies an explicit intermediate metric set and what to do with
   `objective_mean`, a primary joint metric, and joint-only studies.

## References

Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023).
Sampling-based accuracy testing of posterior estimators for general inference.
See [`references.md`](../references.md); OpenAlex `W4319453761`. Sections used,
and their tracing status, are itemized in §6.

Linhart, J., Gramfort, A., & Rodrigues, P. L. C. (2023). L-C2ST: Local
diagnostics for posterior approximations in simulation-based inference. See
[`references.md`](../references.md). Algorithms 1–2 and Theorem 3.1, as
implemented in `validation/c2st.py`.

Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N., Faltejsková, K.,
Gelman, A., & Vehtari, A. (2025). Simulation-based calibration checking for
Bayesian computation: The choice of test quantities shapes sensitivity.
*Bayesian Analysis, 20*(2), 461–488. See [`references.md`](../references.md) and
§6.

— Claude Code (Claude Opus 5)
