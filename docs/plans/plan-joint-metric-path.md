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
`bayesflow_irt.sbc.compute_tarp_coverage`, whose docstring carries the
section-level citations.

---

## 1. The finding that settles the design

**This package has already built the joint, data-dependent metric path — twice,
by duplication, outside the pipeline.**

`make_lc2st_validate_fn` ([`validation/c2st.py:511`](../../src/bayesflow_hpo/validation/c2st.py))
is a `ValidateFn` factory that re-implements the body of
`run_validation_pipeline` ([`validation/pipeline.py:23`](../../src/bayesflow_hpo/validation/pipeline.py)):
it builds its own `infer_fn`, runs its own condition loop, calls
`compute_condition_metrics` per parameter, and *additionally* assembles

- joint draws — `draws_3d`, shape `(n_sims, n_samples, n_params)`
  (`c2st.py:630`), and
- the **data** — `obs`, concatenated from `validation_data.data_keys`
  (`c2st.py:627`),

and passes both to `lc2st(...)`. L-C2ST is precisely a joint, data-dependent
metric. The only reason it does not appear in the registry is that
`register_metric` ([`validation/registry.py:74`](../../src/bayesflow_hpo/validation/registry.py))
cannot express it — the same wall #82 and #75 both hit, confirmed by
`_reshape_for_bf` (`registry.py:423`) and its comment that "validation runs one
parameter at a time".

The second copy is `bayesflow_irt.hpo.make_irt_hooks`' validation closure, cited
in #82's constraint 2. TARP would be the third.

So open question 5 — *"does it belong in `bayesflow-hpo` at all, or should the
joint metric be computed by the dependent package and passed in as a scalar?"* —
is already answered by the repository's own history: the scalar-from-outside
route was taken for L-C2ST, and it produced a ~140-line duplicate of the
pipeline that drifted. The duplicate has no `timing`, no `per_parameter`
result, no `cleanup_trial()` call, and returns a bare `dict` rather than a
`ValidationResult`. **The capability belongs here.**

It also produced a silent defect, which is the second finding:

**`lc2st` has no entry in `METRIC_DIRECTIONS`.** Confirmed by inspection of
[`objectives.py:265-350`](../../src/bayesflow_hpo/objectives.py): the table
holds `correlation`, `contraction`, `log_gamma`, `calibration_error`,
`mean_calibration_error`, `nrmse`, `rmse`, `sbc_ks`, `sbc_chi2` — and nothing
else. `canonical_metric_name` passes unknown names through unchanged (its
docstring: "Passing through unknown names keeps this safe to apply to custom
metrics resolved by a caller's own `validate_fn`"), so an
`objective_metrics=["lc2st"]` study runs without error. Its *direction* is
accidentally correct — the statistic is `mean((p - 0.5)**2)` (`c2st.py:346`),
lower-is-better, and the no-direction branch of `_metric_to_minimize`
(`objectives.py:549`) passes lower-is-better values through unchanged. Its
*worst case* is not: `worst_raw_value` falls back to `math.inf`
(`objectives.py:573`) for a statistic bounded above by 0.25. That is safe but
uninformative — a trial that failed to report `lc2st` is scored infinitely bad
rather than 0.25-bad, which is defensible in isolation and wrong as a Pareto
coordinate.

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

A parallel registry for joint metrics, keyed the same way, with a single
frozen-dataclass argument rather than positional arrays:

```python
@dataclass(frozen=True)
class JointMetricInputs:
    draws: np.ndarray                     # (n_sims, n_samples, n_params)
    true_values: np.ndarray               # (n_sims, n_params)
    param_keys: tuple[str, ...]           # column order of the two above
    sim_batch: Mapping[str, np.ndarray]   # the whole condition batch
    data_keys: tuple[str, ...]            # which of its keys are data
    approximator: Any                     # for #75's log-densities
    cond_id: int

JointMetricFn = Callable[[JointMetricInputs], dict[str, float]]
```

Registered through `register_joint_metric(name, fn, ...)`, mirroring
`register_metric`'s keyword set (`aliases`, `overwrite`, `description`, `kind`,
`requires`, `outputs`) so the two registries stay discoverable together in
`describe_metrics`.

Open question 2 asked whether the metric path gets the whole validation batch
including `x`, or a declared subset, and flagged that the former "leaks the data
into every metric". **It gets the whole batch, and nothing leaks**, because the
registries are separate: a `MetricFn` registered through `register_metric` keeps
its `(draws[n, s], true_values[n]) -> dict` signature untouched and never sees
`JointMetricInputs`. The leak only exists if one registry serves both, which is
what the dataclass avoids — and the dataclass is also what lets `approximator`
be added for #75 without breaking a signature TARP already depends on.

`param_keys` is carried explicitly because `draws` and `true_values` are
column-ordered arrays: a joint metric that assumes a column order is a defect
waiting for someone to reorder `param_keys` in a study config.

### D3 — Where it runs

Inside the existing loop in `run_validation_pipeline`, after the per-parameter
dispatch, not in a second pass. Two reasons, both read from the source:
inference is the expensive step (`timing["inference"]`, `pipeline.py:77`) and a
second pass doubles it; and `cleanup_trial()` runs at the end of each iteration
(`pipeline.py:106`), so a joint metric evaluated outside the loop would face
released state.

Joint outputs are per-condition scalars, aggregated by the existing
`aggregate_condition_rows` (`metrics.py:54`) — they are numeric and
non-identifier, so they aggregate with no change to that function. They belong
in the top-level `summary` only, never in `per_parameter`: a joint metric has no
per-parameter value, and writing one would be a lie that the overall-summary
`nanmean` at `pipeline.py:134` would then average.

`make_lc2st_validate_fn` is refactored onto this path as the proof, which
deletes the duplicate loop. That refactor is the acceptance test for the
contract: if L-C2ST does not fit it, the contract is wrong.

### D4 — `tarp_error`'s direction entry (open question 3)

```python
register_metric_direction("tarp_error", higher_is_better=False, worst_raw=1.0)
```

`tarp_error` is the median over credibility levels of `|ECP - level|`
(`compute_tarp_coverage` docstring). Both terms lie in [0, 1], so the deviation
is bounded by 1 and the median of bounded values is bounded. Lower is better.
Unlike `log_gamma` it needs no infinite penalty — #82's own reading, confirmed.

Register `lc2st` at the same time, `worst_raw=0.25`, closing §1's defect:
`(p - 0.5)**2` for `p` in [0, 1] is bounded by 0.25.

Registering a lower-is-better metric changes no conversion; what it buys is the
tighter `worst_raw` and an explicit scale, which `objectives.py:294-299`
documents as the thing that distinguishes a known [0, 1] metric from an unknown
one for the training-loss fallback.

### D5 — The reference-mode contract (#82 constraint 3)

The measured table in #82 is two different metrics sharing a name: 0.0103 under
a random reference and 0.1678 under a classical one, for the *same* degenerate
estimator. A stored trial value without its reference mode is uninterpretable.

`compute_tarp_coverage` already refuses to guess — it reports
`reference_mode="provided"`, not `"data_dependent"`, whenever the caller
supplies the array (verified in `bayesflow-irt` at `src/bayesflow_irt/sbc.py`,
commit `ffc68d5`), because a supplied reference may still have been generated
independently of the data. `bayesflow-hpo` inherits that refusal and adds two
rules:

1. **Different keys, not one key with a mode field.** A random-reference run
   emits `tarp_error_random` and is registered `kind="diagnostic"`; only a
   provided reference emits `tarp_error`, registered `kind="objective"`. Two
   numbers that cannot be compared must not be comparable by name. This is
   stronger than storing the mode alongside the value, and it is the only form
   that survives a `trials_to_dataframe()` read six months later.
2. **The mode is recorded on the study, not the trial.** A study whose trials
   mix modes is not a study; the mode goes into the objective schema (§D6) and a
   mismatch is refused the way `bayesflow_hpo_objective_schema` (`api.py:874`)
   already refuses an incompatible resumed study.

`bayesflow-hpo` does not compute reference points. It accepts them as an array
or a `Callable[[JointMetricInputs], np.ndarray]`, so a dependent package
supplies its own. This is what keeps #82 constraint 4 out of this package:
whether `classical_item_statistics` covers 1PL/2PL but not PCM/GPCM is a
`bayesflow-irt` question about *which* reference is available, not a question
about the contract. The companion `bayesflow-irt` issue owns it.

### D6 — Pinning the nuisance parameters (#82 constraint 5)

`compute_tarp_coverage(..., resolution=20, metric="euclidean",
standardize=True, seed=None)` moves with every one of those, with the number of
validation datasets, and with the reference draw. Its docstring also records a
**floor that moves with `n_draws`** — measured on a perfectly calibrated
posterior: 0.051 at 5 draws, 0.024 at 20, 0.007 at 100, 0.005 at 500. That is
the same defect shape as #75's constraint 2 and the #72 findings, and here it is
worse than #75's, because `n_posterior_samples` is a *validation* setting a
caller can change between studies without touching the search space at all.

Rules:

- `seed` must not default to `None`. The factory takes an explicit `int`.
- `resolution`, `metric`, `standardize`, `n_posterior_samples`, the number of
  validation conditions, and the reference mode are recorded in the study's
  objective schema alongside the metric names, and a resumed study with
  different settings is refused rather than silently continued.
- The docstring states the ≥100-draw recommendation as a floor condition, not as
  advice.

### D7 — Cost per trial (open question 4)

Not decided here, by design: #82 asks for it to be *measured* before TARP
becomes an axis in a search whose trials are already GPU-bound, and this session
did not run it. The implementation session measures on the same validation
dataset, using the `timing["metrics"]` accumulator already in place
(`pipeline.py:64`).

The work per condition is `n_sims × n_samples × n_params` distance evaluations
against one reference point per simulation, thresholded at `resolution` levels —
to be timed at the study's actual `n_posterior_samples` and condition count, and
reported as a fraction of `timing["inference"]`. Decision rule: if the joint
metric is a non-trivial fraction of inference time, it is computed on a
sub-sample of conditions, and that sub-sample size joins §D6's pinned schema.

---

## 3. What this does *not* decide

- **#75's floor treatment.** Whether `coverage_error` fixes `B` and `n_samples`
  at validation time or reports excess over the floor, and whether the floor
  correction holds empirically, remain #75's open questions. This contract
  carries `approximator`, which is all #75 needed from *this* issue.
- **`mode` for `coverage_error`** (#75 constraint 3). Exploitable at the package
  default; a metric-side decision, not a contract one.
- **The `bayesflow-irt` consumer change.** #82 constraint 2 is explicit that
  `make_irt_hooks`' closure would not reach TARP even through a fixed pipeline.
  A companion issue there is required and is out of this repository's scope.
- **PCM/GPCM reference coordinates** (#82 constraint 4) — see §D5.

## 4. Suggested order for the implementation session

1. `JointMetricInputs` / `register_joint_metric` / joint dispatch in
   `run_validation_pipeline`; `describe_metrics` covers both registries.
2. Refactor `make_lc2st_validate_fn` onto it and delete the duplicate loop.
   Register `lc2st`'s direction (`worst_raw=0.25`).
3. Measure the per-trial cost (§D7) on the L-C2ST path, which is the more
   expensive of the two — it fits an MLP per fold.
4. TARP: the metric, the reference-provider contract, the two-key mode split,
   the pinned schema, `tarp_error`'s direction entry.
5. #75's `coverage_error` on the same contract, behind
   `requires="calibration-loss"`.

Steps 1–2 are worth landing alone: they remove a duplicated pipeline and close a
live `worst_raw` defect, independently of whether TARP ever ships.

## References

Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023).
Sampling-based accuracy testing of posterior estimators for general inference.
See [`references.md`](../references.md); OpenAlex `W4319453761`. Algorithm 2
(the TARP estimator), Theorem 3 (positionable regions identify the posterior),
§3.1 (expected HPD coverage is blind to a data-independent estimator), §4.2
(robustness to the distance metric), §4.3 (an x-independent reference carries
the same blind spot).

Linhart, J., Gramfort, A., & Rodrigues, P. L. C. (2023). L-C2ST: Local
diagnostics for posterior approximations in simulation-based inference. See
[`references.md`](../references.md). Algorithms 1–2 and Theorem 3.1, as
implemented in `validation/c2st.py`.

Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N., Faltejsková, K.,
Gelman, A., & Vehtari, A. (2025). Simulation-based calibration checking for
Bayesian computation: The choice of test quantities shapes sensitivity.
*Bayesian Analysis, 20*(2), 461–488. See [`references.md`](../references.md).
Marginal rank statistics are blind to an estimator that ignores the data.

— Claude Code (Claude Opus 5)
