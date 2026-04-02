# Plan: C2ST Metrics (Package E)

**Created**: 2026-04-02
**Author**: Claude Code + Matze

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Spec | DONE | 2026-04-02 | Papers read from LaTeX source |
| Plan | DONE | 2026-04-02 | |
| Phase 1: L-C2ST + global C2ST standalone functions | PENDING | | |
| Phase 2: ValidateFn factory + exports + tests | PENDING | | |
| Ship | PENDING | | |

## Spec

_Design decisions and requirements — the "what and why"._

### Summary

**Motivation**: The existing 13 validation metrics all operate
per-parameter on 1D marginals. They cannot detect multivariate
posterior misspecification (e.g., wrong correlations between parameters
with individually correct marginals). C2ST-based metrics provide
multivariate posterior diagnostics grounded in classifier two-sample
testing theory.

**Outcome**: Users can call `lc2st(posterior_samples, true_params,
observations)` for reference-free multivariate posterior validation, or
pass `validate_fn=make_lc2st_validate_fn()` to `optimize()` to include
L-C2ST as an HPO metric. A separate `global_c2st()` function is
available for settings where reference posterior samples exist.

### Requirements

- R1: `lc2st()` implements L-C2ST (Linhart et al., 2023) as a
  standalone function. Input: `posterior_samples (n_sims, n_samples,
  n_params)`, `true_params (n_sims, n_params)`, `observations (n_sims,
  n_obs)`. Returns `LC2STResult` dataclass.
- R2: `global_c2st()` implements standard C2ST (López-Paz & Oquab,
  2017) as a standalone function. Input: `samples_p (n, d)`,
  `samples_q (n, d)`. Returns `GlobalC2STResult` dataclass.
- R3: `make_lc2st_validate_fn()` returns a `ValidateFn` compatible
  with `optimize(validate_fn=...)`. Runs inference once per condition,
  computes both standard metrics and L-C2ST from the same draws.
- R4: `scikit-learn` is an optional dependency. Functions raise
  `ImportError` with install instructions when sklearn is missing.
  Install via `pip install bayesflow-hpo[sklearn]`.
- R5: L-C2ST is NOT registered as a standard `MetricFn` — the
  per-parameter 2D signature is fundamentally incompatible with its
  multivariate data needs.
- R6: Global C2ST is purely post-hoc (not usable as HPO objective —
  requires reference posterior per trial).
- R7: All implementations must match the algorithms described in the
  original papers (verified from LaTeX source in `docs/references/`).

### Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Integration approach | (A) Force into MetricFn, (B) New multivariate metric concept, (C) Standalone + ValidateFn factory | (C) Standalone + factory | MetricFn is per-parameter 2D; L-C2ST needs all params + observations jointly. ValidateFn already receives full ValidationDataset. No pipeline refactor needed. |
| Registry entries | (1) Placeholder that raises, (2) No registration | No registration | A metric that raises on call is confusing UX. Functions are discoverable via imports and docs. |
| Inference in factory | (1) Call run_validation_pipeline() then re-infer for L-C2ST, (2) Single inference pass for both | Single pass | Avoids double inference cost. Factory calls `make_bayesflow_infer_fn()` once, computes standard metrics and L-C2ST from same draws. |
| Null distribution in factory | (1) Always compute, (2) Skip by default | Skip by default (`n_null_trials=0`) | Permutation test is expensive (re-trains classifier N times). During HPO, only the statistic is needed for ranking. Users run full test post-hoc. |
| L-C2ST classifier config | (1) Paper's original (alpha=0, max_iter=25000), (2) SBIBM config (10*ndim layers, early_stopping) | SBIBM config | Matches reference implementation (JuliaLinhart/lc2st `sbibm_clf_kwargs`). Early stopping prevents overfitting on small calibration sets. |
| CV folds | (1) 10-fold (reference impl), (2) 5-fold | 5-fold default | Faster for HPO context. Configurable via parameter. |
| File location | (1) In registry.py, (2) New c2st.py | New `validation/c2st.py` | Follows pattern of separate modules per concern (sbc_tests.py, inference.py, etc.) |

### Scope

#### In Scope

- `lc2st()` standalone function with `LC2STResult`
- `global_c2st()` standalone function with `GlobalC2STResult`
- `make_lc2st_validate_fn()` factory
- `sklearn` optional dependency in pyproject.toml
- Tests for all three + import guard
- Documentation updates (references.md, TODO.md)
- Public API exports in `__init__.py`

#### Out of Scope

- L-C2ST-NF variant (normalizing flow latent-space test; Linhart et
  al. 2023, Section 3.1) — requires access to flow inverse transform,
  not available via BayesFlow's public API
- PP-plot graphical diagnostics (paper Algorithm 3) — can be added
  later as a visualization function
- Using L-C2ST as an Optuna objective metric for pruning — too
  expensive for intermediate validation steps at scale
- C2ST-KNN variant (López-Paz & Oquab 2017) — MLP is sufficient

### Architecture Overview

```
# Post-hoc diagnostic (standalone)
from bayesflow_hpo import lc2st, global_c2st

result = lc2st(posterior_samples, true_params, observations)
# result.statistic, result.p_value, result.per_observation_stats

# HPO integration (via validate_fn)
from bayesflow_hpo import optimize, make_lc2st_validate_fn

study = optimize(
    ...,
    validate_fn=make_lc2st_validate_fn(
        base_metrics=["calibration_error", "nrmse"],
    ),
    objective_metrics=["calibration_error", "nrmse", "lc2st"],
)
```

```
make_lc2st_validate_fn(base_metrics=["calibration_error", "nrmse"])
    │
    ▼
returned closure(approximator, validation_data, n_posterior_samples)
    │
    ├─ make_bayesflow_infer_fn()     ← reuses validation/inference.py
    │
    ├─ for each condition:
    │   ├─ infer_fn(sim_batch, n_posterior_samples) → draws (3D)
    │   ├─ compute_condition_metrics(draws_2d, true_values, ...)
    │   │                              ← per-parameter standard metrics
    │   ├─ assemble true_params, observations from sim_batch
    │   └─ lc2st(draws, true_params, observations, n_null_trials=0)
    │
    ├─ aggregate_condition_rows()    ← reuses validation/metrics.py
    ├─ average lc2st statistic across conditions
    │
    └─ return {**standard_summary, "lc2st": avg_statistic}
```

### Constraints

- Must work with sklearn >= 1.3 (KFold, MLPClassifier stable API)
- Must not import sklearn at module level — lazy import only
- Must not break existing installs that lack sklearn
- `ValidateFn` is called by `PeriodicValidationCallback` during
  intermediate pruning — factory must be fast enough with
  `n_null_trials=0`
- All implementations must match paper algorithms (source-backed)

### Open Questions

_None — papers read in full, reference implementations reviewed._

## Implementation Plan

### Phase 1: L-C2ST + global C2ST standalone functions

**Files to create:**
- `src/bayesflow_hpo/validation/c2st.py`

**Steps:**

1. **Add `_require_sklearn()` import guard** — lazy import of
   `MLPClassifier` and `KFold` from sklearn. Raises `ImportError` with
   install instructions if missing.

2. **Add `_default_clf_kwargs(ndim: int)`** — returns SBIBM-style MLP
   config matching reference implementation:
   `hidden_layer_sizes=(10*ndim, 10*ndim)`, `activation="relu"`,
   `solver="adam"`, `max_iter=25000`, `alpha=0`,
   `early_stopping=True`, `n_iter_no_change=50`.

3. **Add `LC2STResult` dataclass** (frozen):
   - `statistic: float` — mean MSE_0 across observations
   - `p_value: float | None` — permutation p-value (None if skipped)
   - `null_statistics: np.ndarray` — shape `(n_null_trials,)` or empty
   - `per_observation_stats: np.ndarray` — shape `(n_eval,)` per-obs

4. **Implement `lc2st()`** following Linhart et al. 2023, Algorithms 1-2:
   - **Input validation**: check shapes, require n_sims > n_folds
   - **Training data construction** (paper eq. 299): for each sim n,
     class 1 = `concat(true_params[n], observations[n])` (joint),
     class 0 = `concat(posterior_samples[n, 0, :], observations[n])`
     (approximate). One posterior sample per sim → balanced classes.
   - **K-fold cross-validation**: for each fold, train
     `MLPClassifier(**clf_kwargs, random_state=fold_idx)` on fold-train,
     predict `predict_proba(·)[:, 1]` on fold-val class-0 samples only
   - **Single-class MSE_0** (paper Theorem 3.1, eq. 310):
     `per_obs = (d_n - 0.5)²`, `statistic = mean(per_obs)`
   - **Null distribution** (paper Algorithm 1, lines 10-14): for each
     of `n_null_trials`, permute labels → re-run CV → compute MSE_0.
     `p_value = mean(null_stats >= statistic)`. Skip if
     `n_null_trials == 0`.

5. **Add `GlobalC2STResult` dataclass** (frozen):
   - `accuracy: float`, `p_value: float`, `n_test: int`

6. **Implement `global_c2st()`** following López-Paz & Oquab 2017:
   - Pool samples, shuffle, 50-50 stratified train/test split
   - Train `MLPClassifier(hidden_layer_sizes=(20,), max_iter=100,
     solver="adam")`
   - `accuracy = clf.score(X_test, y_test)`
   - `p_value = 1 - norm.cdf((accuracy - 0.5) / sqrt(1/(4*n_test)))`
     (paper Theorem 1, null = `N(0.5, 1/(4*n_test))`)

### Phase 2: ValidateFn factory + exports + tests

**Files to modify:**
- `src/bayesflow_hpo/validation/c2st.py` — add factory
- `src/bayesflow_hpo/validation/__init__.py` — add exports
- `src/bayesflow_hpo/__init__.py` — add public API exports
- `pyproject.toml` — add `sklearn` optional dependency
- `docs/references.md` — update C2ST/L-C2ST entries
- `docs/TODO.md` — move Package E to Done

**Files to create:**
- `tests/test_validation/test_c2st.py`

**Steps:**

1. **Implement `make_lc2st_validate_fn()`** in `c2st.py`:
   - Parameters: `base_metrics`, `n_folds`, `n_null_trials` (default 0),
     `clf_kwargs`, `seed`
   - Returned closure:
     - Creates `infer_fn` via `make_bayesflow_infer_fn()` (reuse
       `validation/inference.py:20`)
     - Per condition: run inference once, compute standard per-parameter
       metrics via `compute_condition_metrics()` (reuse
       `validation/metrics.py:21`), assemble multivariate arrays, call
       `lc2st()`
     - Aggregate via `aggregate_condition_rows()` (reuse
       `validation/metrics.py:54`)
     - Return merged dict

2. **Add exports** to `validation/__init__.py`: `LC2STResult`,
   `GlobalC2STResult`, `lc2st`, `global_c2st`, `make_lc2st_validate_fn`

3. **Add exports** to top-level `__init__.py` — same symbols in imports
   and `__all__`

4. **Add `sklearn` extra** to `pyproject.toml`:
   `sklearn = ["scikit-learn>=1.3"]`

5. **Write tests** in `tests/test_validation/test_c2st.py`:

   **L-C2ST tests (6):**
   - `test_lc2st_detects_mismatch` — biased posterior → statistic >> 0,
     low p-value
   - `test_lc2st_consistent_posterior` — well-calibrated → statistic
     near 0
   - `test_lc2st_result_fields` — dataclass fields and array shapes
   - `test_lc2st_seed_reproducibility` — deterministic with same seed
   - `test_lc2st_single_param` — works with n_params=1
   - `test_lc2st_no_null_trials` — n_null_trials=0 → p_value=None

   **Global C2ST tests (4):**
   - `test_global_c2st_same_distribution` — accuracy ≈ 0.5, high p-value
   - `test_global_c2st_different_distribution` — accuracy >> 0.5, low
     p-value
   - `test_global_c2st_result_fields`
   - `test_global_c2st_seed_reproducibility`

   **Factory tests (3):**
   - `test_make_lc2st_validate_fn_returns_callable`
   - `test_lc2st_validate_fn_output_keys` — base metric keys + "lc2st"
   - `test_lc2st_validate_fn_single_param`

   **Import guard (1):**
   - `test_require_sklearn_error_message` — mock absent sklearn

6. **Update `docs/references.md`** — note implementations available for
   C2ST/L-C2ST entries

7. **Update `docs/TODO.md`** — move Package E to Done section

## Verification & Validation

- **Automated**: `pytest tests/test_validation/test_c2st.py -v`
  (new tests), `pytest tests/ -v` (no regressions),
  `ruff check src/ tests/` (lint clean)
- **Import check**: `python -c "from bayesflow_hpo import lc2st,
  global_c2st, make_lc2st_validate_fn"` succeeds with sklearn installed
- **Import guard**: verify helpful `ImportError` when sklearn is absent

## Dependencies

- `scikit-learn >= 1.3` (optional, via `pip install bayesflow-hpo[sklearn]`)
- `scipy.stats.norm` (already a core dependency) — for global C2ST p-value

## Notes

### References

- Linhart, J., Gramfort, A., & Rodrigues, P. L. C. (2023). L-C2ST:
  Local diagnostics for posterior approximations in simulation-based
  inference. *NeurIPS 2023*. LaTeX source: `docs/references/arXiv-2306.03580v2.tar.gz`
- López-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample
  tests. *ICLR 2017*. LaTeX source: `docs/references/arXiv-1610.06545v4.tar.gz`
- Reference implementation: https://github.com/JuliaLinhart/lc2st
  (reviewed `lc2st.py` and `c2st.py`)
- sbi package implementation: `sbi.diagnostics.lc2st` (reviewed for
  API patterns)

### Key equations (from paper LaTeX)

- **Joint-conditional equivalence** (Linhart eq. 303):
  `d*(θ,x) = p(θ|x) / (p(θ|x) + q(θ|x)) = d*_x(θ)`
- **Single-class MSE_0** (Linhart Theorem 3.1, eq. 310):
  `t̂_MSE₀(x_o) = (1/N_v) Σ (d(Θ^q_n, x_o) - 1/2)²`
- **C2ST null** (López-Paz Theorem 1):
  `t̂ ~ N(1/2, 1/(4·n_te))` under H₀
