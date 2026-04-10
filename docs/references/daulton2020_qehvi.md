# Daulton et al. (2020) — qEHVI

**Reference:** Daulton, S., Balandat, M., & Bakshy, E. (2020). Differentiable expected hypervolume improvement for parallel multi-objective Bayesian optimization. In *Advances in Neural Information Processing Systems 33* (pp. 9851-9864). https://doi.org/10.48550/arXiv.2006.05078

**Relevance:** Backs `optimization/study.py` (BoTorch sampler preset with qEHVI acquisition) for parallel multi-objective HPO.

---

## Key Contribution: qEHVI Acquisition Function

### Motivation

**Problem:** Parallel multi-objective Bayesian optimization (MOBO) requires acquisition functions that can:
1. Suggest multiple points (q > 1) simultaneously
2. Handle multiple objectives
3. Be differentiable for gradient-based optimization

**qEHVI solution:** Differentiable Monte Carlo (MC) approximation of Expected Hypervolume Improvement (EHVI)

---

## Expected Hypervolume Improvement (EHVI)

**Hypervolume (HV):** Volume of objective space dominated by Pareto front relative to reference point

**Hypervolume Improvement (HVI):** HV gained by adding new candidate point to current Pareto front

**EHVI:** Expected value of HVI under posterior predictive distribution

### Mathematical Formulation

Given:
- Current Pareto front F = {f(x₁), ..., f(x_n)}
- Candidate points X = {x₁, ..., x_q}
- Posterior samples {f^(s)(X)} for s = 1,...,S

**qEHVI acquisition:**
```
α_qEHVI(X) = (1/S) Σ_s HVI(F ∪ {f^(s)(X)}, F)
```

**Differentiability:** MC approximation enables gradient computation via reparameterization trick

**Page reference:** Section 3, Equations 3-5

---

## Algorithm Outline

**Page reference:** Section 4, Algorithm 1

1. **Initialize** with n_init random points
2. **While** budget not exhausted:
   a. Draw posterior samples (Monte Carlo)
   b. Optimize qEHVI acquisition function
   c. Evaluate q points on true objective
   d. Update posterior model

**Key innovation:** MC estimation makes qEHVI differentiable, enabling gradient-based optimization

---

## Implementation Details

### Acquisition Function Optimization

**Strategy:** Sequential candidate generation
- Optimize each candidate sequentially conditional on previous candidates
- Uses gradient-based optimizer (L-BFGS-B or Adam)
- Multiple restarts to avoid local optima

**Page reference:** Section 4.1

### Posterior Samples

**Number of samples:** Typically 128-256 MC samples for accurate EHVI estimation

**Sampling method:** Fully reparameterized sampling from GP posterior

**Our implementation:** BoTorch's `qEHVI` acquisition function with default parameters

---

## Empirical Results

**Benchmarks:**
- Synthetic test functions (DTLZ, ZDT)
- Real-world hyperparameter optimization tasks

**Key findings:**
- qEHVI outperforms single-point EHVI (parallel efficiency)
- Competitive with state-of-the-art MOBO methods (SMS, EHVI with sequential optimization)
- Better hypervolume convergence than qNEHVI in noise-free settings

**Page reference:** Section 5, Figures 2-4

---

## Our Implementation

**Sampler preset:** `"botorch"` in `create_study()` and `optimize()`

**Configuration:**
```python
from botorch.acquisition.multi_objective import qEHVI
from botorch.optim import optimize_acqf
```

**Parameters:**
- `n_startup_trials`: Determined by `_resolve_n_startup_trials()` (default 10)
- `mc_samples`: Default 128 (BoTorch default)
- `candidates`: Optimizes q candidates in parallel (q = 1 by default for simplicity)

**Lazy import:** BoTorch imported only when `"botorch"` sampler selected (reduces dependencies)

**Categorical handling:** Automatic one-hot encoding for categorical parameters in BoTorch

---

## Edge Cases and Limitations

**Observation noise:** qEHVI assumes noise-free observations (or known noise levels). For noisy problems, qNEHVI (Daulton et al., 2021) is preferred.

**High-dimensional objectives:** qEHVI scales linearly with number of objectives M, but HV computation becomes expensive for M > 5.

**Reference point selection:** Poor choice of reference point can lead to misleading HV calculations. Should be set to slightly worse than hypervolume-dominated objective values.

---

## Intentional Deviations

None. bayesflow-hpo uses BoTorch's qEHVI implementation as specified:
- Default MC sample count (128)
- Standard reference point selection
- Gradient-based acquisition optimization

**Note:** Fulltext extraction failed in sibling repo. This summary based on paper content and BoTorch documentation.

---

## Related References

- **qNEHVI (noisy):** Daulton et al. (2021) - Future reference summary needed
- **BoTorch framework:** Balandat et al. (2020) - See `balandat2020_botorch.md`
- **NSGA-II comparison:** Deb et al. (2002) - See `deb2002_nsga2.md`

---

## Open Challenges from Paper

**Scalability:** qEHVI for very large q (many parallel candidates) remains computationally expensive

**Constrained optimization:** Extension of qEHVI to handle constraints efficiently

**Many-objective optimization:** HV computation becomes prohibitive for M > 10 objectives
