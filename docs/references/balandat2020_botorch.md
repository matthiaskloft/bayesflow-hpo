# Balandat et al. (2020) — BoTorch

**Reference:** Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson, A. G., & Bakshy, E. (2020). BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. In *Advances in Neural Information Processing Systems 33* (pp. 21524-21538). https://doi.org/10.48550/arXiv.1910.06403

**Relevance:** Backs `optimization/study.py` (BoTorch sampler preset) for gradient-based Bayesian optimization with Monte Carlo acquisition functions.

---

## Key Contribution: Modular BO Framework

### Motivation

**Problem:** Existing BO frameworks lack:
1. **Modularity** for easy experimentation with acquisition functions
2. **Performance** for gradient-based optimization
3. **Parallelization** support for batch optimization

**BoTorch solution:** PyTorch-based BO framework with:
- Differentiable acquisition functions via auto-differentiation
- Efficient Monte Carlo (MC) acquisition functions
- Seamless integration with PyTorch ecosystem

---

## Architecture Design

### Core Components

**1. Models (Posterior distributions)**
- Gaussian Process (GP) models via GPyTorch
- Multi-output GPs for multi-fidelity optimization
- Support for custom probabilistic models

**2. Acquisition Functions**
- MC acquisition functions (qEI, qUCB, qEHVI)
- Analytic acquisition functions (EI, UCB)
- Modular design for easy custom acquisition functions

**3. Optimizers**
- Gradient-based optimization via torch.optim
- Multiple restarts for global optimization
- Candidate generation strategies

**Page reference:** Section 2, Figure 1

---

## Monte Carlo Acquisition Functions

### q-Acquisition Functions

**Batch acquisition:** Suggest q points simultaneously for parallel optimization

**MC estimation:**
```
α(X) = (1/S) Σ_s α(f^(s)(X))
```
where f^(s) are posterior samples, α is base acquisition function

**Benefits:**
- **Differentiability:** Reparameterization trick enables gradient computation
- **Flexibility:** Works with any differentiable α
- **Parallelism:** Naturally extends to q > 1

**Page reference:** Section 3

---

## Performance Optimization

### 1. Fast Predictive Distributions

**GPyTorch integration:** Efficient GP inference with:
- LOVE (Linearly Ofset-time Exact GP inference) for fast predictions
- Cache-friendly computation
- GPU acceleration

**Page reference:** Section 4.1

### 2. Acquisition Optimization

**Strategy:**
- Sequential candidate generation
- Multiple restarts with random initialization
- L-BFGS-B or Adam optimizer
- Warm-start from previous iterations

**Page reference:** Section 4.2

### 3. Distributed Optimization

**Parallel acquisition evaluation:**
- Candidates generated in parallel
- Asynchronous evaluation support
- Fault tolerance for distributed settings

---

## Our Implementation

**Sampler preset:** `"botorch"` in `create_study()` and `optimize()`

**Configuration:**
```python
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, RBFKernel
```

**Default settings:**
- **Model:** Single-task GP with Matérn kernel
- **Acquisition:** qEI (q-Expected Improvement)
- **Optimizer:** L-BFGS-B with 20 restarts
- **q (batch size):** 1 (single-point for simplicity)
- **n_startup_trials:** 10 (determined by `_resolve_n_startup_trials()`)

**Lazy import:** BoTorch imported only when `"botorch"` sampler selected

**Categorical parameters:** One-hot encoding handled by Optuna

**Constraints:** Supports inequality constraints via `constraints_func` parameter

---

## Comparison to Other Frameworks

| Framework | Language | Acquisition | Gradient-based |
|-----------|----------|--------------|-----------------|
| **BoTorch** | Python | MC (qEI, qEHVI) | ✅ Yes |
| GPyOpt | Python | Analytic | ❌ No |
| Spearmint | Python | EI | ❌ No |
| RoBO | Python | EI, UCB | ❌ No |
| Dragonfly | Python | Various | ✅ Partial |

**Page reference:** Section 5, Table 1

---

## Key Algorithms

### Algorithm 1: Bayesian Optimization Loop

**Page reference:** Section 2

```
1. Initialize with n_init random points
2. While budget not exhausted:
   a. Fit GP model to observed data
   b. Optimize acquisition function → X*
   c. Evaluate f(X*)
   d. Update GP model
```

### Algorithm 2: MC Acquisition Optimization

**Page reference:** Section 4.2

```
Input: Acquisition α, model f, q candidates
Output: Candidate points X

1. Generate n_restarts random initializations
2. For each restart:
   a. Optimize α via gradient ascent
   b. Keep best candidate
3. Return best candidate across restarts
```

---

## Edge Cases and Limitations

**High-dimensional X:** GP scalability degrades for d > 20
- Remedy: Use additive GPs or dimensionality reduction

**Large q:** MC acquisition becomes expensive
- Remedy: Use quasi-MC methods or sequential optimization

**Non-Gaussian likelihoods:** Standard GP assumes Gaussian noise
- Remedy: Use variational GPs or heteroscedastic models

**Observation noise:** qEI assumes noise-free observations
- Remedy: Use qNEHVI for noisy observations (Daulton et al., 2021)

---

## Intentional Deviations

None. bayesflow-hpo uses BoTorch as specified:
- Standard GP with Matérn kernel
- qEI acquisition for single-objective
- qEHVI acquisition for multi-objective (via daulton2020_qehvi)
- L-BFGS-B optimization with multiple restarts

**Key difference:** bayesflow-hpo specializes for BayesFlow NPE models with built-in:
- Validation pipeline
- Parameter/memory budget constraints
- Custom network architectures
- SBC and C2ST validation metrics

---

## Related References

- **qEHVI:** Daulton et al. (2020) — See `daulton2020_qehvi.md`
- **qNEHVI (noisy):** Daulton et al. (2021) — See `daulton2021_qnehvi.md`
- **GPyTorch:** Gardner et al. (2018) — GP inference backend
- **Optuna integration:** Akiba et al. (2019) — See `akiba2019_optuna.md`

---

## Future Enhancements

**Multi-fidelity BO:** Support for multi-fidelity optimization (e.g., different validation budgets)

**Transfer learning:** Warm-start GP from previous HPO runs

**High-dimensional BO:** Add support for additive GPs or REMBO

**Constrained optimization:** Better support for complex constraints
