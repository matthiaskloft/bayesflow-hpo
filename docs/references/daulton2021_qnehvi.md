# Daulton et al. (2021) — qNEHVI

**Reference:** Daulton, S., Balandat, M., & Bakshy, E. (2021). Parallel Bayesian optimization of multiple noisy objectives with expected hypervolume improvement. In *Advances in Neural Information Processing Systems 34* (pp. 2187-2200). https://doi.org/10.48550/arXiv.2105.08195

**Relevance:** Extension of qEHVI for noisy objective functions. Relevant for bayesflow-hpo when validation metrics have observation noise (e.g., Monte Carlo variance in posterior samples).

---

## Key Contribution: qNEHVI for Noisy Multi-Objective Optimization

### Motivation

**Problem:** Standard qEHVI assumes noise-free observations, but many real-world problems have:
1. **Observation noise** in objective measurements
2. **Stochastic evaluation** of objectives
3. **Unknown noise levels** that vary across objective space

**qNEHVI solution:** Bayesian treatment of hypervolume improvement that accounts for observation noise

---

## Methodology: Bayesian Treatment of HV

### Key Innovation

**Page reference:** Section 3, Equations 3-5

Instead of treating observed values as ground truth, qNEHVI:
1. **Models noise** explicitly in GP posterior
2. **Integrates over posterior** of objective values
3. **Computes expected HV** under posterior predictive distribution

### Algorithm Outline

**Page reference:** Section 4, Algorithm 1

```
1. Initialize with n_init random points
2. While budget not exhausted:
   a. Draw posterior samples of objective functions
   b. Compute expected HV under posterior
   c. Optimize qNEHVI acquisition → X*
   d. Evaluate f(X*) (observe noisy outcomes)
   e. Update GP with noisy observations
```

---

## Noise Modeling

### Gaussian Observation Noise

**Assumption:** Observations y = f(x) + ε, where ε ~ N(0, σ²_noise)

**GP posterior:** Marginalizes over both function values and noise

**Noise estimation:**
- **Known noise:** σ² fixed if known from domain knowledge
- **Unknown noise:** Learn σ² via Type-2 MLE (marginal likelihood maximization)

**Page reference:** Section 3.1

---

## Acquisition Function: qNEHVI

### Expected Hypervolume Improvement

**Definition:**
```
qNEHVI(X) = E_{f(x)|D}[HV(F ∪ {f(X)}, F_ref) - HV(F, F_ref)]
```

where:
- F is current Pareto front
- F_ref is reference point
- f(x)|D is posterior predictive distribution

**Monte Carlo estimation:**
```
qNEHVI(X) ≈ (1/S) Σ_s HV(F ∪ {f^(s)(X)}, F_ref)
```

**Page reference:** Section 3.2, Equation 7

### Differentiability

**Reparameterization trick:** Enables gradient-based optimization of qNEHVI

**Gradient computation:** Standard backpropagation through GP posterior samples

---

## Empirical Results

**Benchmarks:**
- Synthetic test functions (DTLZ, ZDT)
- Real-world hyperparameter optimization with noisy evaluations

**Key findings (Section 5):**
- qNEHVI outperforms qEHVI under observation noise
- Robust to heteroscedastic noise (noise varies across input space)
- Better hypervolume convergence than SMS-EGO+ and other baselines

**Page reference:** Section 5, Figures 2-4

---

## Comparison: qNEHVI vs qEHVI

| Aspect | qEHVI | qNEHVI |
|--------|-------|--------|
| **Noise assumption** | Noise-free | Noisy observations |
| **HV computation** | Deterministic | Expected (MC) |
| **Use case** | Deterministic objectives | Stochastic/noisy objectives |
| **Computational cost** | Lower | Higher (due to MC) |

---

## Our Implementation

**Note:** bayesflow-hpo currently uses qEHVI (Daulton et al., 2020) for the `"botorch"` sampler preset.

**When to use qNEHVI:**
- Validation metrics have high Monte Carlo variance
- Training is stochastic (e.g., random initialization, data shuffling)
- Limited validation data leads to noisy metric estimates

**Future implementation:**
```python
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement

# In optimization/study.py:
if acquisition == "qnehvi":
    acquisition_fn = qNoisyExpectedHypervolumeImprovement(...)
```

---

## Practical Considerations

### Noise Estimation

**Known noise:** If validation variance is known (e.g., from MC sampling), set GP noise accordingly

**Unknown noise:** Let qNEHVI learn noise from data (default behavior)

### Reference Point Selection

**Challenge:** Noisy observations make reference point selection difficult

**Recommendation:** Set reference point to pessimistic values accounting for noise

**Page reference:** Section 4.2

### Computational Cost

**MC samples:** qNEHVI requires more MC samples for stable estimates (typically 256 vs 128 for qEHVI)

**GP inference:** Same as qEHVI (GPyTorch backend)

---

## Edge Cases and Limitations

**High noise:** When σ² >> signal, qNEHVI struggles to identify improvements
- Remedy: Increase sample size for validation metrics

**Heteroscedastic noise:** Noise variance varies across input space
- qNEHVI handles this naturally via GP

**Many objectives (M > 5):** HV computation becomes expensive
- Remedy: Use hypervolume approximations or R2 indicator

---

## Intentional Deviations

**Current implementation:** bayesflow-hpo uses qEHVI (noise-free assumption)

**Rationale:** Most validation metrics in bayesflow-hpo are:
- Based on large validation sets (low Monte Carlo variance)
- Averaged across multiple conditions
- Deterministic (e.g., model architecture metrics)

**Future work:** Add qNEHVI option for stochastic training scenarios

---

## Related References

- **qEHVI (noise-free):** Daulton et al. (2020) — See `daulton2020_qehvi.md`
- **BoTorch framework:** Balandat et al. (2020) — See `balandat2020_botorch.md`
- **NSGA-II comparison:** Deb et al. (2002) — See `deb2002_nsga2.md`

---

## Implementation Notes for bayesflow-hpo

**Scenarios where qNEHVI would help:**
1. **Small validation datasets:** High variance in SBC/C2ST metrics
2. **Stochastic training:** Non-deterministic posterior samples
3. **Early stopping:** Validation metrics from intermediate checkpoints (high variance)

**Integration point:** `optimization/study.py:create_study()` sampler configuration

---

## Open Challenges

**Adaptive noise estimation:** Learning noise levels during optimization

**Multi-fidelity qNEHVI:** Combining noise modeling with multi-fidelity optimization

**Scalability:** qNEHVI for many objectives (M > 10)
