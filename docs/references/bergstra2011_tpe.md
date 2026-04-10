# Bergstra et al. (2011) — Tree-Structured Parzen Estimator (TPE)

**Reference:** Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. In *Advances in Neural Information Processing Systems 24* (pp. 2546-2554).

**Relevance:** Backs `optimization/study.py` (TPE sampler preset) and sequential model-based optimization approach.

---

## Key Method: TPE Algorithm

### Core Innovation (Line numbers not available - extracted from sibling repo index)

TPE models P(x|y) and P(y) instead of P(y|x) used in typical Bayesian optimization:
- **p(x|l)**: Distribution of hyperparameters for "good" observations (below γ-quantile)
- **p(x|g)**: Distribution of hyperparameters for "bad" observations (above γ-quantile)

### Algorithm Outline

1. **Split observations** by quantile γ (typically γ = 0.25):
   - l(x) = p(x | y < y*) where y* is γ-quantile of observed losses
   - g(x) = p(x | y ≥ y*)

2. **Model** each distribution using Parzen window estimators (kernel density estimates)

3. **Sample** candidate hyperparameters from l(x) (good distribution)

4. **Select** candidate maximizing expected improvement:
   ```
   EI(x) = (y* - y) · p(x|l) / p(x|g)
   ```

### Tree-Structured Handling

TPE naturally handles conditional hyperparameters through tree structure:
- Parent nodes determine which child branches are valid
- Each node has its own l(x) and g(x) distributions
- Sampling follows tree structure automatically

**Our implementation:** `NetworkSelectionSpace` and `SummarySelectionSpace` provide tree-structured conditional parameters compatible with TPE sampling.

---

## Empirical Results

**Page reference:** NIPS 2011 proceedings

TPE demonstrated significant improvement over random search:
- More efficient exploration of hyperparameter space
- Better convergence in fewer trials
- Handles high-dimensional spaces better than standard Bayesian optimization

---

## Our Implementation

**Sampler preset:** `"tpe"` in `create_study()` and `optimize()`

**Implementation details:**
- Uses Optuna's `TPESampler` with default parameters
- Multivariate=True for handling conditional parameters
- n_startup_trials defaults: determined by `_resolve_n_startup_trials()` (checks population_size for multi-objective, fallback 10)

**Key configurations:**
- `multivariate=True`: Enables tree-structured handling
- `constant_liar=True`: For parallel trials (suggests placeholder values for running trials)

---

## Edge Cases and Recommendations

**High-dimensional spaces:** TPE scales better than Gaussian Process-based methods for >20 dimensions

**Conditional parameters:** TPE handles naturally via tree structure (no need for manual encoding)

**Categorical parameters:** Supported via one-hot encoding in Parzen estimator

**Early stopping:** TPE benefits from intermediate validation (pruning) to avoid full evaluation of poor configurations

---

## Intentional Deviations

None. bayesflow-hpo uses TPE as specified:
- Standard TPESampler from Optuna
- Default quantile γ = 0.25
- Multivariate handling enabled
- Compatible with conditional search spaces

**Note:** Fulltext extraction failed in sibling repo. This summary based on paper content and Optuna documentation.

---

## Related References

- **Optuna integration:** See `akiba2019_optuna.md`
- **Multi-objective TPE:** See `ozaki2022_mo_tpe` (not in current references, future work)
- **TPE theory:** Original Bergstra & Bengio (2012) JMLR paper provides extended theoretical analysis
