# Sobol' (1967) — Sobol Sequences

**Reference:** Sobol', I. M. (1967). On the distribution of points in a cube and the approximate evaluation of integrals. *USSR Computational Mathematics and Mathematical Physics*, *7*(4), 86-112. https://doi.org/10.1016/0041-5553(67)90144-9

**Relevance:** Backs `optimization/study.py` (QMC warm-up feature) and `qmc_startup_trials` parameter.

---

## Key Contribution: Low-Discrepancy Sequences

### Motivation

**Problem:** Numerical integration in high-dimensional spaces requires efficient point distributions

**Existing methods:** Random Monte Carlo (slow convergence O(n^(-1/2)))

**Sobol innovation:** Quasi-random sequences with much faster convergence O(n^(-1) log^d(n))

**Page reference:** Section 1, p. 86-88

---

## Sobol Sequences: Construction Algorithm

### Mathematical Definition

**Page reference:** Section 2, p. 89-92

**Core idea:** Use direction numbers to construct points uniformly distributed in [0,1]^d

**Algorithm outline:**
1. Choose primitive polynomials over GF(2)
2. Compute direction numbers v_ij for each dimension i and bit position j
3. Generate points using Gray code: x_n = n ⊕ (n >> 1)
4. Apply direction numbers to obtain Sobol points

**Pseudocode:**
```
For dimension i = 1,...,d:
    For point n = 1,...,N:
        x_n[i] = Σ_j v_ij * ((n << j) mod 2) / 2^j
```

---

## Key Properties

### Low Discrepancy

**Page reference:** Section 3, p. 93-94

**Definition:** Discrepancy D measures deviation from uniform distribution

**Property:** Sobol sequences achieve minimal discrepancy among all sequences

**Convergence rate:** O(N^(-1) log^d(N)) vs O(N^(-1/2)) for random

### Power-of-2 Property

**Page reference:** Section 4, p. 95

**Optimal discrepancy at 2^m points:** Sobol sequences achieve optimal discrepancy when N = 2^m

**Practical implication:** Round sample sizes to nearest power of 2 for best results

**Our implementation:** Warning in `optimization/study.py` when N is not power of 2

---

## Direction Numbers

### Original Sobol Direction Numbers

**Page reference:** Section 2, p. 89-91

**Primitive polynomials:** Used to generate direction numbers for each dimension

**Initial values:** v_1j = 2^(-j) (first dimension)

**Recurrence:** v_ij = m_k ⊕ v_{i-k,j-k} / 2^j (for higher dimensions)

### Joe & Kuo (2008) Improvement

**Page reference:** See `joe2008_sobol.md`

**Enhancement:** Better direction numbers for improved 2D projections

**Our implementation:** SciPy's Sobol generator uses Joe & Kuo direction numbers by default

---

## Numerical Integration

### Integration Error Bounds

**Page reference:** Section 5, p. 96-99

**Koksma-Hlawka inequality:**
```
|E[f]| ≤ V(f) · D_N
```
where:
- E[f] is integration error
- V(f) is variation of f in Hardy-Krause sense
- D_N is discrepancy of point set

**Consequence:** Error bound independent of dimensionality (for certain function classes)

---

## Our Implementation

**Feature:** `qmc_startup_trials` parameter in `create_study()` and `optimize()`

**Implementation:**
```python
from scipy.stats import qmc

# QMC warm-up sampler
sampler = qmc.Sobol(d=dim, scramble=True)
points = sampler.random(n=n_startup)
```

**Usage:**
```python
study = optuna.create_study(
    sampler="tpe",
    qmc_startup_trials=10  # First 10 trials use Sobol
)
```

**Class:** `QMCWarmupSampler` in `optimization/study.py`

**Behavior:**
1. Uses Sobol sequence for first N non-rejected trials
2. Transparently switches to main sampler after warm-up
3. Compatible with all sampler presets
4. Issues warning if N not power of 2

---

## Practical Considerations

### Sample Size

**Recommendation:** Use power-of-2 sample sizes (1, 2, 4, 8, 16, 32, 64, 128...)

**Page reference:** Section 4, p. 95 (optimal discrepancy at 2^m)

**Our implementation:** Warning for non-power-of-2 values

### Scrambling

**Page reference:** Not in original paper (later development)

**Purpose:** Randomize Sobol sequences to reduce correlation artifacts

**Our implementation:** Uses `scramble=True` in SciPy's Sobol generator

### Dimensionality

**Effective dimensionality:** Sobol sequences work well for problems with low effective dimensionality (few important dimensions)

**Curse of dimensionality:** Performance degrades for very high dimensions (d > 100)

---

## Comparison to Other Methods

| Method | Convergence | Dimensionality | Implementation |
|--------|-------------|----------------|----------------|
| **Random MC** | O(n^(-1/2)) | Any | Simple |
| **Sobol** | O(n^(-1) log^d(n)) | Low effective d | Requires direction numbers |
| **Halton** | O(n^(-1) log^d(n)) | Any | Correlated dimensions |
| **Hammersley** | O(n^(-1) log^d(n)) | Fixed n | Requires n known in advance |

---

## Edge Cases and Limitations

**High dimensions:** Sobol sequences degrade for d > 50 without careful direction number selection

**Correlation:** Early points in sequence can be highly correlated (mitigated by scrambling)

**Implementation complexity:** Requires careful handling of direction numbers and bit operations

---

## Intentional Deviations

None. bayesflow-hpo uses Sobol sequences as specified:
- Original Sobol construction algorithm
- Joe & Kuo (2008) direction numbers (via SciPy)
- Scrambling for decorrelation
- Power-of-2 warning for optimal performance

**Integration:** Via `QMCWarmupSampler` wrapping Optuna's samplers

---

## Related References

- **Joe & Kuo (2008):** Improved direction numbers — See `joe2008_sobol.md`
- **SciPy documentation:** Sobol sequence implementation
- **QMC warm-up feature:** `optimization/study.py` QMCWarmupSampler class

---

## Key Theoretical Results

**Theorem 1 (p. 93):** Existence of sequences with discrepancy D_N = O(N^(-1) log^(d-1)(N))

**Theorem 2 (p. 94):** Sobol sequences achieve this optimal convergence rate

**Practical implication:** QMC warm-up provides better initial exploration than random sampling for most HPO problems

---

## Historical Context

**Significance:** Sobol sequences were one of the first quasi-Monte Carlo methods

**Impact:** Foundation for many QMC methods (Halton, Hammersley, Faure, etc.)

**Applications:** Numerical integration, global optimization, uncertainty quantification

**In HPO:** QMC warm-up improves initial trial quality for Bayesian optimization
