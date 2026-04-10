# Joe & Kuo (2008) — Improved Sobol Direction Numbers

**Reference:** Joe, S., & Kuo, F. Y. (2008). Constructing Sobol sequences with better two-dimensional projections. *SIAM Journal on Scientific Computing*, *30*(5), 2635-2654. https://doi.org/10.1137/070709359

**Relevance:** Improves QMC warm-up quality via better direction numbers used in SciPy's Sobol generator.

---

## Key Contribution: Enhanced Direction Numbers

### Motivation

**Problem:** Original Sobol direction numbers (Sobol' 1967) produce poor 2D projections for some dimensions

**Impact:** Poor 2D projections lead to:
- Slower convergence
- Non-uniform coverage of 2D subspaces
- Suboptimal integration accuracy

**Joe & Kuo solution:** New direction numbers with excellent 2D projections

**Page reference:** Section 1, p. 2635-2637

---

## Background: Sobol Sequences

### Direction Numbers

**Page reference:** Section 2, p. 2638-2640

**Definition:** Direction numbers v_ij control how Sobol points are distributed in each dimension i at bit position j

**Requirements:**
1. **Primitive polynomial condition:** Ensures uniformity
2. **Initialization condition:** Proper starting values
3. **Recurrence condition:** Generates valid direction numbers

**Original Sobol direction numbers:** Simple but produce poor 2D projections for many dimension pairs

---

## New Direction Numbers

### Selection Criteria

**Page reference:** Section 3, p. 2641-2643

**Goal:** Minimize worst-case 2D projection discrepancy

**Method:** Exhaustive search over primitive polynomials and initial values

**Criteria:**
1. **Primitive polynomials** with good equidistribution properties
2. **Initial values** that optimize 2D projections
3. **Recurrence coefficients** that preserve quality

**Algorithm:**
```
For each dimension i:
    1. Enumerate primitive polynomials of degree di
    2. For each polynomial, test all initial values
    3. Select combination minimizing 2D discrepancy
    4. Verify recurrence condition holds
```

---

## Main Results

### Up to Dimension 21201

**Page reference:** Section 4, p. 2644-2646

**Achievement:** Generated direction numbers for dimensions 1-21201

**Quality metrics:**
- **t-value:** Measures uniformity of 2D projections
- **Maximum t-value:** Minimized across all dimension pairs

**Comparison:** New direction numbers achieve much lower t-values than original Sobol

### Tables of Direction Numbers

**Page reference:** Appendix, p. 2650-2653

**Format:** Tables listing:
- Primitive polynomials
- Initial direction numbers (v_1, v_2, ...)
- Recurrence coefficients

**Usage:** Direct plug-in replacement for Sobol's original direction numbers

---

## Quality Assessment

### t-Value Metric

**Page reference:** Section 5, p. 2647-2649

**Definition:** t-value measures deviation from uniformity in 2D projections

**Computation:**
```
t = max_{i<j} D_{ij}(N)
```
where D_{ij} is 2D discrepancy for dimensions i and j

**Good direction numbers:** Have low t-values (uniform 2D projections)

### Empirical Results

**Page reference:** Section 6, p. 2649-2651

**Test function:** High-dimensional integration

**Findings:**
- Joe & Kuo direction numbers converge faster than original Sobol
- Improvement especially pronounced for dimensions 50-1000
- Error reduction factors of 2-10x for many test functions

---

## Our Implementation

**Usage via SciPy:**

```python
from scipy.stats import qmc

# Joe & Kuo direction numbers are default in SciPy
sampler = qmc.Sobol(d=dim, scramble=True)
points = sampler.random(n=N)
```

**Backend:** SciPy's Sobol generator uses Joe & Kuo direction numbers internally

**Verification:** Can verify by checking 2D projections of generated points

---

## Comparison to Original Sobol

| Aspect | Original Sobol (1967) | Joe & Kuo (2008) |
|--------|----------------------|------------------|
| **2D projections** | Poor for many dims | Excellent |
| **Convergence** | O(n^(-1) log^d(n)) | Same asymptotically |
| **Practical error** | Higher constant factor | Lower constant factor |
| **Max dimension** | Limited by quality | Up to 21,201 |

---

## Practical Recommendations

### When to Use Joe & Kuo Direction Numbers

**Default choice:** Always prefer Joe & Kuo over original Sobol

**Exception:** Original Sobol may be sufficient for very low dimensions (d < 10)

**Implementation:** Most modern libraries (SciPy, GSL, Boost) use Joe & Kuo by default

---

## Integration with QMC Warm-up

**Feature:** `qmc_startup_trials` in bayesflow-hpo

**Benefit:** Better 2D coverage improves initial exploration quality

**Our implementation:**
- SciPy's Sobol uses Joe & Kuo direction numbers by default
- Scrambling enabled for additional randomization
- Power-of-2 sample size warning still applies

---

## Intentional Deviations

None. bayesflow-hpo uses Joe & Kuo direction numbers via SciPy:
- Default in `scipy.stats.qmc.Sobol`
- No changes or modifications needed
- Automatically provides better 2D projections

---

## Related References

- **Sobol (1967):** Original Sobol sequences — See `sobol1967_qmc.md`
- **SciPy documentation:** Direction number tables
- **QMC literature:** Bratley et al. (1992), Niederreiter (1992)

---

## Key Technical Details

### Primitive Polynomials

**Page reference:** Section 2.1, p. 2638

**Definition:** Polynomials over GF(2) that cannot be factored

**Role:** Determine recurrence structure for direction numbers

**Selection:** Chosen to maximize equidistribution properties

### Initial Values

**Page reference:** Section 2.2, p. 2639

**Definition:** Starting values v_ij for j = 1,...,m_i

**Optimization:** Chosen to minimize 2D discrepancy

### Recurrence Condition

**Page reference:** Section 2.3, p. 2640

**Formula:** v_ij = m_k ⊕ v_{i-k,j-k} / 2^j (mod 1)

**Purpose:** Generate infinite sequence of direction numbers

---

## Legacy and Impact

**Citation count:** Highly cited (2000+ citations)

**Usage:** Default in most scientific computing libraries

**Impact:** Significantly improved practical performance of Sobol sequences

**Historical note:** Joe & Kuo's direction numbers are now considered standard for Sobol sequences
