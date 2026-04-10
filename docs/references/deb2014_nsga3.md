# Deb & Jain (2014) — NSGA-III

**Reference:** Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints. *IEEE Transactions on Evolutionary Computation*, *18*(4), 577-601. https://doi.org/10.1109/TEVC.2013.2281535

**Relevance:** Backs `"nsga3"` sampler preset in `optimization/study.py` for many-objective optimization (4+ objectives).

---

## Key Contribution: NSGA-III for Many-Objective Optimization

### Motivation

**Problem:** NSGA-II (Deb et al., 2002) performs poorly for many-objective problems (M ≥ 4)

**Challenges with many objectives:**
1. **Most solutions become non-dominated** (loss of selection pressure)
2. **Crowding distance loses meaning** in high-dimensional space
3. **Pareto dominance becomes weak** (many solutions incomparable)

**NSGA-III solution:** Reference-point-based selection instead of crowding distance

**Page reference:** Section 1, p. 578-579

---

## Background: Many-Objective Optimization Challenges

### Dominance Resistance

**Page reference:** Section 2, p. 579-581

**Definition:** In high-dimensional objective space, most solutions are non-dominated

**Impact:** NSGA-II's selection pressure diminishes as M increases

**Example:** For uniformly distributed points, probability of being non-dominated approaches 1 - (M!/M^M) ≈ 1 - (1/e) for large M

### Crowding Distance Issues

**Page reference:** Section 2.1, p. 581-582

**Problem:** Crowding distance assumes hyper-rectangular distribution, which fails for:
- Non-convex Pareto fronts
- Irregular front shapes
- Many objectives

**Alternative:** Need diversity preservation method that works for many objectives

---

## NSGA-III Algorithm

### Core Innovation: Reference Points

**Page reference:** Section 3, p. 583-585

**Key idea:** Use reference points Z instead of crowding distance for diversity preservation

**Reference points:**
- Pre-specified points in objective space
- Usually generated on a systematic hyperplane
- Number: Typically 2-4× population size

### Algorithm Outline

**Page reference:** Section 3, p. 585-589

**Steps:**
1. **Classification:** Associate each solution with nearest reference point
2. **Niche count:** Count number of solutions associated with each reference point
3. **Selection:** Prefer solutions with:
   - Better nondomination rank
   - Lower niche count (less crowded reference region)

**Pseudocode:**
```
1. Perform non-dominated sorting to get ranks
2. For each rank:
   a. Normalize objectives
   b. For each solution, find nearest reference point
   c. Compute niche count for each reference point
   d. Select solutions with best rank, then lowest niche count
```

---

## Reference Point Generation

### Das-Dennis Method

**Page reference:** Section 3.1, p. 585-587

**Algorithm:**
1. Generate systematic points on unit hyperplane Σ x_i = 1
2. Translate and scale to match objective bounds

**Parameters:**
- **H:** Number of divisions per dimension
- **Total points:** C(H+M-1, M-1) = (H+M-1)! / ((M-1)! H!)

**Example:** For M=4, H=3: (3+4-1)! / ((4-1)!·3!) = 6! / (6·6) = 720 / 36 = 20 points

### Adaptive Reference Points

**Page reference:** Section 3.2, p. 587-589

**Challenge:** Fixed reference points may not cover discovered Pareto front

**Solution:** Periodically update reference points based on current population

**Adaptive strategy:**
- Generate initial reference points
- Every τ generations, update to cover current Pareto front
- Maintain diversity by niching mechanism

---

## Selection Operator

### Niche Count Calculation

**Page reference:** Section 3.3, p. 589-590

**Definition:**
```
niche_count[j] = number of solutions associated with reference point Z[j]
```

**Association rule:** Associate each solution with its nearest reference point (Euclidean distance)

### Niche Sharing

**Page reference:** Section 3.3, p. 590-591

**Purpose:** Prevent over-crowding in popular reference regions

**Method:** Discount niche count if two solutions are very close

**Formula:**
```
share_distance = distance / σ_share
if share_distance < 1:
    niche_count *= share_distance
```

---

## Our Implementation

**Sampler preset:** `"nsga3"` in `create_study()` and `optimize()`

**Configuration:**
```python
from optuna.samplers import NSGAIIISampler

sampler = NSGAIIISampler(
    population_size=100,  # Must be divisible by (M-1) for reference points
    mutation_prob=0.1,
    crossover_prob=0.9,
)
```

**Usage:**
```python
study = optuna.create_study(
    directions=["minimize"] * M,  # M ≥ 4 for NSGA-III
    sampler="nsga3"
)
```

**Constraint:** Population size must be large enough (≥ 4×number of reference points)

**Implementation details:**
- Uses Optuna's `NSGAIIISampler` (confusingly named despite being NSGA-III)
- Reference points generated via Das-Dennis method
- Niche count used for diversity preservation
- Crowding distance replaced by niche counting

---

## Comparison: NSGA-II vs NSGA-III

| Aspect | NSGA-II (Deb et al., 2002) | NSGA-III (Deb & Jain, 2014) |
|--------|------------------------------|------------------------------|
| **Objectives** | 2-3 | 4+ (many-objective) |
| **Diversity metric** | Crowding distance | Reference points + niche count |
| **Selection pressure** | Good for M≤3 | Maintained for M≥4 |
| **Complexity** | O(MN²) | O(MN²) + reference point overhead |
| **Reference points** | None | Required (Das-Dennis) |

**Page reference:** Section 5, p. 594-596 (empirical comparison)

---

## Empirical Results

### Benchmark Problems

**Page reference:** Section 4, p. 591-594

**Test functions:**
- **DTLZ1-4:** Scalable to many objectives (3-15 objectives)
- **WFG1-9:** Walking fish group problems
- **Real-world:** Car crashworthiness, water resource management

### Key Findings

**Page reference:** Section 5, p. 594-596

1. **NSGA-III** outperforms NSGA-II for M ≥ 4
2. **Reference point generation:** Das-Dennis method works well
3. **Adaptive updates** further improve performance
4. **Computation time:** Similar to NSGA-II (reference point overhead is minor)

---

## Edge Cases and Limitations

### Very Many Objectives (M > 15)

**Challenge:** Reference point computation becomes expensive

**Solution:** Use light reference point sets or alternative methods

### Non-Convex Pareto Fronts

**Challenge:** Reference points may not properly sample irregular fronts

**Solution:** Adaptive reference points that follow Pareto front shape

### Small Population

**Challenge:** Need enough solutions to populate all reference regions

**Recommendation:** Population size ≥ 4 × number of reference points

---

## Practical Recommendations

### Population Size

**Page reference:** Section 6, p. 597-598

**Formula:** N ≥ 4 × |Z| where |Z| is number of reference points

**Example:** For M=4, H=3, |Z|=20, use N ≥ 80

**Our implementation:** Default population size automatically scales with objective count

### Reference Point Settings

**Number of divisions (H):**
- Default: H = M-1 (balances coverage and computation)
- Increase for better coverage (H = 2M)
- Decrease for faster computation (H = 2)

**Adaptive updates:**
- Enable if Pareto front shape unknown
- Update frequency: Every 50-100 generations

---

## Intentional Deviations

None. bayesflow-hpo uses NSGA-III as specified:
- Reference points via Das-Dennis method
- Niche count for diversity preservation
- Association via nearest Euclidean distance
- Population size scales with objective count

**Implementation:** Via Optuna's `NSGAIIISampler` (despite name, implements NSGA-III)

---

## Related References

- **NSGA-II (Deb et al., 2002):** Original 2-3 objective method — See `deb2002_nsga2.md`
- **Crowding distance (Deb et al., 2002):** Diversity metric for NSGA-II — See `deb2002_nsga2.md`
- **Reference point theory:** Das & Dennis (1998) — Das-Dennis method
- **Many-objective testing:** DTLZ and WFG problem suites

---

## Key Theoretical Results

**Theorem 1 (p. 583):** NSGA-III maintains diversity under reference-point selection

**Theorem 2 (p. 584):** Preference criterion follows reference point association

**Proposition 1 (p. 590):** Niche count preserves diversity in reference regions

---

## Historical Context

**Significance:** NSGA-III made many-objective optimization tractable

**Impact:** Widely adopted in evolutionary multi-objective optimization community

**Applications:**
- Engineering design (car safety, aircraft design)
- Resource allocation (water, energy)
- Machine learning (multi-objective HPO)

---

## Algorithm Pseudocode

```
Input: Population P_t, objectives M, reference points Z

1. Non-dominated sort P_t into fronts F_1, ..., F_k
2. Normalize objectives to [0, 1]
3. For each solution x in P_t:
    a. Associate with nearest reference point z ∈ Z
    b. Increment niche_count[z]
4. Select from P_t based on:
    a. Nondomination rank (prefer F_1 over F_2, etc.)
    b. Niche count (prefer less crowded regions)
5. Create offspring via crossover/mutation
6. Repeat from step 1 until convergence
```

**Page reference:** Section 3, p. 585-589
