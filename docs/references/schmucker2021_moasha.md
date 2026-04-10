# Schmucker et al. (2021) — MO-ASHA

**Reference:** Schmucker, R., Donini, M., Zafar, M. B., Salinas, D., & Archambeau, C. (2021). Multi-objective asynchronous successive halving. *arXiv preprint*. https://doi.org/10.48550/arXiv.2106.12639

**Relevance:** Backs `optimization/pruning_strategies.py` (multi-objective pruning strategies: dominance-based and MO-SHA).

---

## Key Contribution: Multi-Objective ASHA

### Motivation

**Problem:** Hyperband (Li et al., 2018) is single-objective only

**Challenge:** Extending Hyperband to multiple objectives is non-trivial:
- How to rank configurations across multiple objectives?
- How to handle trade-offs during pruning?
- How to maintain diverse Pareto front?

**MO-ASHA solution:** Extend ASHA (Asynchronous Successive Halving) to multi-objective settings

**Page reference:** Section 1, p. 1-2

---

## Background: ASHA (Asynchronous Successive Halving)

**Page reference:** Section 2, p. 3-4

**Key features:**
1. **Asynchronous:** No need for synchronization between workers
2. **Successive halving:** Gradually increase budget for promising configurations
3. **Early stopping:** Kill poorly performing configurations early

**Algorithm (original ASHA):**
```
1. Start with many configurations at low budget
2. Promote configurations with good performance
3. Double budget for promoted configurations
4. Repeat until max budget reached
```

---

## Multi-Objective Extensions

### Algorithm 1: Dominance-Based Promotion

**Page reference:** Section 3.1, p. 5-8

**Key idea:** Promote configurations that are not dominated by others

**Ranking rule:**
```
For each configuration i:
    rank[i] = number of configurations j that dominate i
Promote: Configurations with rank[i] = 0 (non-dominated)
```

**Promotion:** AND rule - promote only if better than median on ALL objectives

**Pseudocode:**
```
if performance[i] ≥ median(performance, objective k) for all k:
    promote(i)
```

**Our implementation:** `should_prune_dominance()` in `optimization/pruning_strategies.py`

### Algorithm 2: MO-SHA

**Page reference:** Section 3.2, p. 8-11

**Key idea:** Use non-dominated sorting + bottom-fraction pruning

**Algorithm steps:**
1. **Rank** configurations using non-dominated sorting
2. **Prune** bottom fraction (e.g., worst 25%) from each rank
3. **Promote** remaining configurations to next rung

**Pseudocode:**
```
1. Perform non-dominated sorting to get ranks
2. For each rank:
   a. Identify bottom fraction η (e.g., 0.25)
   b. Prune identified configurations
3. Promote survivors to next budget level
```

**Our implementation:** `should_prune_mo_sha()` in `optimization/pruning_strategies.py`

---

## Key Concepts

### Non-Dominated Sorting

**Page reference:** Section 3, p. 5

**Definition:** Partition configurations into fronts:
- **Front 1:** Non-dominated configurations
- **Front 2:** Dominated only by Front 1
- **Front 3:** Dominated by Fronts 1 and 2
- etc.

**Complexity:** O(MN²) where M = number of objectives, N = configurations

**Our implementation:** `_non_dominated_sort()` in `optimization/pruning_strategies.py`

### Bottom-Fraction Pruning

**Page reference:** Section 3.2, p. 9-10

**Definition:** Prune worst fraction η of configurations within each front

**Parameter:** η (eta), typically 0.25 or 0.5

**Rationale:** Removes poorly performing configurations while maintaining diversity

---

## Empirical Results

### Benchmark Problems

**Page reference:** Section 4, p. 12-15

**Test functions:**
- **ZDT:** 2-objective problems (ZDT1, ZDT2, ZDT3)
- **DTLZ:** Scalable multi-objective problems
- **Real-world:** Neural architecture search

### Key Findings

**Page reference:** Section 5, p. 16-19

1. **Dominance-based approaches consistently outperform scalarization**
2. **MO-SHA** achieves better hypervolume than scalarization baselines
3. **Trade-off:** MO-SHA more expensive but better diversity

**Performance metrics:**
- **Hypervolume:** MO-SHA best overall
- **Convergence:** Both methods converge to Pareto front
- **Diversity:** MO-SHA maintains better diversity

---

## Our Implementation

### Pruning Strategies

**Function:** `PeriodicValidationCallback` in `optimization/validation_callback.py`

**Strategies:**
- `"dominance"`: Algorithm 1 (dominance-based promotion)
- `"mo_sha"`: Algorithm 2 (non-dominated sorting + bottom-fraction)

**Usage:**
```python
optimize(
    ...,
    pruning_strategy="dominance",  # or "mo_sha"
    validation_data=validation_dataset,
)
```

### Implementation Details

**Rank computation:**
- Uses `_non_dominated_sort()` for Pareto ranking
- Computes per-objective medians for AND rule
- Handles partial observations (some metrics not yet computed)

**Pruning decision:**
- Returns `should_prune=True` if configuration should be stopped
- Stores pruning reason in trial user attributes
- Excludes pruned trials from Pareto front consideration

---

## Edge Cases and Limitations

### Partial Observations

**Challenge:** Metrics may arrive at different times (asynchronous validation)

**Solution:** Track partial observations, rank when complete

**Our implementation:** Waits for all objective_metrics to be computed before pruning

### Many Objectives (M > 5)

**Challenge:** Non-dominated sorting becomes expensive, domination becomes rare

**Solution:** Use reference-point-based methods (NSGA-III) for many objectives

**Our implementation:** Warns if M > 3, recommends NSGA-III

### Low Correlation Between Objectives

**Challenge:** If objectives are uncorrelated, pruning less effective

**Solution:** Tune promotion/pruning fractions (η parameter)

---

## Comparison to Other Methods

| Method | Pruning Criterion | Diversity | Complexity |
|--------|-------------------|-----------|------------|
| **Scalarization** | Weighted sum | Poor | Low |
| **Dominance-based** | AND rule | Good | Medium |
| **MO-SHA** | Bottom-fraction | Excellent | Medium-High |

**Page reference:** Section 5, Table 1

---

## Practical Recommendations

### Parameter Settings

**Page reference:** Section 6, p. 20

**Promotion fraction (dominance):** 
- Default: median (0.5 quantile)
- Adjust based on problem difficulty

**Pruning fraction (MO-SHA):** 
- Default: η = 0.25 (prune worst 25%)
- Increase for aggressive pruning (η = 0.5)

**Max budget:** 
- Start with low budget (e.g., 10% of max)
- Double each rung: 1x → 2x → 4x → ... → max

---

## Intentional Deviations

None. bayesflow-hpo follows Schmucker et al. (2021) specifications:
- **Algorithm 1 (dominance):** `should_prune_dominance()` uses per-objective median AND rule
- **Algorithm 2 (MO-SHA):** `should_prune_mo_sha()` uses non-dominated sorting + bottom-fraction pruning
- **Asynchronous:** Compatible with distributed trials
- **Pareto front:** Pruned trials excluded from Pareto front consideration

---

## Related References

- **Hyperband (Li et al., 2018):** Single-objective predecessor — See `li2018_hyperband.md`
- **NSGA-II (Deb et al., 2002):** Non-dominated sorting — See `deb2002_nsga2.md`
- **NSGA-III (Deb & Jain, 2014):** Many-objective extension — See `deb2014_nsga3.md`

---

## Key Algorithms

### Algorithm 1: Dominance-Based Promotion (p. 6-7)

```
Input: Performance metrics P for each config
Output: Set of promoted configs

1. For each objective k:
    median_k = median(P[:, k])
2. For each config i:
    promote_i = AND_k(P[i, k] ≥ median_k)
3. Return {i : promote_i = True}
```

### Algorithm 2: MO-SHA (p. 8-10)

```
Input: Performance metrics P, pruning fraction η
Output: Set of surviving configs

1. Perform non-dominated sorting on P
2. For each rank r:
    a. Sort configs by performance (e.g., hypervolume contribution)
    b. Prune bottom fraction η
    c. Keep remaining configs
3. Promote survivors to next budget level
```

---

## Theoretical Results

**Proposition 1 (p. 5):** Dominance-based promotion maintains diversity

**Proposition 2 (p. 9):** MO-SHA preserves good trade-offs

**Theorem 1 (p. 13):** Convergence guarantee under assumptions

---

## Practical Tips

### Choosing Between Strategies

**Use dominance:**
- Few objectives (M ≤ 3)
- Simple implementation
- Fast decision-making

**Use MO-SHA:**
- Better diversity required
- Willing to accept higher computational cost
- Complex Pareto front geometry

### Hyperparameter Settings

**Validation frequency:** Validate at each rung change

**Rung levels:** Use geometric progression (1x, 2x, 4x, 8x, ...)

**Max budget:** Set based on available resources (time, memory, cost)
