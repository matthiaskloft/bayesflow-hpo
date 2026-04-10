# Emmerich & Deutz (2018) — Multi-Objective Optimization Tutorial

**Reference:** Emmerich, M. T. M., & Deutz, A. H. (2018). A tutorial on multiobjective optimization: Fundamentals and evolutionary methods. *Natural Computing*, *17*(3), 585-609. https://doi.org/10.1007/s11047-018-9685-y

**Relevance:** Backs `optimization/pruning_strategies.py` (MOO fundamentals, Pareto dominance definitions) and `results/extraction.py` (Pareto front selection).

---

## Scope of Tutorial

**Page reference:** Section 1, p. 586-587

Comprehensive tutorial covering:
1. **MOO fundamentals** (Pareto optimality, dominance)
2. **Evolutionary algorithms** (NSGA-II, SMS-EMOA, IBEA)
3. **Indicator-based methods** (Hypervolume, R2)
4. **Benchmark problems** and performance assessment

---

## MOO Fundamentals (Section 2)

### Pareto Dominance (Definition 5)

**Page reference:** Section 2.1, p. 588

**Definition:** For minimization problems, x dominates y if:
- f_i(x) ≤ f_i(y) for all objectives i = 1,...,M
- f_j(x) < f_j(y) for at least one objective j

**Notation:** x ≺ y (x dominates y)

**Our implementation:** `_non_dominated_sort()` in `optimization/pruning_strategies.py`

### Pareto Optimality (Definition 6)

**Page reference:** Section 2.1, p. 588

**Definition:** x* is Pareto optimal if there is no x such that x ≺ x*

**Pareto set:** Set of all Pareto optimal solutions

**Pareto front:** Image of Pareto set under objective function f

---

## Key Concepts

### Dominance Relations

**Page reference:** Section 2.1

**Pareto dominance:** As defined above

**ε-dominance:** x ε-dominates y if f_i(x) ≤ f_i(y) + ε for all i

**Hypervolume dominance:** Based on hypervolume contribution

### Performance Assessment

**Page reference:** Section 2.2, Definitions 7-9

**Quality indicators (Proposition 7):**
- **Hypervolume (HV):** Volume dominated by Pareto front relative to reference point
- **R2 indicator:** Average distance to reference set
- **Unary hypervolume:** Individual contribution to hypervolume

**Our implementation:** `summarize_study()` computes hypervolume via Optuna's `calculate_hypervolume()`

---

## Evolutionary MOO Algorithms (Section 3)

### NSGA-II Overview

**Page reference:** Section 3.2, p. 593-594

**Key components:**
1. **Fast non-dominated sorting:** O(MN²) complexity
2. **Crowding distance:** Diversity preservation
3. **Elitism:** Combines parent and offspring populations

**See `deb2002_nsga2.md` for detailed coverage**

### SMS-EMOA (S-Metric Selection EMOA)

**Page reference:** Section 3.3, p. 595-596

**Key innovation:** Selection based on hypervolume contribution

**Algorithm:**
1. Generate offspring via mutation/recombination
2. Select based on contribution to hypervolume
3. Maintain archive of non-dominated solutions

### IBEA (Indicator-Based Evolutionary Algorithm)

**Page reference:** Section 3.4, p. 597-598

**Key innovation:** Binary indicator for fitness (e.g., hypervolume difference)

**Benefits:** Flexible, can use any quality indicator

---

## Complexity Analysis (Proposition 9)

**Page reference:** Section 3.5, p. 599

**Non-dominated sorting complexity:** O(MN²)

**Proof sketch:** Each solution can be in at most M fronts, each solution compared at most M times per front

**Our implementation:** Matches this complexity in `_non_dominated_sort()`

---

## Benchmark Problems (Section 4)

### Test Suites

**ZDT (Zitzler-Deb-Thiele):** 2-objective problems
- ZDT1: Convex Pareto front
- ZDT2: Non-convex front
- ZDT3: Disconnected front
- ZDT4: Non-uniform search space

**DTLZ (Deb-Thiele-Laumanns-Zitzler):** Scalable to M objectives
- DTLZ1: Linear front
- DTLZ2: Spherical front
- DTLZ3: Scalable test

**Page reference:** Section 4.1, p. 600-601

---

## Our Implementation

### Pareto Dominance

**Function:** `_non_dominated_sort()` in `optimization/pruning_strategies.py`

**Complexity:** O(MN²) as per Emmerich & Deutz (2018)

**Usage:**
- Multi-objective pruning strategies
- Pareto front visualization
- Trial selection in `select_best_trial()`

### Crowding Distance

**Page reference:** Section 3.2 (discussing NSGA-II)

**Used for:** Diversity preservation in trial selection

**Function:** Not directly implemented, but equivalent to Deb et al. (2002) crowding distance

### Hypervolume Calculation

**Page reference:** Section 2.2, Definition 7

**Implementation:** Uses Optuna's hypervolume computation via `._hypervolume`

**Reference point:** Automatically determined from worst objective values

---

## Key Definitions

### Definition 5: Pareto Dominance (p. 588)

See above under MOO Fundamentals

### Definition 7: Hypervolume (p. 589)

**Definition:** HV(F) = λ({z ∈ ℝ^M : ∃x∈F, z ≺ x})

**Interpretation:** Volume of objective space dominated by F

**Properties:**
- Monotonic with respect to dominance
- Requires reference point for bounded computation

### Definition 9: Unary Hypervolume (p. 590)

**Definition:** Contribution of individual solution to total hypervolume

**Use:** Selection in SMS-EMOA

---

## Practical Recommendations

### Algorithm Selection

**Page reference:** Section 5, p. 603-604

| Scenario | Recommended algorithm |
|----------|----------------------|
| General purpose | NSGA-II |
| Hypervolume-focused | SMS-EMOA |
| Many objectives (M > 3) | NSGA-III or SMS-EMOA |
| Limited budget | MOEA/D or similar |

### Parameter Settings

**Population size:** 100-200 for 2-3 objectives
**Number of generations:** 500-1000
**Crossover probability:** 0.8-0.9
**Mutation probability:** 1/N (N = number of variables)

---

## Intentional Deviations

None. bayesflow-hpo follows Emmerich & Deutz (2018) definitions:
- Pareto dominance: As defined in Definition 5
- Non-dominated sorting: O(MN²) complexity
- Hypervolume: As defined in Definition 7

**Implementation details:**
- Uses numpy for efficient dominance computation
- Crowding distance equivalent via mean-rank tiebreaking
- Reference point selection follows best practices

---

## Related References

- **NSGA-II details:** Deb et al. (2002) — See `deb2002_nsga2.md`
- **NSGA-III:** Deb & Jain (2014) — Future reference summary
- **Hypervolume theory:** Zitzler et al. (2003) — Original HV paper

---

## Key Takeaways

1. **Pareto optimality** is fundamental concept (Definition 6)
2. **Non-dominated sorting** has O(MN²) complexity (Proposition 9)
3. **Hypervolume** is gold standard quality indicator (Definition 7)
4. **Crowding distance** preserves diversity (NSGA-II Section 3.2)
5. **Evolutionary algorithms** are well-suited for MOO (Section 3)

---

## Open Challenges

**Many-objective optimization:** Scalability for M > 10 (Section 6)

**Expensive optimization:** Limited function evaluations (Section 6)

**Dynamic objectives:** Objectives changing over time (Section 6)

**Stochastic objectives:** Noise in objective evaluations (Section 6)
