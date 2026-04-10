# Deb et al. (2002) — NSGA-II

**Reference:** Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, *6*(2), 182-197. https://doi.org/10.1109/4235.996017

**Relevance:** Backs `optimization/pruning_strategies.py` (_non_dominated_sort, lexicographic-Pareto selection) and `results/extraction.py` (select_best_trial with mean-rank tiebreak).

---

## Fast Non-Dominated Sorting (Line 252-275)

**Page reference:** Section III-A, p. 183-184

**Computational complexity:** O(MN²) where M = number of objectives, N = population size

### Algorithm
```
For each solution p in population:
    Calculate n_p = domination count (number of solutions dominating p)
    Calculate S_p = set of solutions dominated by p
    (Requires O(MN²) comparisons)

Initialize front counter i = 1
Initialize front F_i = {all p with n_p = 0}

while F_i is not empty:
    Initialize next front Q = empty
    for each p in F_i:
        for each q in S_p:
            n_q = n_q - 1
            if n_q == 0:
                q.rank = i + 1
                Q = Q ∪ {q}
    i = i + 1
    F_i = Q
```

**Key insight:** Each solution visited at most M times before domination count reaches zero. Total complexity O(MN²).

**Our implementation note:** `optimization/pruning_strategies.py:_non_dominated_sort()` follows this algorithm exactly, using numpy for efficiency.

---

## Crowding Distance (Line 336-366)

**Page reference:** Section III-B, p. 185

**Purpose:** Density estimation for diversity preservation without user-defined parameters

### Definition
Crowding distance of solution i = average perimeter of cuboid formed by nearest neighbors along each objective.

**Boundary solutions:** Assigned infinite distance (smallest and largest function values for each objective)

**Intermediate solutions:** For each objective m:
```
distance_m(i) = |f_m(i+1) - f_m(i-1)| / (f_m^max - f_m^min)
```

**Total crowding distance:** Sum across all objectives

**Computational complexity:** O(MN log N) dominated by sorting

**Our implementation note:** Used in `select_best_trial()` for tiebreaking among Pareto-optimal trials.

---

## Crowded-Comparison Operator (Line 385-391)

**Page reference:** Section III-B, p. 185

**Partial order ≺_n:**
```
p ≺_n q if:
    rank(p) < rank(q)
    OR
    rank(p) == rank(q) AND distance(p) > distance(q)
```

**Interpretation:**
- Prefer solution with better (lower) nondomination rank
- If same rank, prefer solution in less crowded region (higher crowding distance)

**Our implementation:** Used for trial selection in `results/extraction.py:select_best_trial()`.

---

## Key Algorithm Features

**Elitism (Line 22):** Selection operator combines parent and offspring populations, selecting best with respect to fitness and spread.

**No sharing parameter:** Unlike original NSGA, NSGA-II uses crowding distance instead of sharing function approach, eliminating need for user-specified σ_share parameter.

---

## Intentional Deviations

None. bayesflow-hpo implements NSGA-II concepts exactly as specified:
- `_non_dominated_sort()` follows fast sorting algorithm
- Crowding distance used for tiebreaking in `select_best_trial()`
- Dominance-based pruning follows Section III-B principles

---

## Fulltext

See `fulltexts/deb2002.md` for complete paper text.
