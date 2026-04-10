# Feurer & Hutter (2019) — Hyperparameter Optimization

**Reference:** Feurer, M., & Hutter, F. (2019). Hyperparameter optimization. In F. Hutter, L. Kotthoff, & J. Vanschoren (Eds.), *Automated Machine Learning: Methods, Systems, Challenges* (pp. 3-45). Springer. https://doi.org/10.1007/978-3-030-05318-5_1

**Relevance:** Comprehensive HPO foundations. Backs sampler preset defaults, validation strategy, and overall optimization methodology.

---

## Chapter Scope

**Page reference:** Chapter 1, pp. 3-45

Comprehensive coverage of:
1. **Problem formulation** and examples
2. **HPO algorithms** (grid search, random search, Bayesian optimization, etc.)
3. **Multi-fidelity methods** (successive halving, Hyperband)
4. **Multi-objective HPO** (performance vs. resource trade-offs)
5. **Practical considerations** and best practices

---

## Problem Formulation (Section 1)

### Formal Definition

**Page reference:** Section 1, p. 4-5

**HPO as optimization problem:**
```
minimize: A(λ, D_val, D_train)
subject to: λ ∈ Λ
```

where:
- **A**: Algorithm (e.g., neural network training)
- **λ**: Hyperparameter configuration
- **Λ**: Search space (domain of hyperparameters)
- **D_train**: Training data
- **D_val**: Validation data
- **Objective**: Validation error (or related metric)

**Our implementation:** `GenericObjective` in `optimization/objective.py` follows this formulation

---

## Search Space Design (Section 2)

### Hyperparameter Types

**Page reference:** Section 2.1, p. 6-7

| Type | Description | Example |
|------|-------------|---------|
| **Continuous** | Real-valued | Learning rate (log scale) |
| **Integer** | Discrete numeric | Number of layers |
| **Categorical** | Unordered | Optimizer choice |
| **Conditional** | Depends on others | # units only if layer exists |

**Best practices (Section 2.3):**
- Start with literature defaults
- Use log-scale for multiplicative parameters
- Document search space rationale
- Handle conditional parameters carefully

**Our implementation:**
- `search_spaces/` provides BayesFlow-specific defaults
- `log=True` for learning rate and similar parameters
- `constant` fields for well-established BayesFlow defaults
- Conditional parameters via dataclass `if` statements

---

## HPO Algorithms (Section 3)

### Grid Search

**Page reference:** Section 3.1, p. 8-9

**Definition:** Evaluate all combinations of discretized hyperparameters

**Complexity:** O(∏_i d_i) where d_i is number of values for parameter i

**Use case:** Only for very small search spaces (< 20 total configurations)

**Our implementation:** Not recommended for bayesflow-hpo; use TPE or BO instead

### Random Search

**Page reference:** Section 3.2, p. 9-10

**Definition:** Sample random configurations uniformly

**Benefits:** Simple, parallelizable, often competitive

**Theoretical guarantee:** With probability 1 - ε, finds (1-ε)-quantile configuration with O(log(1/ε)) samples

**Our implementation:** `"random"` sampler preset

### Bayesian Optimization (Section 3.3)

**Page reference:** Section 3.3, p. 10-18

**Core idea:** Model objective function with probabilistic surrogate model

**Components:**
1. **Surrogate model:** GP (most common), TPE, Random Forest
2. **Acquisition function:** EI, UCB, PI
3. **Optimization:** Optimize acquisition to select next configuration

**Key algorithms:**
- **SMAC** (Sequential Model-based Algorithm Configuration)
- **TPE** (Tree-structured Parzen Estimator)
- **BOHB** (Bayesian Optimization with Hyperband)

**Our implementation:** `"tpe"`, `"gp"`, `"botorch"` sampler presets

---

## Multi-Fidelity Methods (Section 4)

### Successive Halving (SH)

**Page reference:** Section 4.1, p. 19-20

**Idea:** Start with many configurations, evaluate on small budget, keep only promising ones

**Algorithm:**
1. Evaluate n configurations on budget b
2. Keep top η fraction (usually 1/4)
3. Evaluate remaining on budget η·b
4. Repeat until max budget

**Complexity:** O(n·log(b_max/b_min))

### Hyperband

**Page reference:** Section 4.2, p. 20-22

**Innovation:** Run multiple SH instances in parallel with different budget allocations

**Benefits:** Strong empirical performance, parallelizable

**Our implementation:** `pruning_strategy="hyperband"` in `optimize()`

---

## Multi-Objective HPO (Section 5)

### Problem Formulation

**Page reference:** Section 5.1, p. 23-24

**Objectives:**
- Primary: Model performance (validation error)
- Secondary: Resource cost (training time, memory, inference speed)

**Solution concepts:**
- **Pareto optimality:** No single best configuration
- **Lexicographic ordering:** Prioritize objectives
- **Scalarization:** Weighted sum of objectives

**Our implementation:**
- Multi-objective optimization via `directions=["minimize", "minimize"]`
- `select_best_trial()` with `priorities` for lexicographic selection
- Pareto front visualization

---

## Best Practices (Section 6)

### Validation Strategy

**Page reference:** Section 6.1, p. 27-29

1. **Use fixed validation set** for consistency
2. **Report uncertainty** (multiple runs with different seeds)
3. **Use statistical tests** for significant differences
4. **Consider resource constraints**

**Our implementation:**
- `ValidationDataset` for fixed validation sets
- Per-condition metrics with aggregations
- Resource budgets via `max_param_count`, `max_memory_mb`

### Resource Management

**Page reference:** Section 6.2, p. 29-30

**Strategies:**
- **Early stopping:** Prune poorly performing trials
- **Parallelization:** Evaluate multiple configurations simultaneously
- **Warm starting:** Use previous HPO results to initialize new runs

**Our implementation:**
- `PeriodicValidationCallback` with pruning strategies
- Distributed storage via Optuna's `storage` parameter
- `warm_start_study` parameter for continuing previous studies

---

## Benchmark Problems (Section 7)

### Classic HPO Benchmarks

**Page reference:** Section 7, p. 31-35

| Benchmark | Domain | Objective |
|-----------|--------|------------|
| **SVM** | Classification | CV accuracy |
| **Random Forest** | Classification | CV accuracy |
| **Neural Networks** | Deep learning | Validation loss |
| **XGBoost** | Gradient boosting | CV log-loss |

**Our implementation:** bayesflow-hpo focuses on NPE models:
- **Two Moons**: Simple 2D benchmark
- **SLCP**: Stochastic Liear Gaussian Process
- **Lotka-Volterra**: Ordinary differential equation model

---

## Practical Tips

### Search Space Initialization

**Page reference:** Section 6.3, p. 30-31

**Recommendations:**
- Start with literature defaults
- Use wider ranges for less critical parameters
- Narrow ranges based on pilot runs

**Our implementation:**
- BayesFlow defaults from framework documentation
- `src/bayesflow_hpo/search_spaces/` provides well-researched ranges

### Algorithm Selection

**Page reference:** Section 6.4, p. 31-32

| Scenario | Recommended algorithm |
|----------|----------------------|
| Low-dimensional (<10) | Bayesian optimization |
| High-dimensional (>20) | Random search or TPE |
| Conditional parameters | TPE or SMAC |
| Multi-fidelity | Hyperband or BOHB |
| Multi-objective | NSGA-II or NSGA-III |

**Our implementation:**
- Default: `"tpe"` for most cases
- High-dimension: `"random"` as fallback
- Multi-objective: `"nsga2"` or `"nsga3"`
- GPU: `"botorch"` for gradient-based BO

---

## Intentional Deviations

None. bayesflow-hpo follows Feurer & Hutter (2019) recommendations:
- **TPE as default** (effective for conditional spaces)
- **Fixed validation datasets** (reproducibility)
- **Resource budgets** (param count, memory)
- **Multi-objective support** (performance vs. cost)
- **Early stopping** (pruning strategies)

**Key differences:**
- Specialized for BayesFlow NPE models
- Built-in SBC and C2ST validation
- Custom network architectures (CouplingFlow, FlowMatching, etc.)

---

## Related References

- **TPE details:** Bergstra et al. (2011) — See `bergstra2011_tpe.md`
- **SMAC:** Hutter et al. (2011) — Sequential Model-based Algorithm Configuration
- **BOHB:** Falkner et al. (2018) — Bayesian Optimization with Hyperband

---

## Key Takeaways

1. **Problem formulation:** HPO as black-box optimization (Section 1)
2. **Random search** is a strong baseline (Section 3.2)
3. **Bayesian optimization** effective for low-dimensional spaces (Section 3.3)
4. **Multi-fidelity methods** dramatically improve efficiency (Section 4)
5. **Best practices:** Fixed validation, resource budgets, uncertainty quantification (Section 6)

---

## Open Challenges

**Very high-dimensional spaces:** d > 100 hyperparameters

**Heterogeneous search spaces:** Mixed categorical/continuous/conditional

**Multi-objective trade-offs:** Automated selection among Pareto solutions

**Distributed optimization:** Scalable parallel HPO
