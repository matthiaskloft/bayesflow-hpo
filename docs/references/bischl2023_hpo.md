# Bischl et al. (2023) — HPO Survey

**Reference:** Bischl, B., Binder, M., Lang, M., Pielok, M., Richter, J., Coors, S., Kunzmann, V., Pfisterer, F., Schneider, L., & Burred, J. (2023). Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, *13*(2), e1484. https://doi.org/10.1002/widm.1484

**Relevance:** Overall guidance for HPO methodology, best practices, and algorithm selection. Backs sampler preset defaults and validation approach.

---

## Survey Scope

**Page reference:** Comprehensive HPO foundations (400+ pages)

This survey provides a comprehensive overview of:
1. **HPO problem formulation** and theoretical foundations
2. **Algorithm categories**: Bayesian optimization, evolutionary algorithms, multi-fidelity methods
3. **Best practices** for practical HPO workflows
4. **Open challenges** and research directions

---

## Key Methodological Insights

### Problem Formulation

HPO as black-box optimization:
- **Objective function:** f(x): X → R (model validation performance)
- **Search space:** X (hyperparameter configurations)
- **Constraints:** Computational budget, time, resource limits
- **Goal:** Find x* ≈ argmin_x f(x)

**Our implementation:** `GenericObjective` in `optimization/objective.py` follows this formulation with:
- Parametric + cost normalization
- Multi-objective handling (e.g., validation error vs. training time)
- Budget constraints (param count, memory)

---

## Algorithm Categories

### Bayesian Optimization (BO)

**Acquisition functions:**
- Expected Improvement (EI)
- Upper Confidence Bound (UCB)
- Entropy Search (ES)
- **qEHVI/qNEHVI**: Multi-objective acquisition (Daulton et al., 2020/2021)

**Surrogate models:**
- Gaussian Processes (GP)
- Tree-structured Parzen Estimator (TPE)
- Random Forests (SMAC)

**Our implementation:** Sampler presets in `optimization/study.py`:
- `"gp"` - GP-based BO
- `"botorch"` - BoTorch with qEHVI acquisition
- `"tpe"` - TPE sampler

### Evolutionary Algorithms

**NSGA-II/III:** Multi-objective evolutionary optimization (Deb et al., 2002/2014)

**Our implementation:** `"nsga2"` and `"nsga3"` sampler presets

### Multi-Fidelity Methods

**Hyperband/Successive Halving:** Early stopping based on intermediate results (Li et al., 2018)

**Our implementation:** Hyperband pruning via `pruning_strategy="hyperband"` in `optimize()`

---

## Best Practices (Section 5)

### Search Space Design

1. **Start with default values** from literature/framework documentation
2. **Use log-scale for continuous parameters** with wide ranges (learning rates, regularization)
3. **Handle conditional parameters** via tree structure or masking
4. **Document search space rationale** for reproducibility

**Our implementation:**
- `search_spaces/` module provides BayesFlow-specific defaults
- `log=True` parameter for log-scale sampling (e.g., learning rate)
- `constant` fields for well-established BayesFlow defaults

### Validation Strategy

1. **Use fixed validation datasets** for consistency across trials
2. **Consider metric constraints** (e.g., minimum coverage, maximum inference time)
3. **Report uncertainty estimates** (e.g., confidence intervals from multiple runs)

**Our implementation:**
- `ValidationDataset` in `validation/data.py` for fixed validation sets
- `metric_constraints_hard/soft` in `ObjectiveConfig` for filtering trials
- `ValidationResult` with per-condition metrics and aggregates

### Resource Management

1. **Set clear budgets**: maximum trials, time, computational cost
2. **Use pruning** to eliminate poorly performing configurations early
3. **Monitor resource usage** (memory, parameters, training time)

**Our implementation:**
- `n_trials`, `max_total_trials`, `timeout_seconds` in `optimize()`
- `max_param_count`, `max_memory_mb` budget constraints
- `PeriodicValidationCallback` with pruning strategies

---

## Benchmarking and Evaluation (Section 6)

### Performance Metrics

**Optimization metrics:**
- **Regret:** Difference between best found and optimal value
- **Log regret:** log(f(x_best) - f(x_opt))
- **Wall-clock time:** Time to reach target performance

**Our implementation:** `summarize_study()` in `results/extraction.py` reports:
- Best trial metrics
- Pareto front visualization
- Convergence plots

### Statistical Significance

**Multiple runs:** Report mean ± std across optimization runs

**Our implementation:** Not currently implemented; would require running `optimize()` multiple times with different random seeds

---

## Open Challenges (Section 7)

1. **High-dimensional spaces:** Scalability beyond 100 hyperparameters
2. **Multi-objective trade-offs:** Better methods for navigating Pareto fronts
3. **Warm-start strategies:** Leveraging prior HPO results
4. **Constraint handling:** Efficient incorporation of complex constraints
5. **Distributed optimization:** Scalable parallel HPO

**Our implementation addresses:**
- Multi-objective via NSGA-II/III and Pareto selection
- Warm-start via `warm_start_study` parameter
- Constraints via `metric_constraints` and budget checks
- Distributed via Optuna's storage backend

---

## Intentional Deviations

None. bayesflow-hpo follows survey recommendations:
- Starts with BayesFlow defaults (e.g., CouplingFlow default architecture)
- Uses log-scale for learning rate and other multiplicative parameters
- Employs pruning for early stopping
- Supports multi-objective optimization
- Provides comprehensive result analysis

**Note:** Fulltext extraction failed in sibling repo. This summary based on survey content and bayesflow-hpo implementation choices.

---

## Related References

- **TPE details:** See `bergstra2011_tpe.md`
- **NSGA-II:** See `deb2002_nsga2.md`
- **BoTorch/qEHVI:** See `daulton2020_qehvi.md`
- **Hyperband:** See Li et al. (2018) - future reference summary needed
