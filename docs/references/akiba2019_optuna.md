# Akiba et al. (2019) — Optuna

**Reference:** Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623-2631). https://doi.org/10.1145/3292500.3330701

**Relevance:** Backs `optimization/study.py` (sampler presets, study management) and overall HPO framework design.

---

## Design Criteria (Line 191-199)

**Page reference:** Section 1, p. 4

Optuna proposes three design criteria for next-generation HPO frameworks:

1. **Define-by-run API** (Line 193-194): Dynamically construct search space during runtime
2. **Efficient sampling and pruning** (Line 195-196): User-customizable algorithms
3. **Versatile architecture** (Line 197-199): From lightweight interactive to distributed computing

---

## Define-by-Run API (Line 217-330)

**Page reference:** Section 2, p. 5-6

### Key Concepts

**Study:** Single optimization process
**Trial:** Single evaluation of objective function

### Implementation Pattern
```python
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 4)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_l{i}', 1, 128))
    # ... train model and return score
```

**Benefits over define-and-run (e.g., Hyperopt):**
- No static search space definition required
- Natural use of loops/conditionals for conditional parameters
- More modular, interpretable code (compare Figure 1 vs Figure 2)

**Our implementation:** bayesflow-hpo's `optimize()` follows similar pattern, with search space objects (dataclass fields) sampled during trial execution.

---

## Sampling Algorithms

**Page reference:** Section 1, p. 3-4

Optuna supports multiple sampling strategies:
- **TPE** (Tree-structured Parzen Estimator) - default
- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
- **Random sampling**
- **Grid search**

**Our implementation:** `optimization/study.py` provides sampler presets:
- `"tpe"` - TPESampler
- `"random"` - RandomSampler
- `"botorch"` - BoTorch-based Bayesian optimization (lazy import)
- `"gp"` - Gaussian Process (GP)
- `"nsga2"` - NSGA-II for multi-objective
- `"nsga3"` - NSGA-III for many-objective
- `"auto"` - Automatic sampler selection

---

## Pruning Algorithms

**Page reference:** Section 1, p. 3

Optuna supports pruning strategies (monitor intermediate results, kill unpromising trials):
- **Hyperband** (Li et al., 2018) - bandit-based algorithm
- **Median pruning** - prune based on median intermediate score
- **Successive Halving** - early stopping strategy

**Our implementation:** `optimization/pruning_strategies.py` provides:
- `"median"` - MedianPruner (single-objective)
- `"hyperband"` - Hyperband pruning
- Custom multi-objective strategies via `PeriodicValidationCallback`

---

## Architecture Features

**Easy installation:** Single command pip install (Line 154-155)

**Distributed computing:** Support for parallel trials (Line 132-136)

**Storage backend:** RDBMS-based storage for study persistence and distributed optimization

**Our implementation:**
- Supports `storage=` parameter in `create_study()` and `optimize()`
- Checkpoint persistence via `CheckpointPool`
- Database storage for trial results and Pareto front tracking

---

## Intentional Deviations

None. bayesflow-hpo's design closely follows Optuna's principles:
- Define-by-run via dataclass search spaces
- Modular objective functions
- Efficient pruning with `PeriodicValidationCallback`
- Versatile deployment (local or distributed)

**Key difference:** bayesflow-hpo specializes for BayesFlow NPE models, with built-in:
- Validation pipeline with fixed datasets
- Parameter/memory budget constraints
- Custom network architectures (CouplingFlow, FlowMatching, etc.)
- SBC and C2ST validation metrics

---

## Fulltext

See `fulltexts/akiba2019.md` for complete paper text.
