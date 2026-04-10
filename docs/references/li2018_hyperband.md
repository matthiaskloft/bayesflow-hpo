# Li et al. (2018) — Hyperband

**Reference:** Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research*, *18*, 1-52. https://doi.org/10.5555/3294984.3295026

**Relevance:** Backs `pruning_strategy="hyperband"` in `optimize()` and budget-based early stopping.

---

## Key Contribution: Successive Halving for HPO

### Motivation

**Problem:** HPO requires evaluating many configurations, each potentially expensive

**Challenge:** Random search explores uniformly but wastes resources on poor configurations

**Insight:** Most configurations are bad; quickly identify promising ones

**Hyperband solution:** Bandit-based resource allocation that adaptively focuses on promising configurations

**Page reference:** Section 1, p. 1-4

---

## Background: Bandit Problems

### Multi-Armed Bandit Problem

**Page reference:** Section 2, p. 4-5

**Setting:** Choose among K arms (actions) to maximize total reward

**Trade-off:** 
- **Exploration:** Try different arms to learn their rewards
- **Exploitation:** Use currently known best arm

**Regret:** Difference between optimal and achieved reward

**Goal:** Minimize cumulative regret over time horizon

---

## Successive Halving

### Core Algorithm

**Page reference:** Section 3, p. 5-8

**Setting:** 
- N configurations (arms)
- Budget B (e.g., number of epochs, training samples)
- Reduction rate η (typically 4)

**Algorithm:**
```
1. Start with N configurations at budget B/η^s (s = 0,...,S)
2. Evaluate all configurations at current budget
3. Keep top 1/η fraction (best performers)
4. Double budget for remaining configurations: B/η^(s-1) → B/η^s
5. Repeat until max budget B reached
```

**Complexity:** O(N log η(B_max/B_min))

### Key Properties

**Page reference:** Section 3.1, p. 6-7

**Theorem 1:** SH achieves O(log N) regret for pure exploration

**Theorem 2:** SH optimal under strong hierarchical assumptions

---

## Hyperband: Extension to HPO

### Challenge: Variable Configuration Costs

**Page reference:** Section 4, p. 8-10

**Problem:** Different ML models have different training costs

**Example:** Deep nets expensive, linear models cheap

**Solution:** Treat each model as separate bandit problem

### Algorithm

**Page reference:** Section 4.1, p. 10-13

**Overview:**
1. **Sample** random configurations
2. **Run** SH for each configuration separately
3. **Return** best configuration found

**Pseudocode:**
```
Input: max_budget B, reduction rate η

for s = 1 to S:
    n = ⌈N·η^s⌉
    for i = 1 to n:
        Sample configuration x_i
        budget = B/η^s
        Evaluate x_i for budget epochs
        Keep track of best performance
    Select top 1/η configurations
```

---

## Our Implementation

**Pruning strategy:** `"hyperband"` in `optimize()`

**Configuration:**
```python
from optuna.pruners import HyperbandPruner

study = optuna.create_study(
    ...,
    pruner=HyperbandPruner(
        min_resource=1,      # Minimum budget (e.g., 1 epoch)
        max_resource=81,     # Maximum budget (e.g., 81 epochs)
        reduction_factor=3,  # η parameter (default 3 or 4)
    )
)
```

**Integration:**
- Works with `PeriodicValidationCallback` for validation-based pruning
- Budget measured in terms of validation checkpoints or epochs
- Compatible with all sampler presets

**Key parameters:**
- `min_resource`: Smallest budget (e.g., 10% of max epochs)
- `max_resource`: Full budget (e.g., all training epochs)
- `reduction_factor`: η (typically 3 or 4)

---

## Practical Considerations

### Budget Definition

**Page reference:** Section 5, p. 13-15

**Common budget types:**
- **Training epochs:** Number of training iterations
- **Data samples:** Number of training examples
- **Validation checkpoints:** Number of validation evaluations

**Our implementation:** Budget measured by validation checkpoint count

### Reduction Factor (η)

**Page reference:** Section 5.1, p. 14

**Default values:** 
- η = 4 (standard)
- η = 3 (more aggressive, less expensive)

**Trade-off:**
- Larger η: Faster but may discard good configurations early
- Smaller η: Slower but more thorough

### Resource Allocation

**Page reference:** Section 5.2, p. 15-17

**Challenge:** How to distribute resources across brackets?

**Strategy 1:** Fixed budget per bracket (synchronous)
**Strategy 2:** Asynchronous resource allocation (ASHA)

**Our implementation:** Asynchronous via Optuna's distributed optimization

---

## Comparison to Other Methods

| Method | Exploration | Sample efficiency | Parallelization |
|--------|-------------|-------------------|----------------|
| **Random search** | Uniform | Poor | Easy |
| **Grid search** | Structured | Very poor | Easy |
| **Bayesian optimization** | Focused | Good | Difficult |
| **Hyperband** | Adaptive | Excellent | Easy |

**Page reference:** Section 6, p. 17-22 (empirical comparison)

---

## Edge Cases and Limitations

### Very Expensive Configurations

**Challenge:** Some configurations may be too expensive even at minimum budget

**Solution:** Set `min_resource` appropriately or use budget constraints

**Our implementation:** `max_param_count` and `max_memory_mb` budget constraints

### Early Stopping Criteria

**Challenge:** How to decide when to stop a trial?

**Solution:** Validation-based pruning with `PeriodicValidationCallback`

**Our implementation:** Pruning based on intermediate validation metrics

### Non-Hierarchical Budgets

**Challenge:** Hyperband assumes geometric progression of budgets

**Solution:** Adjust rungs to match available budget levels

---

## Hyperband Variants

### BOHB (Bayesian Optimization + Hyperband)

**Page reference:** Section 7.3, p. 25-27

**Innovation:** Combine TPE for configuration sampling with Hyperband for resource allocation

**Benefits:** Best of both worlds (TPE + Hyperband)

**Our implementation:** Not currently implemented; could be added as `"bohb"` sampler preset

### ASHA (Asynchronous Successive Halving)

**Page reference:** Section 7.2, p. 24-25

**Innovation:** Remove synchronization requirement

**Benefits:** Better for parallel/distributed optimization

**Our implementation:** Hyperband is inherently asynchronous via Optuna

---

## Theoretical Results

### Regret Bounds

**Page reference:** Section 3.2, p. 7-8

**Theorem 3:** Hyperband achieves O(log N) regret under certain assumptions

**Conditions:**
- Hierarchical structure in configuration space
- Sufficient budget for exploration

### Convergence Rate

**Page reference:** Section 4.2, p. 11-13

**Proposition 1:** Successive halving optimal for pure exploration

**Proposition 2:** Hyperband consistent with successive halving

---

## Intentional Deviations

None. bayesflow-hpo uses Hyperband as specified:
- Successive halving with reduction factor η = 3 or 4
- Budget measured by validation checkpoints
- Integration with validation-based pruning
- Asynchronous evaluation via Optuna

**Key difference:** bayesflow-hpo specializes for NPE models with:
- Fixed validation datasets (not traditional epoch-based budgets)
- Per-condition validation metrics
- Custom pruning strategies (dominance, MO-SHA)

---

## Related References

- **ASHA (Li et al., 2016):** Asynchronous variant — See Li et al. (2016), JMLR 17(142)
- **BOHB (Falkner et al., 2018):** Bayesian optimization + Hyperband
- **MO-ASHA (Schmucker et al., 2021):** Multi-objective variant — See `schmucker2021_moasha.md`

---

## Key Algorithms

### Algorithm 1: Successive Halving (p. 6)

```
Input: N configurations, budget B, reduction rate η

for s = 1 to S:
    n = ⌈N·η^s⌉
    budget = B/η^s
    For each i = 1,...,n:
        Evaluate configuration i for budget
    Select top 1/η configurations by performance
```

### Algorithm 2: Hyperband (p. 10)

```
Input: max budget B, reduction rate η

for s = 1 to S:
    n = ⌈N·η^s⌉
    For each i = 1,...,n:
        Sample configuration x_i
        Evaluate x_i for budget = B/η^s
    Select top 1/η configurations
Return best configuration across all rungs
```

---

## Practical Tips

### Parameter Settings

**Reduction factor η:**
- Default: 4 (standard)
- Aggressive: 3 (faster, but may discard good configs)
- Conservative: 5 (slower, more thorough)

**Max budget:** Set based on available resources
- Too small: Insufficient training
- Too large: Wastes resources on poor configs

**Min resource:** Smallest meaningful budget
- For neural nets: 1-10% of max epochs
- For tree models: 100-1000 samples

### Convergence Detection

**Early stopping:** Stop if performance plateaus

**Validation frequency:** Validate at each rung change

**Our implementation:** `PeriodicValidationCallback` handles validation

---

## Applications in bayesflow-hpo

### Validation-Based Pruning

**Use case:** Early stopping based on intermediate validation metrics

**Integration:**
```python
PeriodicValidationCallback(
    validation_data=validation_dataset,
    pruning_strategy="hyperband",
    n_startup_trials=10,
    max_n_checks=5,  # Check 5 times during training
    ...
)
```

### Budget-Aware Trial Management

**Feature:** Trial counting excludes budget-rejected trials

**Rationale:** Hyperband prunes aggressively; shouldn't count toward trial budget

**Implementation:** `max_total_trials` only counts non-rejected trials

---

## Historical Significance

**Impact:** Hyperband popularized bandit-based HPO

**Influence:** Led to many variants (BOHB, ASHA, MO-ASHA)

**Citation:** Highly cited (5000+ citations)

**Legacy:** Foundation for modern multi-fidelity HPO methods
