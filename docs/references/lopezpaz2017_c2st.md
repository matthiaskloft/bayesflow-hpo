# López-Paz & Oquab (2017) — Global C2ST

**Reference:** López-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample tests. In *Proceedings of the 5th International Conference on Learning Representations (ICLR 2017)*. https://arxiv.org/abs/1610.06545

**Relevance:** Backs `validation/c2st.py` (global_c2st function) for comparing posterior distributions using classifier-based two-sample tests.

---

## Key Method: Classifier Two-Sample Tests (C2ST)

### Intuition (Section 3)

**Goal:** Test whether two samples S_P ~ P^n and S_Q ~ Q^m are from same distribution

**Method:** Train binary classifier to distinguish samples:
- Label P samples as positive
- Label Q samples as negative
- If P = Q, classifier accuracy ≈ chance (0.5)
- If P ≠ Q, classifier accuracy > chance

**Benefits:**
- Learn representation automatically
- Interpretable test statistic (classification accuracy)
- Simple null distribution
- Can interpret where P and Q differ

---

## Algorithm (Section 3)

**Page reference:** Section 3, Equations

```
Input: S_P = {x₁, ..., x_n}, S_Q = {y₁, ..., y_m}

1. Construct dataset:
   D = {(x_i, 0)}_i=1^n ∪ {(y_i, 1)}_i=1^m

2. Shuffle and split into train/test:
   D = D_tr ∪ D_te

3. Train binary classifier f: X → [0,1] on D_tr
   (f(z) estimates p(l=1|z))

4. Compute test statistic (classification accuracy):
   t̂ = (1/n_te) Σ_{(z_i,l_i)∈D_te} I[I(f(z_i) > 0.5) = l_i]

5. Compute p-value from null distribution
```

---

## Null Distribution (Section 3.1)

**Page reference:** Section 3.1

### Under H₀: P = Q

Each test term is independent Bernoulli(p) with p = 0.5 (chance-level classification)

**Exact distribution:** n_te · t̂ ~ Binomial(n_te, 0.5)

**Asymptotic approximation (CLT):** t̂ ~ Normal(0.5, 1/(4n_te))

**P-value computation:**
```python
# Use binomial test
from scipy.stats import binomtest
p_value = binomtest(int(n_te * accuracy), n_te, p=0.5, alternative='greater')

# Or normal approximation
from scipy.stats import norm
z = (accuracy - 0.5) / np.sqrt(0.25 / n_te)
p_value = 1 - norm.cdf(z)
```

---

## Testing Power (Section 3.2)

**Page reference:** Section 3.2, Theorem 1

### Under H₁: P ≠ Q

**Effect size:** ε = accuracy - 0.5 (departure from chance)

**Approximate power:**
```
Power(α, ε, n_te) = Φ((ε√n_te - Φ⁻¹(1-α)/2) / √(0.25 - ε²))
```

**Trade-off:** Maximize test accuracy vs. maximize test set size
- Simple classifiers: Lower accuracy but larger n_te (less training data needed)
- Flexible classifiers: Higher accuracy but smaller n_te (more training data needed)

**Optimal order:** O(n^(-1/2)) for fixed dimensionality, matching optimal multi-dimensional tests

---

## Our Implementation

**Function:** `global_c2st(reference_samples, test_samples, ..., return_p_value=True)`

**Parameters:**
- `reference_samples`: Samples from reference distribution P
- `test_samples`: Samples from test distribution Q  
- `num_bootstrap`: Number of classifier trainings for uncertainty estimation
- `train_size`: Proportion for training (default 0.8)
- `random_state`: RNG seed

**Returns:**
- `statistic`: Mean classification accuracy across bootstrap runs
- `p_value`: Proportion of runs achieving statistic ≥ observed

**Classifier:** sklearn MLPClassifier with default hyperparameters
- Single hidden layer with 100 neurons
- Adam optimizer
- Max 1000 iterations

---

## Edge Cases and Recommendations

**Sample size:** Minimum n_te ≥ 20 for normal approximation validity

**Class imbalance:** Ensure equal n and m for P and Q samples

**High-dimensional data:** C2ST benefits from representation learning (unlike kernel tests requiring manual feature engineering)

**Multiple comparisons:** Apply Bonferroni correction when testing multiple parameter pairs

**Interpretability:**
- Use classifier feature importance to identify discriminating dimensions
- Use predictive uncertainty (prediction probabilities) to assess confidence

---

## Intentional Deviations

None. bayesflow-hpo's `global_c2st()` follows paper specification:
- Binary classifier trained on labeled samples
- Classification accuracy as test statistic
- Binomial/null distribution for p-value computation
- Bootstrap ensemble for uncertainty quantification

---

## Related References

- **L-C2ST (local):** See `linhart2023_lc2st.md` for joint sample (θ, x) testing
- **SBC comparison:** C2ST used alongside SBC for posterior validation

---

## Fulltext

See `fulltexts/lopezpaz2017_c2st/` (arXiv source extracted)
