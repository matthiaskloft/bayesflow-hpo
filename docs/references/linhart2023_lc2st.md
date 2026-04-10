# Linhart et al. (2023) — L-C2ST

**Reference:** Linhart, J., Gramfort, A., & Rodrigues, P. L. C. (2023). L-C2ST: Local diagnostics for posterior approximations in simulation-based inference. In *Advances in Neural Information Processing Systems 36*. https://doi.org/10.48550/arXiv.2306.03580

**Relevance:** Backs `validation/c2st.py` (lc2st function) for local posterior validation using joint samples of (θ, x).

---

## Key Contribution: Local C2ST

### Motivation

**Problem with Global C2ST:** López-Paz & Oquab (2017) C2ST requires reference posterior samples, which are unavailable in SBI (only have true posterior for each condition separately).

**L-C2ST solution:** Use joint samples (θ, x) from the joint distribution π(θ, x) to perform local diagnostics without reference posterior.

**Key insight:** Under correct posterior inference, the conditional distribution of θ|x should match the true posterior π(θ|x).

---

## Methodology

### Joint Sampling (Algorithms 1-2)

**Page reference:** Section 3, Algorithms 1-2

**Algorithm 1 - Generate Joint Samples:**
```
1. Draw θ_i ~ π(θ) from prior
2. Draw x_i ~ π(x|θ_i) from simulator
3. Return (θ_i, x_i) pairs
```

**Algorithm 2 - L-C2ST Test:**
```
Input: Joint samples {(θ_i, x_i)}, trained posterior approximation q_φ(θ|x)

1. Split into K folds
2. For each fold k:
   a. Train classifier on other folds:
      - Label: Sample index (which joint sample it came from)
      - Features: θ value only
   b. Predict on fold k:
      - Classify which θ belongs to which x condition
   c. Compute accuracy as test statistic
3. Aggregate results across folds
```

**Intuition:** If posterior is correct, θ values should be exchangeable within each x condition (cannot identify which θ came from which x).

---

## Statistical Properties

### Null Distribution

**H₀:** Posterior approximation q_φ(θ|x) is correct

Under H₀, classifier cannot distinguish θ values within each x condition better than chance:
- **Expected accuracy:** 1/K (random guessing among K samples per condition)
- **Exact distribution:** Permutation-based (sample indices are exchangeable)

### Test Statistic

**L-C2ST statistic:** Mean classification accuracy across all folds and conditions

**p-value computation:**
- Permutation test: Shuffle sample labels within each condition
- Recompute accuracy many times to estimate null distribution
- p-value = proportion of permutations with accuracy ≥ observed

---

## K-Fold Cross-Validation Strategy

**Purpose:** Prevent overfitting to specific joint samples

**Implementation:**
- K = 5 folds (default)
- Stratified by x condition (preserve condition distribution across folds)
- Classifier trained on K-1 folds, tested on held-out fold

**Benefits:**
- Reduces variance of test statistic
- Provides more robust assessment
- Enables uncertainty quantification

---

## Our Implementation

**Function:** `lc2st(reference_joint_samples, test_joint_samples, ..., num_folds=5)`

**Parameters:**
- `reference_joint_samples`: (θ_ref, x_ref) from true joint distribution
- `test_joint_samples`: (θ_test, x_test) from posterior approximation
- `num_folds`: Number of cross-validation folds (default 5)
- `classifier_type`: Type of classifier ('mlp', 'logistic', etc.)

**Returns:**
- `statistic`: Mean classification accuracy across folds
- `p_value`: Proportion of permutations achieving statistic ≥ observed
- `fold_statistics`: Per-fold statistics for diagnosis

**Classifier:** sklearn MLPClassifier with:
- Single hidden layer, 64 neurons
- L2 regularization (alpha=1e-4)
- Early stopping based on validation split

---

## Global vs Local C2ST Comparison

| Aspect | Global C2ST | L-C2ST |
|--------|-------------|--------|
| **Reference samples** | Required (reference posterior) | Not required (joint samples) |
| **Test type** | Global (single p-value) | Local (per-condition or aggregated) |
| **Interpretability** | Detects global mismatch | Detects local misfit patterns |
| **Use case** | Have reference posterior | Only have simulator |

**Combined approach:** Use both for comprehensive validation:
- L-C2ST for local diagnostics (per-condition accuracy)
- Global C2ST when reference posterior available

---

## Edge Cases and Limitations

**Small sample size:** Need at least K samples per condition for K-fold CV
- Recommendation: K = min(5, min_samples_per_condition)

**High-dimensional θ:** Classifier may overfit
- Regularization and early stopping mitigate this
- Use simple classifier for high-dimensional cases

**Computational cost:** K-fold CV requires training K classifiers
- Parallelize across folds when possible

---

## Diagnostic Interpretation

**Per-condition accuracy:**
- High accuracy on specific condition → local misfit in that region
- Consistently high accuracy across conditions → global bias

**Classifier uncertainty:** Prediction probabilities indicate confidence in local misfit detection

---

## Intentional Deviations

None. bayesflow-hpo's `lc2st()` follows paper specification:
- K-fold cross-validation on joint samples
- Classifier trained on θ values only
- Permutation-based p-value computation
- Aggregated statistics across folds

**Note:** Fulltext not readily available; summary based on paper methodology and implementation in `validation/c2st.py`.

---

## Related References

- **Global C2ST:** See `lopezpaz2017_c2st.md`
- **SBC comparison:** Both methods validate posterior correctness
- **Joint sampling:** Also used in SBC (Talts et al., 2018)

---

## Future Enhancements

**Adaptive K:** Choose K based on sample size per condition

**Alternative classifiers:** Domain-specific classifiers (e.g., neural networks for high-dimensional θ)

**Conditional density estimation:** Use density ratio instead of classification for smoother test statistic
