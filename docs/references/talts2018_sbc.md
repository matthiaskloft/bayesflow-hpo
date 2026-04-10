# Talts et al. (2018) — Simulation-Based Calibration

**Reference:** Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. *arXiv preprint* arXiv:1804.06788.

**Relevance:** Backs `validation/sbc_tests.py` (KS test, chi-squared test) and `validation/registry.py` (SBC rank-based coverage metrics).

---

## Key Method: Rank Statistics

### Theorem 1 (Line 254-256)
Let θ̃ ~ π(θ), ỹ ~ π(y|θ̃), and {θ₁,...,θ_L} ~ π(θ|ỹ) for any joint distribution π(y,θ). The rank statistic of any one-dimensional random variable over θ is uniformly distributed over the integers [0, L].

**Implementation note:** This is the fundamental theorem backing SBC. Ranks are uniform iff posterior inference is correct.

### Rank Statistic Formula (Line 249-252)
```
r({f(θ₁),...,f(θ_L)}, f(θ̃)) = Σ_{l=1}^L I[f(θ_l) < f(θ̃)] ∈ [0, L]
```

**Page reference:** Section 4.1, p. 6

**Implementation details:**
- Compute for each scalar parameter or function of parameters
- Returns integer rank from 0 to L (number of posterior samples)
- Uniform distribution under correct inference

---

## Algorithm 1: Basic SBC (Line 262-275)

**Page reference:** Section 4.1, p. 6

```
Initialize histogram with bins at 0, ..., L
for n in N:
    Draw prior sample: θ̃ ~ π(θ)
    Draw simulated data: ỹ ~ π(y|θ̃)
    Draw posterior samples: {θ₁,...,θ_L} ~ π(θ|ỹ)
    for each 1D variable f:
        Compute rank statistic r({f(θ₁),...,f(θ_L)}, f(θ̃))
        Increment histogram bin at rank r
Analyze histogram for uniformity
```

**Edge-case handling:**
- For small N, consider binning neighboring ranks (e.g., L/2 bins)
- Keep N/B ≈ 20 for good variance-resolution tradeoff (p. 6)
- Choose L+1 divisible by power of 2 for easier re-binning

---

## Algorithm 2: SBC for MCMC (Line 443-458)

**Page reference:** Section 5.1, p. 10

**Key addition:** Thinning based on effective sample size (N_eff)

```
if N_eff[f] < L:
    Rerun MCMC for L'·L/N_eff[f] iterations
    Thin uniformly to L states
```

**Thinning strategy (Line 431-432):**
- Thin by ⌈L/N_eff[f]⌉ to reduce autocorrelation
- For antithetic chains (N_eff > N), first thin by 2
- Compute N_eff at empirical quantiles of f(θ) (e.g., 19 equispaced)

**Implementation note:** bayesflow-hpo uses independent posterior samples from neural networks, so thinning is typically unnecessary.

---

## Coverage Intervals (Line 283-286)

**Page reference:** Section 4.1, p. 6

**99% expected variation band:**
- Vertical extent: 0.005 to 0.995 percentile of Binomial(N, (L+1)^(-1))
- Under uniformity, expect 1 in 100 bins to fall outside this band

**Implementation:**
- Credible intervals [α/2, 1-α/2] for coverage checking (Section 4)
- Rank-based coverage is equivalent to checking credible interval calibration

---

## Rank Normalization

**Continuity correction (Line 184-186):**
- For CDF values at 0 or 1, add offset 0.5 (Blom, 1958)
- SBC avoids this entirely by using rank histograms

**Our implementation uses:** (r + 0.5) / (L + 1) normalization
- `# continuity correction, standard practice`
- Maps discrete ranks to continuous [0, 1] interval

---

## Diagnostic Interpretation (Line 300-352)

**Page reference:** Section 4.2, p. 7-8

| Histogram shape | Diagnosis |
|----------------|-----------|
| Uniform | Correct inference |
| Spikes at boundaries (0, L) | Autocorrelation in posterior samples |
| ∩-shape (convex) | Overdispersed posteriors (wider than true) |
| ∪-shape (concave) | Under-dispersed posteriors (narrower than true) |
| Asymmetric left-skewed | Posterior biased high |
| Asymmetric right-skewed | Posterior biased low |

---

## Statistical Tests

**KS test:** Kolmogorov-Smirnov test for uniformity of ranks
- Tests ECDF against uniform CDF
- Sensitive to small deviations (Line 484-488)

**Chi-squared test:** Bin-based goodness-of-fit
- Compare observed bin counts to expected Binomial(N, 1/(L+1))
- More robust for small sample sizes

---

## Intentional Deviations

None. bayesflow-hpo implements SBC exactly as specified:
- Rank computation follows Theorem 1
- Coverage uses Binomial reference distribution
- KS and chi-squared tests match paper recommendations

---

## Fulltext

See `fulltexts/talts2018.md` for complete paper text.
