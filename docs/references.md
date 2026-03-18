# References

Verified via OpenAlex API and web search. APA 7 format.

## All References

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
  next-generation hyperparameter optimization framework. In *Proceedings of
  the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data
  Mining* (pp. 2623–2631). https://doi.org/10.1145/3292500.3330701

Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson,
  A. G., & Bakshy, E. (2020). BoTorch: A framework for efficient Monte-Carlo
  Bayesian optimization. In *Advances in Neural Information Processing Systems
  33* (pp. 21524–21538). https://doi.org/10.48550/arXiv.1910.06403

Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for
  hyper-parameter optimization. In *Advances in Neural Information Processing
  Systems 24* (pp. 2546–2554).

Daulton, S., Balandat, M., & Bakshy, E. (2020). Differentiable expected
  hypervolume improvement for parallel multi-objective Bayesian optimization.
  In *Advances in Neural Information Processing Systems 33* (pp. 9851–9864).
  https://doi.org/10.48550/arXiv.2006.05078

Daulton, S., Balandat, M., & Bakshy, E. (2021). Parallel Bayesian
  optimization of multiple noisy objectives with expected hypervolume
  improvement. In *Advances in Neural Information Processing Systems 34*
  (pp. 2187–2200). https://doi.org/10.48550/arXiv.2105.08195

Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization
  algorithm using reference-point-based nondominated sorting approach, Part I:
  Solving problems with box constraints. *IEEE Transactions on Evolutionary
  Computation*, *18*(4), 577–601. https://doi.org/10.1109/TEVC.2013.2281535

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
  multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on
  Evolutionary Computation*, *6*(2), 182–197.
  https://doi.org/10.1109/4235.996017

Linhart, J., Gramfort, A., & Rodrigues, P. L. C. (2023). L-C2ST: Local
  diagnostics for posterior approximations in simulation-based inference. In
  *Advances in Neural Information Processing Systems 36*.
  https://doi.org/10.48550/arXiv.2306.03580

López-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample tests.
  In *Proceedings of the 5th International Conference on Learning
  Representations (ICLR 2017)*. https://arxiv.org/abs/1610.06545

Lueckmann, J.-M., Boelts, J., Greenberg, D. S., Gonçalves, P. J., & Macke,
  J. H. (2021). Benchmarking simulation-based inference. In *Proceedings of
  the 24th International Conference on Artificial Intelligence and Statistics*,
  PMLR 130, pp. 343–351.
  https://proceedings.mlr.press/v130/lueckmann21a.html

## Summaries

### Akiba et al. (2019) — Optuna
*Source: abstract (arXiv 1907.10902)*

Introduces Optuna, a hyperparameter optimization framework built on a
define-by-run API that constructs search spaces dynamically. Key features:
efficient search and pruning strategies, and a versatile architecture
deployable for distributed optimization. First framework to implement the
define-by-run paradigm for HPO.

### Balandat et al. (2020) — BoTorch
*Source: abstract (arXiv 1910.06403)*

Presents BoTorch, a PyTorch-based framework for Bayesian optimization using
Monte-Carlo acquisition functions, auto-differentiation, and variance
reduction. Its modular design allows flexible specification of probabilistic
models and custom acquisition functions. Includes a novel "one-shot"
Knowledge Gradient formulation. Demonstrates improved sample efficiency
over other BO libraries.

### Bergstra et al. (2011) — TPE
*Source: abstract (NeurIPS 2011 proceedings)*

Proposes Tree-structured Parzen Estimator (TPE) and sequential model-based
optimization for hyperparameter tuning of neural networks and deep belief
networks. Introduces techniques for response surface models that handle
conditional hyperparameters (where some parameters become irrelevant given
values of others). Shows that sequential methods significantly outperform
random search on challenging DBN problems.

### Daulton et al. (2020) — qEHVI
*Source: abstract (arXiv 2006.05078)*

Extends Expected Hypervolume Improvement (EHVI) to parallel multi-objective
Bayesian optimization with exact, differentiable Monte-Carlo estimates.
Unlike prior EHVI methods lacking analytic gradients, qEHVI enables
efficient first-order optimization of the acquisition function via
auto-differentiation. Outperforms state-of-the-art multi-objective BO
methods while requiring substantially less computation.

### Daulton et al. (2021) — qNEHVI
*Source: abstract (arXiv 2105.08195)*

Introduces Noisy Expected Hypervolume Improvement (NEHVI) and its parallel
variant qNEHVI, which handles observation noise by applying Bayesian
treatment to the hypervolume improvement criterion. Reduces computational
complexity from exponential to polynomial in batch size. One-step
Bayes-optimal for hypervolume maximization in both noisy and noiseless
settings. Substantially more robust under observation noise than existing
approaches.

### Deb et al. (2002) — NSGA-II
*Source: abstract (IEEE TEVC, vol. 6, no. 2)*

Proposes NSGA-II, addressing three criticisms of prior multi-objective
evolutionary algorithms: O(MN³) complexity, non-elitism, and the need for
a sharing parameter. Introduces fast non-dominated sorting at O(MN²) and a
crowding-distance-based selection operator that combines parent and offspring
populations. Finds better spread and convergence on the Pareto front than
PAES and SPEA. Extends dominance for constrained multi-objective problems.

### Deb & Jain (2014) — NSGA-III
*Source: abstract (IEEE TEVC, vol. 18, no. 4)*

Extends NSGA-II to many-objective optimization (4+ objectives) using
reference-point-based nondominated sorting. Population members are selected
based on proximity to supplied reference points, maintaining diversity on
high-dimensional Pareto fronts where crowding distance degrades. Evaluated
on problems with 3–15 objectives. Part I covers box-constrained problems;
Part II (sequel) addresses general constraints.

### Linhart et al. (2023) — L-C2ST
*Source: abstract (arXiv 2306.03580)*

Introduces L-C2ST, a local diagnostic for posterior approximations in
simulation-based inference. Unlike standard C2ST which evaluates posterior
quality only globally and requires true posterior samples, L-C2ST provides
per-observation diagnostics using only joint samples p(θ, x). For
normalizing-flow-based posteriors, achieves better statistical power and
computational efficiency than standard C2ST. Matches global C2ST
performance and outperforms HPD coverage tests on SBI benchmarks.

### López-Paz & Oquab (2017) — C2ST
*Source: abstract (arXiv 1610.06545)*

Proposes using binary classifiers as two-sample tests: label samples from
P as positive and Q as negative, then test whether classification accuracy
exceeds chance. Advantages: learns data representations on the fly, returns
interpretable test statistics, has a simple null distribution, and reveals
where distributions differ via predictive uncertainty. Applied to GAN
evaluation and causal discovery. In SBI context, requires samples from both
the approximate and true posterior.

### Lueckmann et al. (2021) — sbibm
*Source: abstract (arXiv 2101.04653)*

Establishes a public benchmark for simulation-based inference algorithms,
covering both neural network methods and classical ABC. Key findings: choice
of performance metric is critical; sequential estimation improves sample
efficiency; no uniformly best algorithm exists; even state-of-the-art methods
have substantial room for improvement. Provides practical guidance and an
interactive companion website for exploring results.

## Index by Topic

### Optuna Framework
- Akiba et al. (2019)

### Samplers
- **TPE**: Bergstra et al. (2011)
- **BoTorch / GP-based BO**: Balandat et al. (2020)
- **qEHVI**: Daulton et al. (2020)
- **qNEHVI**: Daulton et al. (2021)
- **NSGA-II**: Deb et al. (2002)
- **NSGA-III**: Deb & Jain (2014)

### Validation Metrics
- **Global C2ST** (requires reference posterior): López-Paz & Oquab (2017)
- **L-C2ST** (reference-free, uses joint samples): Linhart et al. (2023)
- **SBI Benchmarking (sbibm)**: Lueckmann et al. (2021)
