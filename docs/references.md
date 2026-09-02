# References

Checked against the OpenAlex API, with version exceptions documented below.
APA 7 format.

## Coverage Matrix

Feature implementations and their backing references.

### Optimization

| Feature | Module | Reference |
|---------|--------|-----------|
| Optuna framework | `optimization/study.py` | Akiba et al. (2019) |
| TPE sampler preset | `optimization/study.py` | Bergstra et al. (2011) |
| BoTorch / GP sampler preset | `optimization/study.py` | Balandat et al. (2020) |
| qEHVI acquisition | `optimization/study.py` | Daulton et al. (2020) |
| qNEHVI acquisition | `optimization/study.py` | Daulton et al. (2021) |
| NSGA-II sampler preset | `optimization/study.py` | Deb et al. (2002) |
| NSGA-III sampler preset | `optimization/study.py` | Deb & Jain (2014) |
| HPO foundations and best practices | overall | Bischl et al. (2023) |
| Joint batch-size / learning-rate tuning and coupling | `search_spaces/training.py`, `search_spaces/base.py` | Smith et al. (2018); Shallue et al. (2019) |
| Linear learning-rate warmup | `builders/workflow.py` | Goyal et al. (2017) |
| Inverse-square-root schedule with warmup | `builders/workflow.py` | Vaswani et al. (2017) |

### Pruning

| Feature | Module | Reference |
|---------|--------|-----------|
| Dominance-based pruning | `optimization/pruning_strategies.py` | Schmucker et al. (2021), Alg. 1 |
| MO-SHA non-dominated sorting | `optimization/pruning_strategies.py` | Schmucker et al. (2021), Alg. 2 |
| Primary-metric median pruning | `optimization/pruning_strategies.py` | Akiba et al. (2019) |
| Hyperband / Successive Halving | `optimization/study.py` | Li et al. (2018) |
| Non-dominated sorting (shared) | `optimization/pruning_strategies.py` | Deb et al. (2002) |
| Multi-objective fundamentals | `optimization/pruning_strategies.py` | Emmerich & Deutz (2018) |

### Trial Selection

| Feature | Module | Reference |
|---------|--------|-----------|
| Lexicographic-Pareto selection | `results/extraction.py` | Deb et al. (2002) |

### QMC Warm-Up

| Feature | Module | Reference |
|---------|--------|-----------|
| Sobol quasi-random startup | `optimization/study.py` | Sobol' (1967); Joe & Kuo (2008) |

### Validation Metrics

| Feature | Module | Reference |
|---------|--------|-----------|
| SBC rank uniformity tests | `validation/sbc_tests.py` | Talts et al. (2018) |
| SBC rank-based coverage | `validation/registry.py` | Talts et al. (2018) |
| Global C2ST | `validation/c2st.py` | Lopez-Paz & Oquab (2017) |
| L-C2ST (local) | `validation/c2st.py` | Linhart et al. (2023) |
| Correlation versus agreement | `validation/registry.py` | Bland & Altman (1986) |
| Point-summary/loss consistency | `validation/registry.py` | Gneiting (2011) |
| TARP (possible future extension) | documentation only | Lemos et al. (2023) |
| SBI benchmarking | overall | Lueckmann et al. (2021) |

### BayesFlow Diagnostic Wrappers

The following metrics wrap `bf.diagnostics.*` functions. Their methodological
references are provided by the BayesFlow package, not this package:

- `calibration_error` (ECE)
- `rmse`, `nrmse`
- `contraction` (posterior contraction)
- `z_score` (posterior z-score)
- `log_gamma`

---

## References

### Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019)

Optuna: A next-generation hyperparameter optimization framework. In
*Proceedings of the 25th ACM SIGKDD International Conference on Knowledge
Discovery & Data Mining* (pp. 2623--2631).
https://doi.org/10.1145/3292500.3330701

Introduces Optuna, a define-by-run HPO framework with efficient pruning
strategies and versatile architecture for distributed optimization.

### Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson, A. G., & Bakshy, E. (2020)

BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. In
*Advances in Neural Information Processing Systems 33* (pp. 21524--21538).
https://doi.org/10.48550/arXiv.1910.06403

PyTorch-based BO framework using MC acquisition functions and
auto-differentiation.

### Bergstra, J., Bardenet, R., Bengio, Y., & Kegl, B. (2011)

Algorithms for hyper-parameter optimization. In *Advances in Neural
Information Processing Systems 24* (pp. 2546--2554).

Proposes TPE and sequential model-based optimization, handling conditional
hyperparameters. Shows significant improvement over random search.

### Bischl, B., Binder, M., Lang, M., Pielok, T., Richter, J., Coors, S., Thomas, J., Ullmann, T., Becker, M., Boulesteix, A.-L., Deng, D., & Lindauer, M. (2023)

Hyperparameter optimization: Foundations, algorithms, best practices, and
open challenges. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge
Discovery*, *13*(2), e1484. https://doi.org/10.1002/widm.1484

Comprehensive survey of HPO foundations, algorithms, and open challenges.

### Daulton, S., Balandat, M., & Bakshy, E. (2020)

Differentiable expected hypervolume improvement for parallel multi-objective
Bayesian optimization. In *Advances in Neural Information Processing Systems
33* (pp. 9851--9864). https://doi.org/10.48550/arXiv.2006.05078

Extends EHVI to parallel MOO with differentiable MC estimates (qEHVI).

### Daulton, S., Balandat, M., & Bakshy, E. (2021)

Parallel Bayesian optimization of multiple noisy objectives with expected
hypervolume improvement. In *Advances in Neural Information Processing Systems
34* (pp. 2187--2200). https://doi.org/10.48550/arXiv.2105.08195

Introduces qNEHVI, handling observation noise via Bayesian treatment of the
hypervolume improvement criterion.

### Deb, K., & Jain, H. (2014)

An evolutionary many-objective optimization algorithm using
reference-point-based nondominated sorting approach, Part I: Solving problems
with box constraints. *IEEE Transactions on Evolutionary Computation*,
*18*(4), 577--601. https://doi.org/10.1109/TEVC.2013.2281535

Extends NSGA-II to many-objective optimization (4+ objectives) using
reference-point-based selection.

### Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002)

A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE
Transactions on Evolutionary Computation*, *6*(2), 182--197.
https://doi.org/10.1109/4235.996017

Proposes NSGA-II with fast non-dominated sorting at O(MN^2) and
crowding-distance-based selection. Backs our non-dominated sorting,
lexicographic-Pareto selection, and the NSGA-II sampler preset.

### Emmerich, M. T. M., & Deutz, A. H. (2018)

A tutorial on multiobjective optimization: Fundamentals and evolutionary
methods. *Natural Computing*, *17*(3), 585--609.
https://doi.org/10.1007/s11047-018-9685-y

Tutorial on MOO fundamentals: Pareto dominance (Def. 5), non-dominated
sorting (Eqs. 3--4), complexity bounds (Props. 7, 9). Covers NSGA-II,
indicator-based, and decomposition-based approaches.

### Joe, S., & Kuo, F. Y. (2008)

Constructing Sobol sequences with better two-dimensional projections.
*SIAM Journal on Scientific Computing*, *30*(5), 2635--2654.
https://doi.org/10.1137/070709359

Improved direction numbers for Sobol sequences, used by SciPy's
`scipy.stats.qmc.Sobol` (and thus Optuna's `QMCSampler`).

### Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018)

Hyperband: A novel bandit-based approach to hyperparameter optimization.
*Journal of Machine Learning Research*, *18*(185), 1--52.

Combines random search with adaptive resource allocation via Successive
Halving. Default eta=3 (Section 3.6). Section 6 suggests Sobol sampling
as a promising extension.

### Linhart, J., Gramfort, A., & Rodrigues, P. L. C. (2023)

L-C2ST: Local diagnostics for posterior approximations in simulation-based
inference. In *Advances in Neural Information Processing Systems 36*.
https://doi.org/10.48550/arXiv.2306.03580

Reference-free local posterior diagnostic using joint samples p(theta, x).
Implementation: `bayesflow_hpo.validation.c2st.lc2st()`.

### Modrák, M., Moon, A. H., Kim, S., Bürkner, P.-C., Huurre, N., Faltejsková, K., Gelman, A., & Vehtari, A. (2025)

Simulation-based calibration checking for Bayesian computation: The choice of
test quantities shapes sensitivity. *Bayesian Analysis, 20*(2), 461--488.
https://doi.org/10.1214/23-BA1404

Source of the log-gamma calibration statistic, `log(gamma/gamma_null)`, where
`gamma_null` is the 5th percentile of the null distribution under uniformity of
ranks. This is what fixes the metric's **direction**: `log_gamma < 0` rejects
the hypothesis of uniform ranks at the 5% level, so larger is better, and
minimizing it would search for the most miscalibrated model available.
Recorded in `bayesflow_hpo.objectives.METRIC_DIRECTIONS`; the metric itself is
wrapped from BayesFlow in `bayesflow_hpo.validation.registry._bf_log_gamma`.

The same paper backs two limits on what the statistic can do. Rank-based
calibration over *marginal* parameters returns a clean result for a posterior
that ignores the data entirely, so switching the objective from
`calibration_error` to `log_gamma` improves sensitivity without closing that
blind spot -- detecting it requires test quantities that are functions of data
*and* parameters. And averaging a calibration statistic across conditions lets
opposite failures cancel, which is why per-corner reduction belongs to the
caller rather than the objective.

Bibliographic record verified against OpenAlex work `W4388952075`: eight
authors as listed, *Bayesian Analysis* volume 20, issue 2, pages 461--488, DOI
`10.1214/23-BA1404`. OpenAlex reports `publication_year` 2023 from the
2023-11-23 online-first posting; the issue itself is dated June 2025, which is
the year cited here and the one BayesFlow's own `calibration_log_gamma`
docstring uses.

### Lopez-Paz, D., & Oquab, M. (2017)

Revisiting classifier two-sample tests. In *Proceedings of the 5th
International Conference on Learning Representations (ICLR 2017)*.
https://arxiv.org/abs/1610.06545

Binary classifier as two-sample test: label P positive, Q negative, test
whether accuracy exceeds chance. Implementation:
`bayesflow_hpo.validation.c2st.global_c2st()`.

### Bland, J. M., & Altman, D. G. (1986)

Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, *327*(8476), 307–310. https://doi.org/10.1016/S0140-6736(86)90837-8

Shows that Pearson correlation measures linear association rather than agreement and is insensitive to changes in scale. A corrected full-text reproduction is available at https://www-users.york.ac.uk/~mb55/meas/ba.htm. OpenAlex work `W2015795623`.

### Gneiting, T. (2011)

Making and evaluating point forecasts. *Journal of the American Statistical Association*, *106*(494), 746–762. https://doi.org/10.1198/jasa.2011.r10138

Establishes that point summaries must be evaluated with a consistent loss: the mean is optimal for squared error, while the median is optimal for absolute error. OpenAlex work `W2075965721`.

### Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023)

Sampling-based accuracy testing of posterior estimators for general inference. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 19256–19273). PMLR. https://doi.org/10.48550/arXiv.2302.03026

Introduces Tests of Accuracy with Random Points (TARP) for joint posterior coverage testing using posterior samples. Tracked as a possible future extension; not currently implemented. OpenAlex work `W4319453761`.

### Lueckmann, J.-M., Boelts, J., Greenberg, D. S., Goncalves, P. J., & Macke, J. H. (2021)

Benchmarking simulation-based inference. In *Proceedings of the 24th
International Conference on Artificial Intelligence and Statistics*, PMLR
130, pp. 343--351. https://proceedings.mlr.press/v130/lueckmann21a.html

Public benchmark for SBI algorithms. Key finding: choice of performance
metric is critical; no uniformly best algorithm exists.

### Schmucker, R., Donini, M., Zafar, M. B., Salinas, D., & Archambeau, C. (2021)

Multi-objective asynchronous successive halving. *arXiv preprint*.
https://doi.org/10.48550/arxiv.2106.12639

Extends ASHA to multi-objective settings. Algorithm 1: dominance-based
promotion. Algorithm 2: non-dominated sorting + bottom-fraction pruning.
Key finding: dominance-based approaches consistently outperform
scalarization-based ones.

### Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017)

*Accurate, large minibatch SGD: Training ImageNet in 1 hour* [Preprint].
arXiv. https://doi.org/10.48550/arXiv.1706.02677

Introduces gradual learning-rate warmup to avoid early optimization problems
when training with aggressive learning rates. OpenAlex work
`W2622263826`. Together with the Keras `CosineDecay` documentation, this backs
the optional fixed-budget warmup.

### Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2018)

Don't decay the learning rate, increase the batch size. In *Proceedings of the
6th International Conference on Learning Representations (ICLR 2018)*.
https://doi.org/10.48550/arXiv.1711.00489

Shows that the gradient-noise scale couples learning rate and batch size and
supports reparameterizing known resource relationships instead of tuning
redundant coordinates independently.

### Shallue, C. J., Lee, J., Antognini, J. M., Sohl-Dickstein, J., Frostig, R., & Dahl, G. E. (2019)

Measuring the effects of data parallelism on neural network training. *Journal
of Machine Learning Research, 20*(112), 1–49.
https://www.jmlr.org/papers/v20/18-789.html

Shows that batch-size effects and suitable metaparameter settings vary greatly
between workloads, supporting joint batch-size and learning-rate exploration
instead of a fixed package-wide batch size. OpenAlex work `W2900167092` maps
the preprint record to the cited 2019 JMLR version; full text was verified via
the JMLR article PDF.

### Sobol', I. M. (1967)

On the distribution of points in a cube and the approximate evaluation of
integrals. *USSR Computational Mathematics and Mathematical Physics*,
*7*(4), 86--112. https://doi.org/10.1016/0041-5553(67)90144-9

Seminal paper introducing Sobol low-discrepancy sequences for numerical
integration and optimization. Backs the QMC warm-up feature
(`qmc_startup_trials`).

### Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018)

Validating Bayesian inference algorithms with simulation-based calibration.
*arXiv preprint*. https://doi.org/10.48550/arXiv.1804.06788

Introduces SBC: verify that posterior rank statistics are uniformly
distributed. Backs `sbc_tests.py` and the SBC rank-based coverage metrics.

### Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017)

Attention is all you need. In *Advances in Neural Information Processing
Systems 30* (pp. 5998--6008). https://doi.org/10.48550/arXiv.1706.03762

Section 5.3 defines linear learning-rate warmup followed by decay proportional
to the inverse square root of the optimizer step. This backs the horizon-free
schedule used by `open_ended` training mode. OpenAlex candidate
`W2626778328` is unrelated (a 2025 record with DOI `10.65215/2q58a426`) and is
therefore not used as the bibliographic record. The cited 2017 arXiv DOI,
authors, and Section 5.3 were verified directly against the full text because
OpenAlex did not return a correctly mapped work for that DOI.
