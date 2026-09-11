# References

Checked against the OpenAlex API, with version exceptions documented below.
APA 7 format.

## Audit status (2026-09-11)

Two passes. The first was prompted by three inherited, unchecked citations
found during PR #86, two of which were wrong. The second was a systematic
sweep of every implementation-backing claim in `src/`, this file, and
`docs/references/`.

**Method.** A claim counts as verified only if it was read back against the
full text of the work it cites, or -- for library behaviour -- executed
against the installed version. Metadata was checked separately via the
OpenAlex API. Claims are grouped below by what actually happened to them.

### Corrected (first pass)

| Claim | Was | Is |
|---|---|---|
| Schmucker et al. (2021) algorithm numbers | "Alg. 1: dominance-based promotion, Alg. 2: non-dominated sorting" | Reversed. Alg. 1 is the selector, Alg. 2 is MO-ASHA |
| Daulton et al. (2021) locator | "Section 3.2, Equation 7" | Sections 5.1-5.2, equations (2)-(3) |
| `pruning_strategies.py` median rule | "Schmucker Alg. 1, per-objective median AND rule" | Alg. 1 states no median rule; the rule is ours |
| gamma discrepancy attribution | Modrák et al. (2025), "Equation 7" | Säilynoja et al. (2022); Modrák et al. adopt it in Sec. 4.1, unnumbered display |
| Optuna API entries | stamped 4.9.0, "the version installed" | re-run and re-stamped 5.0.0; both claims still hold |

### Corrected (second pass)

| Claim | Was | Is | Sites |
|---|---|---|---|
| Talts et al. (2018) rank uniformity | "Theorem 2: ranks uniform **iff** posterior correct" | **Theorem 1** (Sec. 4.1, p. 6), and correctness implies uniformity, not conversely | `registry.py` x2, `sbc_tests.py`, this file |
| Median pruning attribution | Akiba et al. (2019) | The paper's Alg. 1 is the Successive Halving pruner and states no median rule; `MedianPruner` is documented only in the Optuna API reference | `pruning_strategies.py` x2, `validation_callback.py`, matrix |
| Hyperband `eta = 3` locator | "Section 3.6: η=3 convention" | The default is in **Algorithm 1**'s input line; §3.6 recommends "3 or 4" and gives `e ≈ 2.718` as the theoretical optimum | `study.py`, this file |
| Emmerich & Deutz (2018) locators | "non-dominated sorting (Eqs. 3--4), complexity bounds (Props. 7, 9)" | Neither exists as cited; Props. 3--4 and 7--9 are cone-order results. Only "Pareto dominance (Def. 5)" was right | this file |
| Sobol power-of-two warning | attributed to Sobol' (1967) | Attributed to the SciPy `qmc.Sobol` docs, which state it and which Optuna's `QMCSampler` wraps | `study.py` |
| `"dominance"` strategy wording | "MO-ASHA's dominance-based promotion" | Stale: missing the first pass's "the median rule is ours" caveat | `validation_callback.py` |

### Verified, and correct as written

- Deb et al. (2002): the abstract states "a fast nondominated sorting
  approach with O(MN^2) computational complexity", as `_non_dominated_sort()`
  claims.
- Emmerich & Deutz (2018), **Definition 5** (p. 588) is Pareto dominance.
- Li et al. (2018), **Section 6** does propose quasi-random sampling
  ("Quasi-random methods like Sobol or latin hypercube") as an extension.
- Li et al. (2018) via `pruning_strategies.py`: the quoted §3.6 wording
  ("in practice we suggest taking eta to be equal to 3 or 4") is verbatim.
- Talts et al. (2018), **Algorithm 1** is the SBC histogram procedure.
- Linhart et al. (2023), **Algorithms 1--2** are `l`-C2ST training and
  evaluation -- what `lc2st()` implements. Algs. 3--4 are the NF variant,
  which we do not implement.
- Joe & Kuo (2008) supplies the direction numbers for `scipy.stats.qmc.Sobol`
  (cited there as reference [4]).
- Schmucker et al. (2021): `eta = 3` really is in the MO-ASHA Algorithm 2
  header.
- Optuna 5.0.0: `Trial.report()` and `should_prune()` still raise
  `NotImplementedError` on a multi-objective study -- re-executed, not
  assumed. This is what justifies `pruning_strategies.py` existing at all.
- Optuna 5.0.0: `HyperbandPruner` defaults to `reduction_factor=3`, matching
  our preset and Li et al.'s Algorithm 1 default.
- Both Optuna behavioural claims (categorical choice-order identity,
  positional `directions`/`values`), re-executed on 5.0.0.

### Bulk metadata check

Every DOI in this file was resolved against the OpenAlex API. All resolve.
Five entries carry a year differing from OpenAlex's (Balandat 2020/2019, Deb &
Jain 2014/2013, Modrák 2025/2023, Smith 2018/2017): each is a published-version
year cited against a preprint or online-first DOI, which is the intended
convention, not an error. Five entries carry no DOI at all (Bergstra 2011, Li
et al. 2018, Lopez-Paz & Oquab 2017, Lueckmann et al. 2021, Shallue et al.
2019) -- a completeness gap, not a known error.

### Known-unreliable: `docs/references/*.md`

The per-paper summaries in `docs/references/` are **not** covered by this
audit and several are demonstrably wrong. Spot checks found:

- `emmerich2018_moo.md` misidentifies Definitions 6, 7 and 9 (it has them as
  Pareto optimality, hypervolume and unary hypervolume; they are the
  search-space pre-order, the strict component order and the non-trivial
  cone) and places Pareto dominance in Section 2.1 when it is in Section 3.
- `li2018_hyperband.md` labels Algorithm 1 "Successive Halving" with
  pseudocode that is not the paper's; Algorithm 1 is Hyperband. It also cites
  an "ASHA (Li et al., 2016), JMLR 17(142)" that does not correspond to a
  real work.
- `sobol1967_qmc.md` gives section, page and theorem locators against pages
  86--112, but the indexed copy is the Russian original (Zh. Vychisl. Mat.
  Mat. Fiz. 7, pp. 784--802); the English translation carries the 86--112
  pagination. The locators cannot have come from the source at hand.

These files back no code path on their own -- every implementation claim
cites this file or the source docstrings, both of which are now verified --
but they should not be trusted as a secondary source, and are best treated as
drafts pending their own pass.

**Still NOT verified.** Entries here whose description summarises a paper's
general contribution without naming a locator have had metadata checked but
not their substance: Bergstra et al. (2011), Balandat et al. (2020), Daulton
et al. (2020), Deb & Jain (2014), Bischl et al. (2023), Goyal et al. (2017),
Smith et al. (2018), Shallue et al. (2019), Lopez-Paz & Oquab (2017),
Lueckmann et al. (2021), Lemos et al. (2023), Bland & Altman (1986), Gneiting
(2011), Sobol' (1967). Also unverified: the two Optuna issue-tracker pointers
in `study.py`, which needed repository access this audit did not have.

## Coverage Matrix

Feature implementations and their backing references.

### Optimization

| Feature | Module | Reference |
|---------|--------|-----------|
| Optuna framework | `optimization/study.py` | Akiba et al. (2019) |
| End-to-end objective ranking | `tests/test_end_to_end/` | Optuna docs; Deb et al. (2002) |
| End-to-end `log_gamma` direction | `tests/test_end_to_end/` | Sailynoja et al. (2022); Modrak et al. (2025), Sec. 4.1 |
| Objective column ordering | `objectives.py` | Optuna 5.0.0 docs |
| Categorical choice-order identity | `search_spaces/base.py` | Optuna 5.0.0 docs |
| `CanonicalMetricName` type | `validation/registry.py` | PEP 484 |
| `RawScore` / `MinimizeScore` types | `objectives.py` | PEP 484 |
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
| Dominance-based pruning (simplified) | `optimization/pruning_strategies.py` | Schmucker et al. (2021), Alg. 1 selector |
| MO-SHA rung/bottom-fraction pruning | `optimization/pruning_strategies.py` | Schmucker et al. (2021), Alg. 2 (selector: Alg. 1) |
| Primary-metric median pruning | `optimization/pruning_strategies.py` | Optuna docs (`MedianPruner`) |
| Hyperband / Successive Halving | `optimization/study.py` | Li et al. (2018), Alg. 1 |
| Non-dominated sorting (shared) | `optimization/pruning_strategies.py` | Deb et al. (2002) |
| Multi-objective fundamentals | `optimization/pruning_strategies.py` | Emmerich & Deutz (2018), Def. 5 |

### Trial Selection

| Feature | Module | Reference |
|---------|--------|-----------|
| Lexicographic-Pareto selection | `results/extraction.py` | Deb et al. (2002) |

### QMC Warm-Up

| Feature | Module | Reference |
|---------|--------|-----------|
| Sobol quasi-random startup | `optimization/study.py` | Sobol' (1967); Joe & Kuo (2008); SciPy `qmc.Sobol` docs |

### Validation Metrics

| Feature | Module | Reference |
|---------|--------|-----------|
| SBC rank uniformity tests | `validation/sbc_tests.py` | Talts et al. (2018), Thm. 1 |
| SBC rank-based coverage | `validation/registry.py` | Talts et al. (2018), Thm. 1, Sec. 4.1 |
| Global C2ST | `validation/c2st.py` | Lopez-Paz & Oquab (2017) |
| L-C2ST (local) | `validation/c2st.py` | Linhart et al. (2023), Algs. 1--2 |
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
strategies and versatile architecture for distributed optimization. Its
Algorithm 1, "Pruning algorithm based on Successive Halving", is a variant of
ASHA and is the only pruner the paper specifies.

It therefore does **not** back median pruning. `"primary"` mirrors Optuna's
`MedianPruner`, which is a software feature documented in the API reference
and absent from this paper; the coverage matrix cites the docs for it.

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

Tutorial on MOO fundamentals. **Definition 5 (p. 588) is Pareto
dominance** -- "not worse in each of the objectives and better in at least
one" -- which is the rule `_non_dominated_sort()` implements. Definition 8
gives the efficient set and Pareto front. Both sit in Section 3, "Order and
dominance".

(An earlier version of this entry cited "non-dominated sorting (Eqs. 3--4)"
and "complexity bounds (Props. 7, 9)". Neither survives the full text:
Propositions 3--4 and 7--9 belong to the cone-order development, and the
complexity bound we actually rely on is Deb et al.'s, not this tutorial's.
Only the Definition 5 locator was correct.)

### Joe, S., & Kuo, F. Y. (2008)

Constructing Sobol sequences with better two-dimensional projections.
*SIAM Journal on Scientific Computing*, *30*(5), 2635--2654.
https://doi.org/10.1137/070709359

Improved direction numbers for Sobol sequences. Verified against the
SciPy API reference, which cites this paper as its reference [4] for the
direction numbers used by `scipy.stats.qmc.Sobol` (and thus by Optuna's
`QMCSampler`). The same page documents the power-of-two property that
`_maybe_warn_qmc_startup()` warns about: Sobol' points "lose their balance
properties if one uses a sample size that is not a power of 2".

### Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018)

Hyperband: A novel bandit-based approach to hyperparameter optimization.
*Journal of Machine Learning Research*, *18*(185), 1--52.

Combines random search with adaptive resource allocation via Successive
Halving. **Algorithm 1 is Hyperband itself**, and takes the reduction factor
as an input annotated "default eta = 3"; SuccessiveHalving is its inner loop
(lines 3--9). Section 3.6, "Setting eta", recommends eta of 3 *or 4* in
practice and notes the theoretically optimal value is e ~= 2.718 -- so the
default and the recommendation are separate claims with separate locators.
Section 6 does suggest quasi-random sampling as a promising extension:
"Quasi-random methods like Sobol or latin hypercube [...] may improve the
performance of Hyperband by giving better coverage of the search space."

(An earlier version of this entry placed the eta=3 default in Section 3.6.)

### Linhart, J., Gramfort, A., & Rodrigues, P. L. C. (2023)

L-C2ST: Local diagnostics for posterior approximations in simulation-based
inference. In *Advances in Neural Information Processing Systems 36*.
https://doi.org/10.48550/arXiv.2306.03580

Reference-free local posterior diagnostic using joint samples p(theta, x).
Algorithm 1 trains the classifier on joint-distribution data and derives the
null by permutation; Algorithm 2 evaluates the test statistic and p-value for
a given observation. Those two are what
`bayesflow_hpo.validation.c2st.lc2st()` implements. Algorithms 3--4 are the
normalizing-flow variant (l-C2ST-NF), which we do not implement.

### Säilynoja, T., Bürkner, P.-C., & Vehtari, A. (2022)

Graphical test for discrete uniformity and its applications in goodness-of-fit
evaluation and multiple sample comparison. *Statistics and Computing, 32*(2).
https://doi.org/10.1007/s11222-022-10090-6

Introduces the gamma discrepancy: the probability, under uniform ranks, of
observing the most extreme point of the empirical rank CDF, together with
methods for evaluating its null distribution for given `M` and `S`. Modrák et
al. (2025) attribute the statistic to this paper ("This metric was introduced
in a paper by Säilynoja et al. (2022)", Section 4.1) and define the log ratio
against its 5th percentile that BayesFlow's `calibration_log_gamma` reports.

Verified via the OpenAlex API (DOI 10.1007/s11222-022-10090-6; *Statistics and
Computing*, volume 32, issue 2, 2022).

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

Extends ASHA to multi-objective settings. **Algorithm 1 is the
multi-objective selector** (`non_dom_sorting`, `selector_eps_net`,
`selector_nsga_ii`); **Algorithm 2 is MO-ASHA itself** (`mo_asha`, `get_job`,
rung promotion), whose header reads "Data: R, r0, s, eta (default eta = 3)".
Algorithm 2 calls the Algorithm 1 selector as
`mo_selector(rung k, |rung k| / eta)`. Key finding: dominance-based approaches
consistently outperform scalarization-based ones.

(An earlier version of this entry had the two algorithms the other way round.
Corrected against the full text.)

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
distributed. **Theorem 1** (Section 4.1, p. 6) is the result the code relies
on: given exact posterior samples, "the rank statistic of any one-dimensional
random variable over theta is uniformly distributed over the integers [0, L]".
Algorithm 1 is the SBC histogram procedure. Backs `sbc_tests.py` and the SBC
rank-based coverage metrics.

Note the direction of the implication. The theorem says correctness implies
uniformity, not the converse, and the paper is explicit that SBC "offers no
guarantee that the posterior will cover the ground truth for any single
observation". Uniform ranks are therefore necessary, not sufficient.

(An earlier version of this entry, and three code comments, cited "Theorem 2"
and stated the equivalence as "iff".)

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

**Optuna developers. (2025). `optuna.trial.Trial.report` and
`optuna.pruners.MedianPruner`.** Optuna 5.0.0 API reference.
https://optuna.readthedocs.io/en/v5.0.0/reference/generated/optuna.trial.Trial.report.html
and
https://optuna.readthedocs.io/en/v5.0.0/reference/generated/optuna.pruners.MedianPruner.html

Two claims, both load-bearing.

`Trial.report()` is unavailable for multi-objective studies, which is the
entire reason `optimization/pruning_strategies.py` exists. Verified by
execution on the installed 5.0.0, not from the docs alone: on a study created
with two directions, `trial.report(0.5, 1)` raises `NotImplementedError`
("Trial.report is not supported for multi-objective optimization"), and
`trial.should_prune()` raises the same. The module docstrings attribute this
to upstream issue #3450; the *behaviour* is what is verified here.

`MedianPruner` is the Optuna feature that the `"primary"` strategy mirrors.
It is documented here and not in Akiba et al. (2019).

### Specification and library documentation

Not research works, so these carry no OpenAlex record and are cited by
specification rather than DOI.

**van Rossum, G., Lehtosalo, J., & Langa, Ł. (2014). PEP 484 -- Type hints.**
Python Enhancement Proposals. https://peps.python.org/pep-0484/

The "NewType" section defines a helper that a type checker treats as a
distinct subtype while, at runtime, the callable returns its argument
unchanged. This backs the claim in `validation/registry.py` and `objectives.py`
that `CanonicalMetricName`, `RawScore` and `MinimizeScore` cost no wrapper
object: the stored values are a plain `str` and plain `float`s. Verified
directly by construction, since the runtime behaviour is the load-bearing part
of the claim.

**Optuna developers. (2025). `optuna.distributions.CategoricalDistribution`.**
Optuna 5.0.0 API reference.
https://optuna.readthedocs.io/en/v5.0.0/reference/generated/optuna.distributions.CategoricalDistribution.html

Backs the 0.2.0 changelog's warning that `FlowMatchingSpace.quality()` breaks
resume for a 0.1.0 study. `choices` is stored as an ordered tuple and forms
part of the distribution's identity, so reordering it makes the stored and
requested distributions unequal and Optuna refuses the parameter as a dynamic
value space. Verified against the installed release rather than from the
documentation alone: `CategoricalDistribution([False, True]) !=
CategoricalDistribution([True, False])`, with `.choices` round-tripping as
`(False, True)` and `(True, False)` respectively. Re-verified on 5.0.0 after
the dependency floor moved; both still hold.

**Optuna developers. (2025). `optuna.study.create_study` and
`optuna.trial.FrozenTrial`.** Optuna 5.0.0 API reference.
https://optuna.readthedocs.io/en/v5.0.0/reference/generated/optuna.study.create_study.html
and
https://optuna.readthedocs.io/en/v5.0.0/reference/generated/optuna.trial.FrozenTrial.html

Version-pinned deliberately: the documentation root serves whichever release
is current, so a claim checked against one release would silently come to
point at pages that may no longer say it. 5.0.0 is the version installed and
verified against here.

`directions` is a sequence and `FrozenTrial.values` is indexed positionally
against it, so objective columns are matched by position and never by name.
This backs the resume-guard schema comparison in `objectives.py`
(`normalize_schema_entry`, `schema_matches`) and the ordering claim recorded
with `MinimizeScore`.
