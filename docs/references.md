# References

Checked against the OpenAlex API, with version exceptions documented below.
APA 7 format.

## Audit status (2026-09-15)

Three passes. The first was prompted by three inherited, unchecked citations
found during PR #86, two of which were wrong. The second was a systematic
sweep of every implementation-backing claim in `src/` and this file. The
third (2026-09-15) closed the remainder: it read back the fourteen entries
that had metadata but no substance check, resolved or annotated the five
entries carrying no DOI, and deleted the `docs/references/` summaries rather
than repair them.

**Verification stamps.** Every locator added from this pass onwards names the
edition it was checked against and the date, in the form
`(Thm. 1 -- arXiv:1804.06788, verified 2026-09-11)`. This is a convention,
not decoration: two of the thirteen errors corrected in PR #86 were a claim
that went stale without anyone touching the sentence, and a locator read off
a different edition than the one at hand. Neither is visible without the
stamp. See [`contributing-references.md`](contributing-references.md).

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
- Optuna 5.0.0: `create_study` raises
  `ValueError("The number of objectives must be greater than 0.")` for an
  empty `directions` list (`optuna/study/study.py:1264`) -- read from the
  installed source and reproduced. This is the floor that makes
  `objective_metrics=[]` with `cost_metric=None` an invalid configuration
  rather than a degenerate one, and it is why `ObjectiveConfig` rejects the
  pair up front instead of letting `create_study` fail later.

### Corrected (third pass, 2026-09-15)

Issue #90 listed fourteen entries with metadata resolved and descriptive text
never read back. One of them, Lemos et al. (2023), had already been verified
against full text in the TARP commit (`a88ffd7`) before this pass began, so
the list was stale by one; **the remaining thirteen were read here.** Two
were wrong, both in the same way -- a result the paper *uses* was described
as a result the paper *introduces*:

| Claim | Was | Is |
|---|---|---|
| Bergstra et al. (2011) | "Proposes TPE **and sequential model-based optimization**" | SMBO is prior art the paper reviews (Sec. 2, citing [8, 9]). The paper's own proposals are the GP-based method (Sec. 3) and TPE, the adaptive-Parzen method (Sec. 4) |
| Smith et al. (2018) | "**Shows** that the gradient-noise scale couples learning rate and batch size" | The noise scale `g = eps (N/B - 1)` and the resulting `B ∝ eps` rule are attributed in Sec. 1 to Smith & Le (2017), with Goyal et al. (2017) having observed the scaling empirically. This paper's own contribution is the *equivalence* of increasing `B` and decaying `eps`. The rule also carries a condition the entry omitted: `B ∝ eps` holds when `B << N` |

Two more were correct but too thin to be checkable, and were given locators
rather than corrected: Lopez-Paz & Oquab (2017) (the statistic is *held-out*
accuracy on `D_te`, Sec. 3 step four, null `N(1/2, 1/(4 n_te))` in Sec. 3.1)
and Gneiting (2011) (Thm. 3.1 and Thm. 3.3, see the entry).

### Verified, and correct as written (third pass)

Balandat et al. (2020) · Daulton et al. (2020) · Deb & Jain (2014) ·
Bischl et al. (2023) · Goyal et al. (2017) · Shallue et al. (2019) ·
Lueckmann et al. (2021) · Bland & Altman (1986) · Sobol' (1967). Each entry
below carries the locator and edition it was checked against.

Three of these are not in the local Zotero index and were verified against a
named public edition instead, which the entries record: Goyal et al. (2017)
and Gneiting (2011) against their arXiv versions, Bland & Altman (1986)
against the authors' corrected reproduction.

### Bulk metadata check

Every DOI in this file was resolved against the OpenAlex API. All resolve.

**Seven entries cite a published year against a preprint or online-first
DOI**, so the year here differs from OpenAlex's `publication_year`. This is
the intended convention, not an error, and the count is stated in full
because an earlier version of this paragraph said "five" and then listed
four:

| Entry | Cited | OpenAlex | DOI resolved |
|---|---|---|---|
| Balandat et al. | 2020 | 2019 | `10.48550/arXiv.1910.06403` |
| Deb & Jain | 2014 | 2013 | `10.1109/TEVC.2013.2281535` (online-first) |
| Modrák et al. | 2025 | 2023 | `10.1214/23-BA1404` (online-first) |
| Smith et al. | 2018 | 2017 | `10.48550/arXiv.1711.00489` |
| Li et al. | 2018 | 2016 | `10.48550/arXiv.1603.06560` |
| Lopez-Paz & Oquab | 2017 | 2016 | `10.48550/arXiv.1610.06545` |
| Shallue et al. | 2019 | 2018 | `10.48550/arXiv.1811.03600` |

The last three are new as of 2026-09-15: resolving the missing DOIs (#91)
necessarily created the mismatches, since the only DOI those works have
belongs to the preprint. Updating the count in the same pass is the point of
the convention -- the previous "five" was correct when written and silently
false afterwards, which is failure mode (2) in
[`contributing-references.md`](contributing-references.md).

**The five entries that carried no DOI are now resolved (2026-09-15).** Four
had an arXiv DOI available for the preprint of the cited version, and take it
under the same published-year-against-preprint-DOI convention as above. One
has no DOI at any version and now says so explicitly:

| Entry | Outcome | OpenAlex |
|---|---|---|
| Bergstra et al. (2011) | **No DOI assigned.** Neither the NeurIPS 24 proceedings version nor any preprint carries one | `W2106411961` (no DOI) |
| Li et al. (2018) | `10.48550/arXiv.1603.06560` (preprint) | JMLR record `W2963815651` carries no DOI; preprint `W2556522401` |
| Lopez-Paz & Oquab (2017) | `10.48550/arXiv.1610.06545` (preprint) | `W2599043313`; ICLR carries no DOI |
| Lueckmann et al. (2021) | `10.48550/arXiv.2101.04653` (preprint) | `W3118581558`; PMLR v130 carries no DOI |
| Shallue et al. (2019) | `10.48550/arXiv.1811.03600` (preprint) | `W2900167092`; JMLR carries no DOI |

OpenAlex indexes the Li et al. JMLR version as *18*(1), 6765--6816 (cumulative
volume pagination) where this file cites *18*(185), 1--52 (JMLR's own article
pagination). Same work, two pagination conventions; the article pagination is
kept.

### Deleted: `docs/references/*.md`

The seventeen per-paper summaries in `docs/references/` were deleted on
2026-09-15 rather than repaired. They backed no code path -- every
implementation claim cites this file or its own docstring -- and every one of
the three that was spot-checked was defective: misidentified definitions, an
algorithm labelled with the wrong name over pseudocode that was not the
paper's, locators taken against a different edition than the indexed copy, and
a citation to a work that does not appear to exist. Rewriting meant re-reading
seventeen papers to a standard the originals never met, to produce a secondary
source nothing consumes. This file is now the single place a claim is stated.

The two arXiv source tarballs that lived in that directory went with it; both
papers (Lopez-Paz & Oquab 2017, Linhart et al. 2023) are in the local Zotero
index and on arXiv.

### Still not verified

The two Optuna issue-tracker pointers in `study.py`, which need repository
access this audit did not have.

### Verified, but before the stamp convention existed

Six entries were read against full texts in the first or second pass and so
carry no verification stamp, because the convention postdates them:

Daulton et al. (2021) · Deb et al. (2002) · Joe & Kuo (2008) ·
Säilynoja et al. (2022) · Naeini et al. (2015) · Lemos et al. (2023)

They are listed rather than left silent because
[`contributing-references.md`](contributing-references.md) defines an absent
stamp as meaning "nobody has read this back", which is not true of these.
Stamp them when they are next touched; do not treat the gap as a finding
against their content.

## Coverage Matrix

Feature implementations and their backing references.

### Optimization

| Feature | Module | Reference |
|---------|--------|-----------|
| Optuna framework | `optimization/study.py` | Akiba et al. (2019) |
| End-to-end objective ranking | `tests/test_end_to_end/` | Optuna docs; Deb et al. (2002) |
| End-to-end `log_gamma` direction | `tests/test_end_to_end/` | Sailynoja et al. (2022); Modrak et al. (2025), Sec. 4.1 |
| Objective column ordering | `objectives.py` | Optuna 5.0.0 docs |
| Optional cost objective (`cost_metric=None`) | `optimization/objective.py`, `api.py` | Optuna 5.0.0 source (>= 1 direction; positional `values`); Deb et al. (2002) for the Pareto-front consequence |
| Categorical choice-order identity | `search_spaces/base.py` | Optuna 5.0.0 docs |
| `CanonicalMetricName` type | `validation/registry.py` | PEP 484 |
| `RawScore` / `MinimizeScore` types | `objectives.py` | PEP 484 |
| TPE sampler preset | `optimization/study.py` | Bergstra et al. (2011) |
| Pruned-checkpoint retention sampling | `optimization/checkpoint_pool.py` | Vitter (1985), Alg. R |
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
| ECE term not claimed for `mean_calibration_error` | `validation/registry.py` | Naeini et al. (2015) |
| TARP joint coverage (`tarp_error`, `tarp_error_random`) | `validation/tarp.py` | Lemos et al. (2023), Secs. 3.1--3.2, 4.1--4.3, Thm. 3, Alg. 2 |
| Per-condition joint rows, against unconditional pooling | `validation/pipeline.py` | Modrák et al. (2025), Sec. 4.1; Lemos et al. (2023), Sec. 4.3 |
| SBI benchmarking | overall | Lueckmann et al. (2021) |

### BayesFlow Diagnostic Wrappers

The following metrics wrap `bf.diagnostics.*` functions. Their methodological
references are provided by the BayesFlow package, not this package:

- `calibration_error` (median absolute coverage deviation over 20
  nominal levels -- despite the historical name, *not* an ECE)
- `mean_calibration_error` (the same deviations aggregated with the mean)
- `rmse`, `nrmse`
- `contraction` (posterior contraction)
- `z_score` (posterior z-score)
- `log_gamma`

`mean_calibration_error` is the one entry in that list with **no upstream
reference**, and is flagged here rather than left to look like an oversight.
It calls `bf.diagnostics.calibration_error` with a non-default `aggregation`,
and neither that aggregation choice nor the metric's name is taken from
BayesFlow or from a cited article -- both are this package's own design. The
*wrapped computation* is BayesFlow's and is documented in that function's
docstring; only the choice of `np.mean` over the default `np.median`, and the
name, originate here.

It is deliberately *not* called an ECE.
`bf.diagnostics.expected_calibration_error` is a different statistic --
bin-size-weighted, over one-hot model indices, for model comparison, after
Naeini et al. (2015) -- and the Expected Calibration Error of that literature
is a weighted mean over bins of predicted probability, not an unweighted mean
over equally spaced nominal coverage levels. The term is therefore not
claimed. See the Naeini et al. (2015) entry below.

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
auto-differentiation. The abstract states the combination directly --
"Monte-Carlo (MC) acquisition functions, a novel sample average approximation
optimization approach, autodifferentiation, and variance reduction
techniques" -- over "probabilistic models written in PyTorch". Backs the
`"botorch"` / GP sampler preset.

(Abstract, Sec. 1 -- arXiv:1910.06403, verified 2026-09-15.)

### Bergstra, J., Bardenet, R., Bengio, Y., & Kegl, B. (2011)

Algorithms for hyper-parameter optimization. In *Advances in Neural
Information Processing Systems 24* (pp. 2546--2554). No DOI assigned.

**No DOI.** Neither the proceedings version nor any preprint carries one;
OpenAlex holds the work as `W2106411961` with a null DOI. Recorded as a
finding, not an omission.

Introduces **TPE**, the adaptive-Parzen-window estimator of Section 4, and a
Gaussian-process-based alternative in Section 3, both built on the expected
improvement criterion. Backs the `"tpe"` sampler preset.

The paper does **not** propose sequential model-based optimization: Section 2
reviews SMBO as prior art ("SMBO algorithms have been used in many
applications", citing [8, 9]). Conditional hyperparameters are handled: the
abstract gives the contribution as "making response surface models `P(y|x)`
in which many elements of hyper-parameter assignment (`x`) are known to be
irrelevant given particular values of other elements" -- the tree-structured
configuration spaces of Section 1. The improvement over random search is the
paper's second stated contribution, "Automatic sequential optimization
outperforms both manual and random search".

(An earlier version of this entry read "Proposes TPE and sequential
model-based optimization". Abstract, Secs. 1--4 -- NeurIPS 24 version,
verified 2026-09-15.)

### Bischl, B., Binder, M., Lang, M., Pielok, T., Richter, J., Coors, S., Thomas, J., Ullmann, T., Becker, M., Boulesteix, A.-L., Deng, D., & Lindauer, M. (2023)

Hyperparameter optimization: Foundations, algorithms, best practices, and
open challenges. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge
Discovery*, *13*(2), e1484. https://doi.org/10.1002/widm.1484

Comprehensive survey ("Advanced Review") of HPO foundations, algorithms, and
open challenges. The abstract scopes it from "simple techniques such as grid
or random search to more advanced methods like evolution strategies, Bayesian
optimization, Hyperband, and racing", plus "practical recommendations
regarding important choices to be made when conducting HPO". Cited as overall
guidance, not for any single locator.

(Abstract -- WIREs DMKD 13(2), verified 2026-09-15.)

### Daulton, S., Balandat, M., & Bakshy, E. (2020)

Differentiable expected hypervolume improvement for parallel multi-objective
Bayesian optimization. In *Advances in Neural Information Processing Systems
33* (pp. 9851--9864). https://doi.org/10.48550/arXiv.2006.05078

Extends EHVI to parallel MOO with differentiable MC estimates (qEHVI). The
abstract defines qEHVI as "an acquisition function that extends EHVI to the
parallel, constrained evaluation setting", exact up to MC integration error,
whose gradients are computed "via auto-differentiation" rather than
approximated. Backs the qEHVI acquisition preset.

(Abstract -- arXiv:2006.05078, verified 2026-09-15.)

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

Extends NSGA-II to many-objective optimization using reference-point-based
selection. The abstract defines many-objective as "having four or more
objectives" and names the algorithm "a reference-point based many-objective
NSGA-II (we call it NSGA-III)" that "emphasizes population members which are
non-dominated yet close to a set of supplied reference points". Backs the
`"nsga3"` sampler preset.

Note the scope: this Part I covers box-constrained problems only
("This paper presents results on unconstrained problems and the sequel paper
considers constrained and other specialties"). Our preset is used on
box-constrained search spaces, so Part I is the right citation.

(Abstract, Sec. I -- IEEE TEVC 18(4), verified 2026-09-15.)

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

**Proposition 9** (Sec. 4.1, "Linear weighting", p. 591) is the second
locator this package relies on, in `pruning_strategies.py`: "In case of a
convex Pareto front, for each solution in `Y_N` there is a solution of a
linear scalarization problem for some weight vector `w`." The consequence
the code cites is the unnumbered remark immediately after it -- "If the
Pareto front is non-convex, then, in general, there can be points on the
Pareto front which are the solutions of no LSP" -- illustrated in Fig. 2.
Note what the proposition does *not* say: by Proposition 8 an LSP solution
always lies on the Pareto front, convex or not. What fails on a non-convex
front is *coverage*, not correctness, and Definition 15 is where "convex
Pareto front" is defined.

(An earlier version of this entry cited non-dominated sorting at
~~Eqs. 3--4~~ and complexity bounds at ~~Props. 7, 9~~. The equations do not
survive the full text, and the complexity bound we rely on is Deb et al.'s,
not this tutorial's. The second-pass audit then over-corrected, recording that
"Props. 3--4 and 7--9 are cone-order results" and that neither existed as
cited; Proposition 9 does exist and is the scalarization result quoted
above. Found by `scripts/check_citations.py`, which flagged the
`pruning_strategies.py` locator as absent from this entry.
Secs. 3--4.1, Defs. 5, 8, 15, Props. 8--9 -- Natural Computing 17(3),
verified 2026-09-15.)

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
https://doi.org/10.48550/arXiv.1603.06560

**DOI note.** The JMLR version carries no DOI (OpenAlex `W2963815651`, null
DOI); the preprint DOI above is cited against the published year, the same
convention used for Balandat, Deb & Jain, Modrák and Smith. OpenAlex
paginates the JMLR version as *18*(1), 6765--6816 (cumulative volume
pagination); JMLR's own article pagination, 18(185), 1--52, is kept here.

Combines random search with adaptive resource allocation via Successive
Halving. **Algorithm 1 is Hyperband itself**, and takes the reduction factor
as an input annotated "default eta = 3"; SuccessiveHalving is its inner loop
(lines 3--9). Section 3.6, "Setting eta", recommends eta of 3 *or 4* in
practice and notes the theoretically optimal value is e ~= 2.718 -- so the
default and the recommendation are separate claims with separate locators.
Section 6 does suggest quasi-random sampling as a promising extension:
"Quasi-random methods like Sobol or latin hypercube [...] may improve the
performance of Hyperband by giving better coverage of the search space."

(An earlier version of this entry placed the eta=3 default in
~~Section 3.6~~; it is in Algorithm 1's input line. Section 3.6 is still a
valid locator for the "3 or 4" recommendation, stated above.)

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

**Theorem 3** ("Local consistency and single class evaluation", Sec. 3) is
the statistic's locator, cited in `objectives.py` for its bound: it states
that for a Bayes-optimal `f` and `N_v -> infinity`, `t_MSE0(f, x_o) = 0` is
necessary and sufficient for local consistency of `q` at `x_o`. `MSE_0` is
the mean squared distance between the predicted class probability and one
half, evaluated on samples from the posterior-approximation class only
(`C = 0`) -- which is what makes the test reference-free. Since a
probability lies in `[0, 1]`, the squared deviation from `0.5` is bounded by
`0.25`, and so is its mean; that bound is what `METRIC_DIRECTIONS` records
as the metric's worst raw value.

The paper numbers its theorems flat (1, 2, 3), not by section. Theorem 1 is
the accuracy-based oracle C2ST and Theorem 2 the regression C2ST of Kim et
al.

(An earlier version of the `objectives.py` comment cited ~~Theorem 3.1~~,
which does not exist. Found by `scripts/check_citations.py`. Secs. 2--3,
Thms. 1--3 -- arXiv:2306.03580, verified 2026-09-15.)

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
Also verified against full text (2026-09-14), for the joint-metric design in
[`plans/plan-joint-metric-path.md`](plans/plan-joint-metric-path.md): **Sec.
4.3** (case study 2) is the source for marginal rank statistics being blind to
a posterior that ignores the data, and the result is sharper than "blind".
Figure 4 splits the rank distribution of the parameters by the average value of
the corresponding data elements and reports that "the distributions for the two
cases exactly compensate to make the overall distribution uniform" -- so the
marginal ranks of a posterior equal to the prior are *exactly* uniform, not
merely hard to distinguish from uniform. The paper's own remedy is a test
quantity involving both data and parameters, with the joint log-likelihood
recommended as "a useful default". **Sec. 4.4** (case study 5) adds the
companion case: a posterior with correct marginals but wrong correlation
structure passes SBC on the univariate parameters while likelihood-based
quantities fail.

This is the paper-side justification for the joint metric path, and it is what
makes `calibration_error`, `log_gamma`, `nrmse`, `coverage` and `sbc_ks` --
every metric in `validation/registry.py`, all of them marginal -- unable on
their own to reject a posterior that ignores the data.

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

### Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015)

Obtaining well calibrated probabilities using Bayesian binning. In
*Proceedings of the AAAI Conference on Artificial Intelligence*, *29*(1).
https://doi.org/10.1609/aaai.v29i1.9602

Defines the Expected Calibration Error as a bin-size-weighted mean of the gap
between confidence and accuracy over bins of predicted probability. Cited here
only to mark what this package's `mean_calibration_error` is **not**: an
unweighted mean over equally spaced nominal coverage levels is a different
statistic, so the ECE name is not used for it. It is also the reference behind
`bf.diagnostics.expected_calibration_error`, which this package does not wrap.
OpenAlex work `W2254249950`.

### Lopez-Paz, D., & Oquab, M. (2017)

Revisiting classifier two-sample tests. In *Proceedings of the 5th
International Conference on Learning Representations (ICLR 2017)*.
https://doi.org/10.48550/arXiv.1610.06545

**DOI note.** ICLR carries no DOI; the preprint DOI above (OpenAlex
`W2599043313`) is cited against the published year, per the convention
above. The indexed PDF's footer reads "Published as a conference paper at
ICLR 2017", confirming the cited year against a Zotero record labelled 2018.

Binary classifier as two-sample test: label the `P` sample positive and the
`Q` sample negative, then test whether accuracy exceeds chance. Two details
of Section 3 are load-bearing for `global_c2st()` and were missing from an
earlier version of this entry:

- The statistic is accuracy on a **held-out** split. Section 3's five steps
  split `D` into disjoint `D_tr` and `D_te`, train `f` on `D_tr`, and return
  the accuracy on `D_te` (Eq. 2) as "our C2ST statistic". Accuracy on the
  training split is not the statistic.
- Section 3.1 gives the null: under `H_0: P = Q` classification is
  impossible, `n_te * t` is `Binomial(n_te, 1/2)`, and for large `n_te` the
  null distribution of the statistic is approximately `N(1/2, 1/(4 n_te))`.
  This is what "exceeds chance" is measured against.

Theorem 1 (Sec. 3.2) gives the test's power in terms of the effect size
`eps`, where accuracy is `1/2 + eps` under `H_1`.

Implementation: `bayesflow_hpo.validation.c2st.global_c2st()`.

(Secs. 3--3.2, Eq. 2, Thm. 1 -- arXiv:1610.06545, verified 2026-09-15.)

### Bland, J. M., & Altman, D. G. (1986)

Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, *327*(8476), 307–310. https://doi.org/10.1016/S0140-6736(86)90837-8

Shows that Pearson correlation measures linear association rather than
agreement and is insensitive to changes in scale. Both halves are the two
numbered points under the heading "Inappropriate use of correlation
coefficient":

1. "`r` measures the strength of a relation between two variables, not the
   agreement between them. We have perfect agreement only if the points
   [...] lie along the line of equality, but we will have perfect
   correlation if the points lie along any straight line."
2. "A change in scale of measurement does not affect the correlation, but it
   certainly affects the agreement."

This is why `validation/registry.py` does not report a correlation as an
agreement metric.

Not in the local Zotero index. Verified against the authors' reproduction at
https://www-users.york.ac.uk/~mb55/meas/ba.htm, which is the 1986 *Lancet*
text as reprinted with a small numerical correction in *Biochimica Clinica*
(1987) -- a different edition from the *Lancet* original cited above, and
recorded as such. OpenAlex work `W2015795623`.

(Sec. "Inappropriate use of correlation coefficient", points 1--2 -- York
reproduction of the 1987 corrected reprint, verified 2026-09-15.)

### Gneiting, T. (2011)

Making and evaluating point forecasts. *Journal of the American Statistical Association*, *106*(494), 746–762. https://doi.org/10.1198/jasa.2011.r10138

Establishes that point summaries must be evaluated with a *consistent*
scoring function -- one whose "expected score is minimized when following the
directive" -- and characterizes which losses are consistent for which
functional. The two pairs this package relies on:

- **Mean / squared error.** Theorem 3.1 (Savage): a scoring function is
  consistent for the mean functional if and only if it is a Bregman
  function, `S(x, y) = phi(y) - phi(x) - phi'(x)(y - x)` for convex `phi`.
  Squared error is the `phi(y) = y^2` case.
- **Median / absolute error.** Theorem 3.3 (Thomson, Saerens): a scoring
  function is consistent for the `alpha`-quantile if and only if it is
  generalized piecewise linear of order `alpha`. The median is
  `alpha = 1/2`, for which absolute error is the canonical member.

The paper's framing is why this matters here rather than being a matter of
taste: it demonstrates that averaging an arbitrary error measure "can lead to
grossly misguided inferences, unless the scoring function and the forecasting
task are carefully matched". A metric that reports a median must not be
scored as though it reported a mean.

Not in the local Zotero index. Verified against the arXiv preprint of the
JASA article. OpenAlex work `W2075965721`.

(Abstract, Secs. 1.1, 3.1, 3.3, Thms. 3.1 and 3.3 -- arXiv:0912.0902,
verified 2026-09-15.)

### Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023)

Sampling-based accuracy testing of posterior estimators for general inference. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 19256–19273). PMLR. https://doi.org/10.48550/arXiv.2302.03026

Introduces Tests of Accuracy with Random Points (TARP) for joint posterior
coverage testing using posterior samples. Implemented in
`bayesflow_hpo.validation.tarp` as the joint metrics `tarp_error` (provided
reference points) and `tarp_error_random` (diagnostic). OpenAlex work
`W4319453761`.

Verified against full text (2026-09-14, extended 2026-09-14 for the
implementation) for the design in
[`plans/plan-joint-metric-path.md`](plans/plan-joint-metric-path.md):

- **Sec. 3.1** ("High posterior density coverage testing") is the section
  behind the claim that expected HPD coverage is blind to a posterior that
  ignores the data. The paper works the case `p_hat(theta|x) = p(theta)`
  explicitly, notes that the HPD generator is then independent of `x` so that
  `H(p_hat, alpha, x) = H(p_hat, alpha)`, and concludes that this estimator
  "has perfect HPD ECP in this case". The same section states that the HPD
  region generator "is not a positionable credible region generator", which is
  why Theorem 3 does not apply to it -- positionability is what the theorem's
  proof needs, since it varies the position function.
- **Sec. 3.2** ("Distance to random point coverage testing") defines the TARP
  region generator as the *positionable* generator producing spherical regions
  around a reference position.
- **Algorithm 2** is the estimator implemented. Transcribed from the PDF
  rather than from the converted text, because the conversion renders the
  algorithm body as an image and the math extraction is lossy exactly there:

  ```
  for i = 1 to N_sims:
      theta_r ~ p~(theta_r | x)                  # reference point
      f_i = (1/n) sum_j 1[d(theta_ij, theta_r) < d(theta*_i, theta_r)]
  ECP(p_hat, alpha, D_theta_r) = (1/N_sims) sum_i 1(f_i < 1 - alpha)
  ```

  So `f_i` is the fraction of posterior draws lying closer to the reference
  point than the truth does, and the coverage curve is the ECDF of the `f_i`
  -- the diagonal for an exact posterior. The header names the inputs as a
  set of simulations, a "parameter distance metric `d`", and a "reference
  point sampling distribution `p~(.|x)`"; the dependence on `x` is the
  paper's own notation, not an embellishment.
- **Sec. 4** records the experimental setup the implementation's defaults
  follow: parameters normalized "to the range [0, 1]", reference points
  generated "uniformly in the D-dimensional hypercube", and "the Euclidean
  or L2 distance as a metric". Sec. 4.1's Gaussian toy model uses
  `theta* ~ U(-5, 5)` with `log sigma ~ U(-5, -1)`, which is what the test
  suite reproduces -- and the small sigma matters, since at a larger one the
  "correct case" stops being calibrated under the truncated prior.
- **Sec. 4.1** states the paper's own reference choice outright: "To pick the
  TARP reference points, we use the prior (`p~(theta_r|x) = p(theta_r)`)."
  This is the backing for the opt-in `reference="prior_derangement"` mode in
  `validation/tarp.py`. It does **not** contradict the Sec. 4 setup bullet
  above, which reports reference points drawn "uniformly in the
  D-dimensional hypercube": the same section normalizes parameters "to the
  range [0, 1]", so in that experiment the prior *is* the unit hypercube and
  the two descriptions coincide. They come apart for any prior that is not
  uniform on a box -- which is the case the derangement mode exists for,
  since permuting the truths samples `p(theta)` whatever its shape.
  (Sec. 4.1 -- arXiv:2302.03026 / PMLR 202, verified 2026-09-16.)

  BayesFlow follows the same choice by a different route. Its
  `bayesflow.diagnostics.metrics.accuracy_random_points` defaults
  `references=None` to "a derangement of the target parameters via a random
  cyclic shift (a pure permutation with no fixed points)". Since the targets
  are prior draws, that samples the prior. This package draws a
  rejection-sampled permutation instead of one `np.roll` offset, because a
  single shift determines the whole reference set from one integer; the
  reasoning is at the draw site in `validation/tarp.py`.

  That function first appears in BayesFlow v2.0.13 (absent in v2.0.12), so
  it is not available at every version this package accepts:
  `pyproject.toml` declares only `bayesflow>=2.0.0` and the repository
  carries no lockfile, so the resolved version depends on when the
  environment was built. Nothing here imports it -- the citation records
  whose convention `prior_derangement` follows, not a dependency -- so the
  floor is deliberately left alone. (BayesFlow
  `bayesflow/diagnostics/metrics/accuracy_random_points.py` at tag
  `v2.0.13`, verified 2026-09-16.)
- **Sec. 4.2** explores "the dependence on the reference point distribution
  and the distance metric", which is the basis for treating the metric
  choice as not changing the verdict. The same section is the backing for
  the claim that the *reference* choice does not move a verdict either: the
  authors re-ran all four toy cases drawing `theta_r` from `U(0, 1)`,
  `U(0, 0.5)`, `N(0.5, sigma)` for sigma between 0.01 and 0.1, and from two
  fixed points, and concluded "the proposed method is robust to different
  distributions for `theta_r`, and choices of distance metric". The one
  qualification is the biased case, where "the different `theta_r`
  distributions led to different curves, but all of them clearly showed
  there was a bias" -- so the robustness is of the verdict, not of the
  number, which is why switching `reference=` changes the settings pin.
  (Sec. 4.2 -- arXiv:2302.03026 / PMLR 202, verified 2026-09-16.)
- **Sec. 4.3** is the section behind the two-key split: TARP with an
  `x`-*independent* reference point shares HPD coverage's blindness to
  `p_hat(theta|x) = p(theta)`. This is why `tarp_error_random` is registered
  as a diagnostic and only a supplied reference emits the objective key
  `tarp_error`.

The earlier attribution of the HPD-blindness result to "Sec. 3.1" was carried
between repositories without a full-text check. It is correct.

### Lueckmann, J.-M., Boelts, J., Greenberg, D. S., Goncalves, P. J., & Macke, J. H. (2021)

Benchmarking simulation-based inference. In *Proceedings of the 24th
International Conference on Artificial Intelligence and Statistics*, PMLR
130, pp. 343--351. https://doi.org/10.48550/arXiv.2101.04653

**DOI note.** PMLR carries no DOI; the preprint DOI above (OpenAlex
`W3118581558`) is cited against the published year, per the convention
above. The PMLR landing page remains
https://proceedings.mlr.press/v130/lueckmann21a.html

Public benchmark for SBI algorithms. Both halves of the key finding are the
abstract's own words: "the choice of performance metric is critical" and
"Neural network-based approaches generally exhibit better performance, but
there is no uniformly best algorithm". Cited as overall guidance for SBI
benchmarking, not for a locator.

(Abstract -- arXiv:2101.04653 / AISTATS 2021, PMLR 130, verified
2026-09-15.)

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

**Section 6, "Experiments"** (p. 7, discussing Fig. 1 on NAS-201 /
ImageNet16-120) is the locator for the scale-sensitivity claim in
`pruning_strategies.py`, and both quoted phrases are verbatim: scalarization
techniques "tend to penalize one objective heavier than the other, focusing
on models with very low prediction time, and avoiding the area of the search
space with slower more accurate models", whereas "globally informed
techniques are more robust towards objectives of different magnitude". This
is an empirical finding on two objectives of very different scale, which is
why the strategy normalizes each metric before comparing.

(An earlier version of this entry had the two algorithms the other way round.
Corrected against the full text.)

### Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017)

*Accurate, large minibatch SGD: Training ImageNet in 1 hour* [Preprint].
arXiv. https://doi.org/10.48550/arXiv.1706.02677

Introduces **gradual** learning-rate warmup to avoid early optimization
problems when training with aggressive learning rates. Section 2.2
("Warmup") is the locator. The paper is careful about what is new: warmup as
such is attributed to He et al. (2016) -- "this issue can be alleviated by a
properly designed warmup [He2016], namely, a strategy of using less
aggressive learning rates at the start of training" -- and the *constant*
warmup of that work is found insufficient at large `k`, where "a transition
out of the low learning rate warmup phase can cause the training error to
spike". Gradual warmup is this paper's proposal: starting from learning rate
`eta` and incrementing it "by a constant amount at each iteration such that
it reaches `eta_hat = k * eta` after 5 epochs", after which the original
schedule resumes. That linear ramp is the shape our warmup implements.

OpenAlex work `W2622263826`. Together with the Keras `CosineDecay`
documentation, this backs the optional fixed-budget warmup.

Not in the local Zotero index; verified against the arXiv full text.

(Sec. 2.2, "Gradual warmup" -- arXiv:1706.02677, verified 2026-09-15.)

### Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2018)

Don't decay the learning rate, increase the batch size. In *Proceedings of the
6th International Conference on Learning Representations (ICLR 2018)*.
https://doi.org/10.48550/arXiv.1711.00489

Backs the learning-rate/batch-size coupling in `search_spaces/training.py`.
Section 1 states the noise scale as `g = eps (N/B - 1)`, with `eps` the
learning rate, `N` the training-set size and `B` the batch size, and reports
that there is "an optimum fluctuation scale `g` which maximizes the test set
accuracy (at constant learning rate)", giving "an optimal batch size
proportional to the learning rate". That is the `B` proportional to `eps`
rule the search space reparameterizes on.

Two qualifications an earlier version of this entry dropped:

- **The noise-scale result is not this paper's.** Section 1 attributes it to
  Smith & Le (2017) -- "Smith & Le (2017) argued one should interpret SGD as
  integrating a stochastic differential equation" -- and notes that "Goyal
  et al. (2017) already observed this scaling rule empirically". This
  paper's own contribution is the *equivalence*: decaying the learning rate
  by a factor and multiplying the batch size by that factor give
  near-identical learning curves, which it states no prior paper had shown
  empirically.
- **The rule is conditional.** The proportionality is stated for
  `B` much smaller than `N`. It is not a claim about the whole batch-size
  range, which matters wherever the derived dimension is evaluated near the
  dataset size.

(An earlier version read "Shows that the gradient-noise scale couples
learning rate and batch size", attributing an inherited result to this
paper. Abstract, Sec. 1 -- arXiv:1711.00489 / ICLR 2018, verified
2026-09-15.)

### Shallue, C. J., Lee, J., Antognini, J. M., Sohl-Dickstein, J., Frostig, R., & Dahl, G. E. (2019)

Measuring the effects of data parallelism on neural network training. *Journal
of Machine Learning Research, 20*(112), 1–49. https://doi.org/10.48550/arXiv.1811.03600

**DOI note.** The JMLR version carries no DOI; the preprint DOI above
(OpenAlex `W2900167092`) is cited against the published year, per the
convention above. The JMLR landing page remains
https://www.jmlr.org/papers/v20/18-789.html

Shows that batch-size effects and suitable metaparameter settings vary
greatly between workloads, supporting joint batch-size and learning-rate
exploration instead of a fixed package-wide batch size. The abstract states
both halves: the relationship between batch size and steps-to-target "varies
with the training algorithm, model, and data set, and [we] find extremely
large variation between workloads", and "disagreements in the literature on
how batch size affects model quality can largely be explained by differences
in metaparameter tuning and compute budgets at different batch sizes". The
paper also reports "no evidence that larger batch sizes degrade out-of-sample
performance", which is why the search space does not cap batch size on
quality grounds.

**Sections 4 and 5** are the locators for the warmup decision in
`search_spaces/training.py`. Section 4's preamble records the tuning cost:
the metaparameters were tuned "independently [...] at each batch size,
including the initial learning rate and, when learning rate decay was used,
the decay schedule", by quasi-random search at roughly 100 non-divergent
trials per batch size, and even so the authors report that "despite sampling
100 metaparameter configurations per batch size and training for up to 25
hours per configuration, it is still not certain whether we truly saturated
performance". Section 5 ("Discussion") states the conclusion as a
recommendation: "practitioners tune all optimization parameters anew when
they change the batch size or they risk masking the true behavior of the
training procedure".

Note what this does *not* say. An earlier version of the `training.py`
docstring cited ~~Sec. 5.1~~ for the claim that a third correlated schedule
axis "made their own tuning unreliable". Section 5 has no subsections, and
the paper reports the opposite: the decay schedule was tuned successfully
alongside the learning rate. What the paper supports is the *cost* of
retuning everything whenever batch size moves; treating that cost as a reason
to fix the warmup length is this package's inference, not theirs.

(Abstract, Secs. 1, 4--5 -- JMLR 20 (2019) 1--49 / arXiv:1811.03600, verified
2026-09-15.)

### Sobol', I. M. (1967)

On the distribution of points in a cube and the approximate evaluation of
integrals. *USSR Computational Mathematics and Mathematical Physics*,
*7*(4), 86--112. https://doi.org/10.1016/0041-5553(67)90144-9

Seminal paper introducing Sobol' low-discrepancy sequences for numerical
integration. Section 1.1 states the aim as classes of nets and sequences
"possessing a new property of uniformity"; Section 3 gives the effective
construction, which needs neither multiplication nor addition but only shifts
and bitwise addition modulo 2; Sections 6 and 7 bound the discrepancy and the
irregularity and conclude that these nets are among the best in uniformity of
distribution. Backs the QMC warm-up feature (`qmc_startup_trials`).

**Edition.** The citation above is the English translation (*USSR Comput.
Math. Math. Phys.* 7(4), 86--112, DOI `10.1016/0041-5553(67)90144-9`). The
copy in the local index is the Russian original, *Zh. Vychisl. Mat. Mat.
Fiz.* 7(4), 784--802, which is what these section numbers were read from --
the section numbering is shared, the pagination is not. Page locators against
this work must say which edition they mean; the deleted
`docs/references/sobol1967_qmc.md` gave translation page numbers for a
reading of the original, and that is how it was caught.

(Secs. 1.1, 3, 6--7 -- Russian original, Zh. Vychisl. Mat. Mat. Fiz. 7(4),
verified 2026-09-15.)

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

(An earlier version of this entry, and three code comments, cited
~~Theorem 2~~ and stated the equivalence as "iff".)

### Vitter, J. S. (1985)

Random sampling with a reservoir. *ACM Transactions on Mathematical
Software*, *11*(1), 37--57. https://doi.org/10.1145/3147.3165

**Algorithm R**, stated in Section 2 (p. 39), is what
`CheckpointPool.save_pruned` implements for retaining pruned trials:
"When the (t + 1)st record in the file is being processed, for t >= n, the
n candidates form a random sample of the first t records. The (t + 1)st
record has a n/(t + 1) chance of being in a random sample of size n of the
first t + 1 records, and so it is made a candidate with probability
n/(t + 1). The candidate it replaces is chosen randomly from the n
candidates." Definition 1 on the same page gives the first step: the first
n records go into the reservoir unconditionally, which is the pool's fill
phase before the cap is reached.

Two things the entry records because they are easy to get wrong:

- **Algorithm R is not Vitter's contribution.** The paper attributes it to
  Alan Waterman ("a reservoir algorithm due to Alan Waterman") and calls it
  "previously the method of choice". Vitter's own result is **Algorithm Z**
  (Section 5), which is not what this package implements: Z optimizes the
  *number of random variates* for a long stream, and a pruned-trial pool
  offers at most a few hundred items, where R's O(N) cost is irrelevant.
- The population here is the pruned trials *offered*, not records in a file,
  and the pool never needs N in advance -- which is the property that makes
  R applicable at all, since a study's pruned count is unknown until it ends.

(Alg. R and Def. 1, Sec. 2, p. 39 -- ACM TOMS 11(1), March 1985, verified
2026-09-15.)

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

**Tsitouras, Ch. (2011). Runge–Kutta pairs of order 5(4) satisfying only the
first column simplifying assumption.** *Computers & Mathematics with
Applications, 62*(2), 770–775. https://doi.org/10.1016/j.camwa.2011.06.002

Verified via the OpenAlex API (title, year, venue, volume/issue, pages and
sole author as printed above). Backs the stage count in
`optimization/constraints.py`: the Tsitouras 5(4) pair is a seven-stage
embedded Runge–Kutta method, and BayesFlow's implementation of it
(`bayesflow/utils/integrate.py`, `tsit5_step`) holds `k1..k7` live
simultaneously alongside `state`, `new_state` and the error estimate. Seven is
therefore a floor on the number of batch-sized tensors an adaptive
flow-matching sampler keeps allocated at once, which is what
`estimate_validation_memory_mb` multiplies its per-row activation proxy by.
That this pair is the default is read from BayesFlow itself
(`bayesflow/networks/defaults.py`: `FLOW_MATCHING_INTEGRATE_DEFAULTS =
{"method": "tsit5", "steps": "adaptive"}`, bayesflow 2.0.12), not assumed.


**SciPy community. (n.d.). `scipy.stats.gmean`.** *SciPy API reference*.
Retrieved September 21, 2026, from
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gmean.html

Defines the geometric mean as the exponential of the arithmetic mean of
natural logarithms. Backs `validation.metrics.reduce_metric`; this package
requires strictly positive inputs explicitly, without a scale-dependent
pseudocount. The official documentation was read directly. An OpenAlex API
search for `scipy.stats.gmean` returned no indexed works; this documentation
reference has no DOI to verify there.

**NumPy developers. (n.d.). `numpy.nanmean`.** *NumPy API reference*.
Retrieved September 21, 2026, from
https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html

Documents arithmetic averaging with NaN omission and NaN for all-NaN
slices. Backs the aggregation missing-value convention; official package
documentation is the primary source rather than an academic-method claim.
An OpenAlex API search for `numpy.nanmean` returned no indexed works.
