# Reference Summaries

This directory contains detailed summaries of research papers backing the implementation of bayesflow-hpo features.

## Fulltexts

Local fulltexts are available in `fulltexts/` directory (PDF format). Additional
extractions and structured data may be maintained in the sibling repository:
```
../bayesflow_hpo_article/literature/
```

## Format

Each reference file (`{firstauthor}{year}_{topic}.md`) contains:

- **Key method / algorithm description** with equation/algorithm numbers
- **Page references** for implementation-critical details
- **Edge-case handling** recommendations from original authors
- **Intentional deviations** from reference method (if any) and rationale
- **Relevance note**: which module/function in bayesflow-hpo this backs

## References by Feature

### Optimization
- `akiba2019_optuna.md` — Optuna framework (`optimization/study.py`)
- `bergstra2011_tpe.md` — TPE sampler (`optimization/study.py`)
- `balandat2020_botorch.md` — BoTorch framework (`optimization/study.py`)
- `daulton2020_qehvi.md` — qEHVI acquisition (`optimization/study.py`)
- `daulton2021_qnehvi.md` — qNEHVI acquisition (`optimization/study.py`)
- `deb2014_nsga3.md` — NSGA-III sampler preset (`optimization/study.py`)
- `li2018_hyperband.md` — Hyperband (`optimization/study.py`)
- `sobol1967_qmc.md` — Sobol sequences for QMC warm-up (`optimization/study.py`)
- `joe2008_sobol.md` — Improved Sobol direction numbers (`optimization/study.py`)
- `feurer2019_hpo.md` — BOHB algorithm (`optimization/study.py`)

### Pruning
- `deb2002_nsga2.md` — NSGA-II, non-dominated sorting (`optimization/pruning_strategies.py`, `results/extraction.py`)
- `schmucker2021_moasha.md` — MO-ASHA pruning strategies (`optimization/pruning_strategies.py`)
- `emmerich2018_moo.md` — Multi-objective fundamentals (`optimization/pruning_strategies.py`)

### Validation Metrics
- `talts2018_sbc.md` — SBC methodology (`validation/sbc_tests.py`, `validation/registry.py`)
- `lopezpaz2017_c2st.md` — Global C2ST (`validation/c2st.py`)
- `linhart2023_lc2st.md` — L-C2ST local diagnostics (`validation/c2st.py`)

### HPO Foundations
- `bischl2023_hpo.md` — HPO survey (overall guidance)

## Citation

All references are formatted in APA 7 style. See main `docs/references.md` for the complete bibliography.
