# Reference Summaries

This directory contains detailed summaries of research papers backing the implementation of bayesflow-hpo features.

## Fulltexts and Extractions

Complete fulltexts and structured extractions are maintained in the sibling repository:
```
../bayesflow_hpo_article/literature/
```

- **Fulltexts**: `../bayesflow_hpo_article/literature/references/fulltext_pdf/`
- **Extractions**: `../bayesflow_hpo_article/literature/pipeline_data/31_all_extractions.json`
- **Reference index**: `../bayesflow_hpo_article/literature/references/_index.json`

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
- `daulton2020_qehvi.md` — qEHVI acquisition (`optimization/study.py`)

### Pruning
- `deb2002_nsga2.md` — NSGA-II, non-dominated sorting (`optimization/pruning_strategies.py`)

### Validation Metrics
- `talts2018_sbc.md` — SBC methodology (`validation/sbc_tests.py`, `validation/registry.py`)
- `lopezpaz2017_c2st.md` — Global C2ST (`validation/c2st.py`)
- `linhart2023_lc2st.md` — L-C2ST local diagnostics (`validation/c2st.py`)

### HPO Foundations
- `bischl2023_hpo.md` — HPO survey (overall guidance)

## Citation

All references are formatted in APA 7 style. See main `docs/references.md` for the complete bibliography.
