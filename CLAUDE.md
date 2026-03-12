# bayesflow-hpo

Generic hyperparameter optimization for BayesFlow 2.x models,
wrapping Optuna multi-objective search with BayesFlow-aware
search spaces, builders, and validation.

## Scope

- Generic HPO package for BayesFlow 2.x models
- Focus on reusable search spaces, objective, validation, and result analysis
- Defaults for optional search space parameters MUST follow bayesflow defaults, but users can override them
- Keep domain-specific simulator logic out of this package

## Commands

```bash
export KERAS_BACKEND=torch          # Required before any run
pip install -e ".[dev]"             # Editable install with dev deps
pytest tests/ -v                    # Run tests
ruff check src/ tests/              # Lint (matches CI)
pip install -e ".[dashboard]"       # Optional: Optuna dashboard
```

## Conventions

- Python >= 3.11
- Keras backend: PyTorch preferred
- src/ layout + setuptools
- pytest + ruff
- NumPy-style docstrings
- Full type hints (mypy-compatible)

## Architecture

```
src/bayesflow_hpo/
├── api.py              # optimize() entry point + adapter key inference
├── registration.py     # Custom network registration helpers
├── search_spaces/      # Dataclass-based search space definitions
│   ├── base.py         # BaseSearchSpace (auto-derives dimensions from fields)
│   ├── composite.py    # CompositeSearchSpace, NetworkSelectionSpace
│   ├── registry.py     # Factory registries + aliasing
│   └── ...             # Per-network spaces (coupling_flow, deep_set, etc.)
├── builders/           # Construct BayesFlow objects from trial params
│   ├── workflow.py     # build_workflow() → BasicWorkflow
│   └── registry.py     # Builder registries for custom networks
├── optimization/       # Optuna study management + trial logic
│   ├── study.py        # create_study(), optimize_until(), warm_start
│   ├── objective.py    # GenericObjective (main trial function)
│   ├── callbacks.py    # OptunaReportCallback, MovingAverageEarlyStopping
│   └── constraints.py  # Memory/param budget checks (pre-training rejection)
├── validation/         # Fixed-dataset validation pipeline
│   ├── data.py         # ValidationDataset generation + save/load
│   ├── registry.py     # 13+ built-in metrics (calibration, NRMSE, SBC, etc.)
│   ├── pipeline.py     # run_validation_pipeline()
│   └── sbc_tests.py    # SBC uniformity tests (KS, chi-squared, C2ST)
└── results/            # Post-optimization analysis + export
    ├── extraction.py   # trials_to_dataframe(), get_pareto_trials()
    └── visualization.py # Pareto front, param importance, metric panels
```

## Key Patterns

- **BaseSearchSpace**: Declare dimensions as dataclass fields → `dimensions` and `sample()` auto-derive; only implement `build()`
- **Registry pattern**: Search spaces, builders, and metrics all use name→factory registries with alias support
- **CompositeSearchSpace**: Combines inference + summary + training sub-spaces; NetworkSelectionSpace lets Optuna choose network type
- **Trial budget**: Trials exceeding `max_param_count` or `max_memory_mb` are rejected pre-training and NOT counted toward `n_trials`

## Gotchas

- `KERAS_BACKEND=torch` must be set before importing; tests fail silently otherwise
- `optimize()` auto-infers `param_keys`/`data_keys` from the adapter; explicit kwargs override inference
- Budget-rejected trials don't count toward `n_trials`, so actual total trials can exceed `max_total_trials`
- Validation dataset keys must match adapter keys or you get a runtime error
