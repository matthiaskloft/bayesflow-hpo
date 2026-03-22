# bayesflow-hpo

## Source-Backed Implementation (MANDATORY)

- EVERYTHING that is implemented must be theoretically backed by either a research article (accessed via full text) or package documentation.
- If an article is not accessible, ask the user to find and download it. Do NOT proceed with unverified assumptions.
- Do NOT blindly use in-memory knowledge from LLM training — verify against actual sources first.
- Document the references backing a given implementation in docstrings and docs (e.g., API/architecture documentation).
- Verify references via the OpenAlex API and save them formatted in APA 7 in [`docs/references.md`](docs/references.md).

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
├── objectives.py       # Param/cost normalization, objective value extraction
├── registration.py     # Custom network registration helpers
├── utils.py            # loguniform_float/int sampling helpers
├── search_spaces/      # Dataclass-based search space definitions
│   ├── base.py         # BaseSearchSpace (auto-derives dimensions from fields)
│   ├── composite.py    # CompositeSearchSpace, NetworkSelectionSpace, SummarySelectionSpace
│   ├── registry.py     # Factory registries + aliasing
│   ├── training.py     # TrainingSpace (learning rate, batch size, etc.)
│   ├── inference/      # Inference network spaces (5 types)
│   │   ├── coupling_flow.py, flow_matching.py, diffusion.py
│   │   ├── consistency.py, stable_consistency.py
│   └── summary/        # Summary network spaces (5 types)
│       ├── deep_set.py, set_transformer.py, fusion_transformer.py
│       └── time_series_network.py, time_series_transformer.py
├── pipeline.py         # check_pipeline() pre-flight validation
├── types.py            # BuildApproximatorFn, TrainFn, ValidateFn type aliases
├── builders/           # Construct BayesFlow objects from trial params
│   ├── workflow.py     # build_continuous_approximator(), _compile_for_compat()
│   ├── adapter.py      # Adapter-related build logic
│   └── registry.py     # Builder registries for custom networks
├── optimization/       # Optuna study management + trial logic
│   ├── study.py        # create_study(), sampler/pruner presets, optimize_until(), warm_start
│   ├── objective.py    # GenericObjective, ObjectiveConfig
│   ├── sampling.py     # sample_hyperparameters()
│   ├── callbacks.py    # OptunaReportCallback, MovingAverageEarlyStopping
│   ├── constraints.py  # Memory/param budget checks (pre-training rejection)
│   ├── checkpoint_pool.py  # CheckpointPool for trial weight persistence
│   ├── cleanup.py      # cleanup_trial()
│   ├── pruning_strategies.py  # Multi-objective pruning (dominance, mo-sha, primary)
│   └── validation_callback.py  # PeriodicValidationCallback (pluggable pruning)
├── validation/         # Fixed-dataset validation pipeline
│   ├── data.py         # ValidationDataset generation + save/load
│   ├── registry.py     # 13 built-in metrics (calibration, NRMSE, SBC, etc.)
│   ├── pipeline.py     # run_validation_pipeline()
│   ├── metrics.py      # compute_condition_metrics(), aggregate_condition_rows()
│   ├── inference.py    # make_bayesflow_infer_fn()
│   ├── result.py       # ValidationResult dataclass
│   ├── dry_run.py      # validate_once()
│   └── sbc_tests.py    # SBC uniformity tests (KS, chi-squared)
└── results/            # Post-optimization analysis + export
    ├── extraction.py   # trials_to_dataframe(), get_pareto_trials(), summarize_study()
    ├── export.py       # save/load_workflow_with_metadata(), get_workflow_metadata()
    └── visualization.py # Pareto front, param importance, metric panels, scatter
```

## Key Patterns

- **BaseSearchSpace**: Declare dimensions as dataclass fields → `dimensions` and `sample()` auto-derive; only implement `build()`
- **Registry pattern**: Search spaces, builders, and metrics all use name→factory registries with alias support
- **CompositeSearchSpace**: Combines inference + summary + training sub-spaces; NetworkSelectionSpace / SummarySelectionSpace let Optuna choose network type
- **Trial budget**: Trials exceeding `max_param_count` or `max_memory_mb` are rejected pre-training and NOT counted toward `n_trials`
- **Public API**: `__init__.py` re-exports ~80 high-level symbols; internal helpers (e.g. `BaseSearchSpace`, `Dimension`, `MetricFn`, `sample_hyperparameters`) are accessible via submodule imports but intentionally excluded from the top-level API for stability

## Project TODOs

See [`docs/TODO.md`](docs/TODO.md) for open and completed project tasks.

## References

See [`docs/references.md`](docs/references.md) for verified APA 7 citations backing the implementation.

## Gotchas

- `KERAS_BACKEND=torch` must be set before importing; tests fail silently otherwise
- `optimize()` auto-infers `param_keys`/`data_keys` from the adapter; `search_space` is required
- Three optional hooks (`build_approximator_fn`, `train_fn`, `validate_fn`) replace build/train/validate steps while reusing the full trial lifecycle
- `default_train_fn` passes `num_batches=` to `approximator.fit()`, compatible with BayesFlow 2.0.8+ (`build_dataset(num_batches=...)`)
- `check_pipeline()` runs automatically at the start of `optimize()` to catch interface errors early
- Budget-rejected trials don't count toward `n_trials`, so actual total trials can exceed `max_total_trials`
- Validation dataset keys must match adapter keys or you get a runtime error
