# Architecture

## Package Structure

```
src/bayesflow_hpo/
├── __init__.py              # Public API (re-exports all user-facing symbols)
├── api.py                   # High-level optimize() convenience function
├── objectives.py            # Parameter counting, normalization, objective extraction
├── registration.py          # Custom network registration facade
├── utils.py                 # loguniform sampling helpers
├── types.py                 # BuildApproximatorFn, TrainFn, ValidateFn type aliases
├── pipeline.py              # check_pipeline() pre-flight validation
├── _display.py              # DisplayDataFrame for hiding row index in display
│
├── search_spaces/           # Declarative hyperparameter dimensions
│   ├── base.py              # Dimension types, SearchSpace protocol, BaseSearchSpace
│   ├── composite.py         # CompositeSearchSpace, NetworkSelectionSpace, SummarySelectionSpace
│   ├── registry.py          # Name → space factory lookup, aliases, registration
│   ├── training.py          # TrainingSpace (lr, batch_size, decay_rate)
│   ├── inference/           # One module per inference network type
│   │   ├── coupling_flow.py
│   │   ├── flow_matching.py
│   │   ├── diffusion.py
│   │   ├── consistency.py
│   │   └── stable_consistency.py
│   └── summary/             # One module per summary network type
│       ├── deep_set.py
│       ├── set_transformer.py
│       ├── time_series_network.py
│       ├── time_series_transformer.py
│       └── fusion_transformer.py
│
├── builders/                # Construct BayesFlow objects from param dicts
│   ├── adapter.py           # (deprecated since v0.2.0)
│   ├── registry.py          # Builder function lookup tables
│   └── workflow.py          # build_continuous_approximator, CosineDecay optimizer
│
├── optimization/            # Optuna integration layer
│   ├── objective.py         # ObjectiveConfig, GenericObjective, default_train/validate_fn
│   ├── callbacks.py         # OptunaReportCallback, MovingAverageEarlyStopping
│   ├── constraints.py       # estimate_param_count, estimate_peak_memory_mb
│   ├── cleanup.py           # GPU/memory cleanup after each trial
│   ├── sampling.py          # sample_hyperparameters (thin wrapper)
│   ├── study.py             # create_study, resume_study, warm_start_study
│   ├── checkpoint_pool.py   # CheckpointPool for trial weight persistence
│   ├── pruning_strategies.py # Multi-objective pruning (dominance, mo-sha, primary)
│   └── validation_callback.py # PeriodicValidationCallback (pluggable pruning)
│
├── validation/              # Post-training evaluation
│   ├── data.py              # ValidationDataset, generate/save/load, grid helpers
│   ├── dry_run.py           # validate_once — quick compatibility check
│   ├── inference.py         # make_bayesflow_infer_fn adapter
│   ├── metrics.py           # compute_condition_metrics, aggregate_condition_rows
│   ├── pipeline.py          # run_validation_pipeline (orchestrator → ValidationResult)
│   ├── registry.py          # Metric registry (MetricFn, register/resolve/list)
│   ├── result.py            # ValidationResult dataclass with table methods
│   ├── c2st.py              # C2ST metrics (L-C2ST, global C2ST, ValidateFn factory)
│   └── sbc_tests.py         # SBC uniformity tests (KS, chi-squared)
│
└── results/                 # Post-optimization analysis
    ├── extraction.py        # get_pareto_trials, select_best_trial, trial_table, best_config
    ├── export.py            # save/load_workflow_with_metadata
    ├── visualization.py     # plot_study, plot_pareto_front, plot_param_importance, etc.
    └── _colors.py           # BayesFlow-aligned color palette
```

## Data Flow

A single HPO run follows this pipeline:

```
                    ┌──────────────────────────────┐
                    │         optimize()            │
                    │     (api.py — entry point)    │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              ▼                ▼                     ▼
    CompositeSearchSpace   ValidationDataset    create_study()
    (inference + summary   (fixed simulations   (Optuna study
     + training spaces)     from simulator)      with sampler)
              │                │                     │
              └────────────────┼─────────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │     GenericObjective          │
                    │  (called once per Optuna      │
                    │   trial)                      │
                    └──────────┬───────────────────┘
                               │
           ┌───────────────────┼───────────────────────┐
           ▼                   ▼                        ▼
    1. Sample params    2. Budget checks         3. Build networks
    from search space   (param count &           (space.build →
    via trial.suggest   memory estimates)        BayesFlow objects)
           │                   │                        │
           └───────────────────┼────────────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   4. build_continuous_        │
                    │      approximator()           │
                    │   (ContinuousApproximator     │
                    │    + Adam/CosineDecay)        │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   5. Train (configurable)     │
                    │   (default: fit_online with   │
                    │    early stopping & pruning,  │
                    │    or user-provided train_fn) │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   6. run_validation_pipeline  │
                    │   (per-condition metrics via  │
                    │    registry → ValidationResult│
                    │    with table methods)        │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   7. Return objectives        │
                    │   (configurable metric key,   │
                    │    normalized_param_score)    │
                    └──────────────────────────────┘
```

## Design Principles

### Protocol-Based Extensibility

All search spaces implement the `SearchSpace` protocol:

```python
class SearchSpace(Protocol):
    @property
    def dimensions(self) -> list[Dimension]: ...
    def sample(self, trial: Any) -> dict[str, Any]: ...
    def build(self, params: dict[str, Any]) -> Any: ...
```

This allows users to register custom network types without modifying the package.

### Default vs. Constant Dimensions

Each dimension can be fixed at a specific value via `constant=<value>`. When set, the parameter is not tuned by Optuna but instead uses the constant value. When `constant` is not set (the `_UNSET` sentinel), the dimension is tunable. This lets users lock individual dimensions while tuning others.

### Fixed Validation for Fair Comparison

`ValidationDataset` is generated once and reused across all trials, ensuring that differences in objective values reflect architecture quality, not simulation noise.

### Configurable Metrics and Objective

The validation pipeline uses a metric registry to resolve metric names to callable functions. Users can:
- Choose which metrics to compute per trial via `metrics=["calibration_error", "coverage", ...]`
- Register custom metrics via `register_metric(name, fn)`
- Select which metric key drives the HPO objective via `objective_metric="calibration_error"`

### Thin BayesFlow Wrappers

The package delegates to BayesFlow for all core functionality:
- **Approximator construction** delegates to `bf.ContinuousApproximator`
- **Adapter construction** is left to the user (no wrapper)
- **Diagnostics** are wrapped as thin metric functions that reshape arrays for the BF API
- **Training** defaults to `approximator.fit(simulator=...)` but can be overridden via `train_fn`

### Multi-Objective Pareto Optimization

Multiple objectives are minimized simultaneously:
1. **Configurable quality metrics** (default: `["calibration_error", "nrmse"]`) — each gets its own Pareto direction in `"pareto"` mode, or they are averaged in `"mean"` mode
2. **Cost metric** — inference time (default) or normalized parameter count

Supports 2–3 objectives. The Pareto front contains all non-dominated solutions, letting users choose their preferred trade-off point. `select_best_trial()` provides lexicographic-Pareto selection with satisficing thresholds.
