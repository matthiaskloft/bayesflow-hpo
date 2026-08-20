# API Reference

Complete list of public symbols exported from `bayesflow_hpo`.

## High-Level API

### `optimize(...) -> optuna.Study`

One-call convenience function that generates validation data, runs `check_pipeline()` pre-flight validation, and executes a full HPO study.

```python
def optimize(
    simulator, adapter, search_space,
    # Custom approximator hooks (all optional)
    build_approximator_fn=None, train_fn=None, validate_fn=None,
    # Validation data
    validation_conditions=None, sims_per_condition=200, n_posterior_samples=500,
    # Objectives
    objective_metrics=None, objective_mode="pareto", cost_metric="inference_time",
    # Training
    training_mode="fixed_budget", epochs=200, num_batches=50,
    early_stopping_patience=None, early_stopping_window=7,
    early_stopping_monitor="objective_mean",
    lr_warmup_epochs=None, lr_warmup_steps=None, lr_warmup_fraction=None,
    pruning_strategy="dominance",
    # Budget
    max_param_count=1_000_000, max_memory_mb=None,
    metric_constraints_hard=None, metric_constraints_soft=None,
    memory_safety_margin=0.2,
    # Study
    n_trials=50, max_total_trials=None,
    study_name="bayesflow_hpo", storage=DEFAULT_STORAGE, resume=False,
    sampler=None, pruner=None,
    # Optional
    directions=None, warm_start_from=None, warm_start_top_k=25,
    qmc_startup_trials=0,
    checkpoint_pool=None, show_progress_bar=True,
) -> optuna.Study
```

| Parameter | Description |
|-----------|-------------|
| `simulator` | BayesFlow simulator for online training and validation data generation. |
| `adapter` | BayesFlow adapter for data preprocessing. `param_keys`/`data_keys` are auto-inferred. |
| `search_space` | **Required.** `CompositeSearchSpace` defining tunable dimensions. |
| `build_approximator_fn` | Optional `(hparams) -> Approximator`. Must return an **uncompiled** approximator. Defaults to `build_continuous_approximator()`. |
| `train_fn` | Optional `(approximator, simulator, hparams, callbacks) -> None`. Defaults to `default_train_fn()`. |
| `validate_fn` | Optional `(approximator, validation_data, n_posterior_samples) -> dict[str, float]`. Defaults to `default_validate_fn()`. |
| `validation_conditions` | Condition grid (e.g. `{"N": [50, 100, 200]}`). |
| `sims_per_condition` | Simulations per condition grid point (default 200). |
| `n_posterior_samples` | Posterior draws for validation (default 500). |
| `objective_metrics` | Metric keys to optimize. Default `["calibration_error", "nrmse"]`. |
| `objective_mode` | `"pareto"` (default) — each metric is its own objective. `"mean"` — arithmetic mean of metrics. |
| `cost_metric` | Cost objective: `"inference_time"` (default) or `"param_count"`. |
| `training_mode` | `"fixed_budget"` (cosine, full budget) or `"open_ended"` (inverse-sqrt, validation early stopping). |
| `epochs` | Training epochs, or safety cap in open-ended mode (default 200). |
| `num_batches` | Online simulation batches per epoch (default 50). |
| `early_stopping_patience` | Validation checks without improvement in open-ended mode (`None` selects 5). |
| `early_stopping_window` | Moving-average window in validation checks for open-ended runs, or epochs for the training-loss callback (default 7). |
| `early_stopping_monitor` | `"objective_mean"` averages `objective_metrics` after minimize-direction conversion; the separate cost objective is excluded. A metric name monitors one validation metric. |
| `lr_warmup_epochs` | Optional epoch override; `None` selects 1 in open-ended mode. A sequence enables categorical HPO. |
| `lr_warmup_steps` | Exact step override; a sequence enables categorical HPO. |
| `lr_warmup_fraction` | Fixed-budget fraction; `None` selects 5%, maximum 10%. A sequence enables categorical HPO. |
| `pruning_strategy` | Multi-objective pruning: `"dominance"` (default), `"mo-sha"`, `("primary", "metric")`, or `"none"`. |
| `max_param_count` | Reject trials exceeding this param count pre-training (default 1 000 000). |
| `max_memory_mb` | Optional peak-memory budget in MB, or `"auto"` for CUDA free-memory auto-detection. |
| `metric_constraints_hard` | Optional hard metric constraints `[(metric, threshold, "above" \| "below"), ...]` (reject after validation). |
| `metric_constraints_soft` | Optional soft metric constraints `[(metric, threshold, "above" \| "below"), ...]` (feasibility-guided sampling for sampler presets). |
| `memory_safety_margin` | Safety margin for `max_memory_mb="auto"` (default `0.2`). |
| `n_trials` | Number of *trained* trials to collect (default 50). |
| `max_total_trials` | Hard cap on non-rejected trials. Defaults to `3 * n_trials`. |
| `sampler` | Sampler preset string (`"tpe"`, `"gp"`, `"botorch"`, `"nsga2"`, `"nsga3"`, `"auto"`, `"random"`) or `BaseSampler` instance. Default `None` = TPE. |
| `pruner` | Pruner preset string (`"median"`, `"hyperband"`, `"none"`) or `BasePruner` instance. |
| `resume` | If `True`, continue a previously persisted study. |
| `qmc_startup_trials` | Sobol QMC trials before main sampler (default 0 = disabled). |
| `checkpoint_pool` | Optional `CheckpointPool` for persisting best trial weights. |

### `check_pipeline(...)`

Pre-flight validation that catches interface errors before launching expensive studies. Called automatically at the start of `optimize()`.

```python
def check_pipeline(
    simulator, adapter, search_space,
    build_approximator_fn=None, train_fn=None, validate_fn=None,
    objective_metrics=("calibration_error", "nrmse"),
    sims_per_condition=5, n_posterior_samples=2,
    validation_conditions=None, epochs=1, num_batches=1,
) -> None
```

Raises `PipelineError` on builder failures, missing metric keys, or signature mismatches.

### `infer_keys_from_adapter(adapter) -> dict`

Walks the adapter's transform list to infer `param_keys`, `data_keys`, and `inference_conditions`.

---

## Type Aliases

```python
BuildApproximatorFn = Callable[[dict[str, Any]], Any]
TrainFn = Callable[[Any, bf.simulators.Simulator, dict[str, Any], list[Any]], None]
ValidateFn = Callable[[Any, ValidationDataset, int], dict[str, float]]
```

---

## Search Spaces

### Dimension Types

| Class | Purpose |
|-------|---------|
| `IntDimension(name, low, high, step, log, constant)` | Integer hyperparameter. Set `constant=<value>` to fix. |
| `FloatDimension(name, low, high, log, constant)` | Float hyperparameter. Set `constant=<value>` to fix. |
| `CategoricalDimension(name, choices, constant)` | Categorical hyperparameter. Set `constant=<value>` to fix. |
| `DerivedDimension(name, derive)` | Value computed after the sampled dimensions. |

When `constant` is set, the dimension is not tuned by Optuna — it uses the constant value instead. When unset (default `_UNSET` sentinel), the dimension is tunable.

### Inference Spaces

| Class | BayesFlow Network |
|-------|-------------------|
| `CouplingFlowSpace()` | `bf.networks.CouplingFlow` |
| `FlowMatchingSpace()` | `bf.networks.FlowMatching` |
| `DiffusionModelSpace()` | `bf.networks.DiffusionModel` |
| `ConsistencyModelSpace(epochs, num_batches)` | `bf.networks.ConsistencyModel` |
| `StableConsistencyModelSpace()` | `bf.networks.StableConsistencyModel` |

`FlowMatchingSpace` untuned constants are synchronized to BayesFlow
defaults at runtime (`bf.networks.TimeMLP` signature defaults and
`bf.networks.FlowMatching.INTEGRATE_DEFAULT_CONFIG`).

`FlowMatchingSpace` profile helpers:
- `FlowMatchingSpace.fast()`
- `FlowMatchingSpace.balanced()`
- `FlowMatchingSpace.quality()`
- `FlowMatchingSpace.preset("default" | "fast" | "balanced" | "quality")`

### Summary Spaces

| Class | BayesFlow Network |
|-------|-------------------|
| `DeepSetSpace()` | `bf.networks.DeepSet` |
| `SetTransformerSpace()` | `bf.networks.SetTransformer` |
| `TimeSeriesNetworkSpace()` | `bf.networks.TimeSeriesNetwork` |
| `TimeSeriesTransformerSpace()` | `bf.networks.TimeSeriesTransformer` |
| `FusionTransformerSpace()` | `bf.networks.FusionTransformer` |

### Training Space

| Class | Controls |
|-------|----------|
| `TrainingSpace()` | `initial_lr`, `batch_size`, `decay_rate` |

### Composite Spaces

| Class | Purpose |
|-------|---------|
| `CompositeSearchSpace(inference_space, summary_space, training_space)` | Bundles all spaces |
| `NetworkSelectionSpace(candidates)` | Optuna picks inference network type |
| `SummarySelectionSpace(candidates)` | Optuna picks summary network type |

### Space Registry Functions

```python
register_inference_space(name, factory, aliases=None, overwrite=False)
register_summary_space(name, factory, aliases=None, overwrite=False)
list_inference_spaces() -> list[str]
list_summary_spaces() -> list[str]
```

---

## Builders

```python
build_continuous_approximator(hparams, adapter, search_space,
                               checkpoint_dir=None) -> ContinuousApproximator
```

Builds an **uncompiled** `ContinuousApproximator` from sampled hyperparameters. Handles inference + optional summary network construction; the objective compiles it with the selected training-mode schedule.

### Default Hook Wrappers

```python
default_train_fn(approximator, simulator, hparams, callbacks) -> None
default_validate_fn(approximator, validation_data, n_posterior_samples) -> dict[str, float]
```

Public default implementations used by `optimize()` when no custom hooks are provided.

---

## Optimization

### ObjectiveConfig

| Field | Default | Description |
|-------|---------|-------------|
| `simulator` | *(required)* | BayesFlow simulator |
| `adapter` | *(required)* | BayesFlow adapter |
| `search_space` | *(required)* | Composite search space |
| `validation_data` | *(required)* | Pre-generated `ValidationDataset` |
| `training_mode` | `"fixed_budget"` | Coherent schedule/stopping mode |
| `epochs` | `200` | Max training epochs per trial |
| `num_batches` | `50` | Online batches per epoch |
| `early_stopping_patience` | `None` | Open-ended validation patience (`None` selects 5) |
| `early_stopping_window` | `7` | Moving-average window |
| `early_stopping_monitor` | `"objective_mean"` | Combined minimize-oriented validation objective or one metric name |
| `lr_warmup_epochs` | `None` | Mode-specific warmup default or categorical choices |
| `lr_warmup_steps` | `None` | Exact warmup-step override |
| `lr_warmup_fraction` | `None` | Fixed-budget 5% default, capped at 10% |
| `max_param_count` | `1_000_000` | Pre-training param budget |
| `max_memory_mb` | `None` | Peak-memory budget (disabled) |
| `metric_constraints_hard` | `None` | Hard metric constraints (post-validation rejection) |
| `n_posterior_samples` | `500` | Posterior draws for final validation |
| `pruning_strategy` | `"dominance"` | Multi-objective pruning strategy (`"dominance"`, `"mo-sha"`, `("primary", metric)`, `"none"`) |
| `pruning_n_startup_trials` | `None` | Min completed trials before pruning (`None` = auto-detect from sampler) |
| `objective_metrics` | `["calibration_error", "nrmse"]` | Metric keys to optimize |
| `objective_mode` | `"pareto"` | `"pareto"` or `"mean"` |
| `cost_metric` | `"inference_time"` | Cost objective (`"inference_time"` or `"param_count"`) |
| `checkpoint_pool` | `None` | Optional `CheckpointPool` |
| `build_approximator_fn` | `None` | Custom build hook |
| `train_fn` | `None` | Custom training hook |
| `validate_fn` | `None` | Custom validation hook |

### GenericObjective

```python
objective = GenericObjective(config: ObjectiveConfig)
values = objective(trial: optuna.Trial)  # tuple of floats
```

### Study Management

```python
create_study(study_name, directions, storage, load_if_exists,
             sampler: str | BaseSampler | None = None,  # "tpe", "gp", "botorch", "nsga2", "nsga3", "auto", "random"
             pruner: str | BasePruner | None = None,    # "median", "hyperband", "none"
             metric_constraints_soft=None,
             metric_names, warm_start_from, warm_start_top_k,
             qmc_startup_trials=0) -> optuna.Study
resume_study(study_name, storage) -> optuna.Study
optimize_until(study, objective, n_trained, max_total_trials, show_progress_bar) -> None
warm_start_study(target_study, source_study, top_k=25) -> int
```

### Sampling

```python
sample_hyperparameters(trial, space: CompositeSearchSpace) -> dict[str, Any]
```

### Callbacks

```python
OptunaReportCallback(trial, monitor="loss", report_frequency=10)
MovingAverageEarlyStopping(monitor="loss", window=5, patience=3, restore_best_weights=True)
PeriodicValidationCallback(trial, approximator, validation_data, ...)
```

### Constraints

```python
estimate_param_count(params) -> int
estimate_peak_memory_mb(params, batch_size=None, dtype_bytes=4) -> float
exceeds_memory_budget(params, max_memory_mb, batch_size=None) -> bool
```

### Checkpoint Pool

```python
class CheckpointPool:
    def __init__(self, pool_dir="checkpoints", pool_size=5): ...
    def maybe_save(self, trial_number, objective_value, approximator) -> bool
    @property
    def best_checkpoint_dir(self) -> Path | None
    @property
    def trial_numbers(self) -> list[int]
    def cleanup(self) -> None
```

### Cleanup

```python
cleanup_trial() -> None
```

---

## Validation

### Data

```python
generate_validation_dataset(simulator, param_keys, data_keys,
                            condition_grid=None, sims_per_condition=200, seed=42) -> ValidationDataset
make_condition_grid(*, linspace=None, logspace=None, values=None) -> dict[str, list]
make_validation_dataset(simulator, param_keys, data_keys, *,
                        linspace=None, logspace=None, values=None,
                        sims_per_condition=200, seed=42) -> ValidationDataset
save_validation_dataset(dataset, path) -> None
load_validation_dataset(path) -> ValidationDataset
```

### Pipeline

```python
run_validation_pipeline(approximator, validation_data, n_posterior_samples=1000,
                        metrics=None) -> ValidationResult
validate_once(approximator, validation_data, n_sims=2,
              n_posterior_samples=10, metrics=None) -> ValidationResult
```

### ValidationResult

```python
@dataclass(frozen=True)
class ValidationResult:
    condition_metrics: pd.DataFrame
    summary: dict[str, float]
    per_parameter: dict[str, ValidationResult] | None = None
    timing: dict[str, float]
    n_conditions: int = 0
    n_posterior_samples: int = 0
    metric_names: list[str]

    def summary_table(self) -> pd.DataFrame
    def condition_table(self, metric: str | None = None) -> pd.DataFrame
    def parameter_table(self) -> pd.DataFrame | None
    def objective_scalar(self, key: str = "calibration_error") -> float
```

### Metric Registry

```python
register_metric(name, fn, aliases=None, overwrite=False) -> None
get_metric(name) -> MetricFn
resolve_metrics(names: list[str]) -> dict[str, MetricFn]
list_metrics() -> list[str]
make_coverage_metric(levels=None, side="two-sided", weights=None, prefix="") -> MetricFn
DEFAULT_METRICS: list[str]
```

### Metrics

```python
compute_condition_metrics(draws, true_values, cond_id, metric_fns) -> dict[str, Any]
aggregate_condition_rows(condition_rows: list[dict]) -> dict[str, float]
```

### C2ST Metrics

```python
lc2st(posterior_samples, true_params, observations, *,
      n_folds=5, n_null_trials=0, clf_kwargs=None, seed=42) -> LC2STResult

global_c2st(samples_p, samples_q, *, clf_kwargs=None, seed=42) -> GlobalC2STResult

make_lc2st_validate_fn(base_metrics=None, n_folds=5, n_null_trials=0,
                        clf_kwargs=None, seed=42) -> ValidateFn
```

`make_lc2st_validate_fn()` returns a `ValidateFn` compatible with `optimize(validate_fn=...)` that computes standard per-parameter metrics and L-C2ST from a single inference pass.

### SBC Tests

```python
compute_sbc_uniformity_tests(ranks, n_posterior_samples, n_bins=20) -> dict[str, float]
```

Returns KS statistic, KS p-value, chi-squared statistic, and chi-squared p-value for SBC rank uniformity.

---

## Results

### Extraction

```python
get_pareto_trials(study) -> list[optuna.trial.FrozenTrial]

trials_to_dataframe(study, trained_only=True, include_pruned=False,
                    extra_attrs=None, include_ranks=True) -> pd.DataFrame

trial_table(study, top_k=None, select_by=0, metrics=None,
            trained_only=True) -> pd.DataFrame

best_config(study, trial_number=None, select_by=0,
            priorities=None) -> dict[str, Any]

compare_trials(study, trial_numbers, metrics=None) -> pd.DataFrame

summarize_study(study, select_by=0) -> str

select_best_trial(study, priorities) -> tuple[FrozenTrial, SelectionResult]
```

### Visualization

```python
plot_study(study, *, third_dim="color", figsize=None, row_labels=True) -> Figure

plot_pareto_front(study, axes=None, *, third_dim="color",
                  max_cols=3, figsize=None) -> Axes

plot_optimization_history(study, axes=None, *, max_cols=3, figsize=None) -> Axes

plot_param_importance(study, axes=None, top_k=10, *,
                      max_cols=3, figsize=None) -> Axes | None

plot_metric_scatter(study, x_metric, y_metric, ax=None, *,
                    show_iso_lines=None) -> Axes

plot_metric_panels(study, metrics=None, axes=None, *,
                   max_cols=3, figsize=None) -> Axes | np.ndarray

plot_pareto_3d(study, ax=None, *, cost_display="color") -> Axes

plot_pareto_projections(study, axes=None, *, cost_display="color",
                        max_cols=3, figsize=None) -> Axes

plot_parallel_coordinates(study, ax=None, *, top_k=20, select_by=0,
                          metric_order=None) -> Axes
```

### Export

```python
get_workflow_metadata(config, model_type, validation_results=None, extra=None) -> dict
save_workflow_with_metadata(approximator, path, metadata) -> Path
load_workflow_with_metadata(path) -> tuple[Any, dict]
```

---

## Registration

```python
register_custom_inference_network(name, space_factory, builder=None, aliases=None, overwrite=False)
register_custom_summary_network(name, space_factory, builder=None, aliases=None, overwrite=False)
list_registered_network_spaces() -> dict[str, list[str]]
```

---

## Utilities

```python
get_param_count(model) -> int
normalize_param_count(param_count) -> float
denormalize_param_count(normalized) -> int
extract_objective_values(metrics, cost_score, objective_metric="calibration_error") -> tuple[float, float]
extract_multi_objective_values(metrics, cost_score, objective_metrics, objective_mode="mean") -> tuple[float, ...]
compute_inference_time_per_dataset(inference_time, n_datasets) -> float
loguniform_int(low, high, alpha=1.0, rng=None) -> int
loguniform_float(low, high, alpha=1.0, rng=None) -> float
```
