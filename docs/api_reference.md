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
    epochs=200, batches_per_epoch=50,
    early_stopping_patience=5, early_stopping_window=7,
    # Budget
    max_param_count=1_000_000, max_memory_mb=None,
    # Study
    n_trials=50, max_total_trials=None,
    study_name="bayesflow_hpo", storage=DEFAULT_STORAGE, resume=False,
    # Optional
    directions=None, warm_start_from=None, warm_start_top_k=25,
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
| `epochs` | Max training epochs per trial (default 200). |
| `batches_per_epoch` | Online simulation batches per epoch (default 50). |
| `early_stopping_patience` | Moving-average patience for early stopping (default 5). |
| `early_stopping_window` | Moving-average window size (default 7). |
| `max_param_count` | Reject trials exceeding this param count pre-training (default 1 000 000). |
| `max_memory_mb` | Optional peak-memory budget in MB. Disabled by default. |
| `n_trials` | Number of *trained* trials to collect (default 50). |
| `max_total_trials` | Hard cap on non-rejected trials. Defaults to `3 * n_trials`. |
| `resume` | If `True`, continue a previously persisted study. |
| `checkpoint_pool` | Optional `CheckpointPool` for persisting best trial weights. |

### `check_pipeline(...)`

Pre-flight validation that catches interface errors before launching expensive studies. Called automatically at the start of `optimize()`.

```python
def check_pipeline(
    simulator, adapter, search_space,
    build_approximator_fn=None, train_fn=None, validate_fn=None,
    objective_metrics=("calibration_error", "nrmse"),
    sims_per_condition=5, n_posterior_samples=2,
    validation_conditions=None, epochs=1, batches_per_epoch=1,
) -> None
```

Raises `PipelineError` on builder failures, missing metric keys, or signature mismatches.

### `infer_keys_from_adapter(adapter) -> dict`

Walks the adapter's transform list to infer `param_keys`, `data_keys`, and `inference_conditions`.

---

## Type Aliases

```python
BuildApproximatorFn = Callable[[dict[str, Any]], Any]
TrainFn = Callable[[Any, bf.simulators.Simulator, dict[str, Any], list], None]
ValidateFn = Callable[[Any, ValidationDataset, int], dict[str, float]]
```

---

## Search Spaces

### Dimension Types

| Class | Purpose |
|-------|---------|
| `IntDimension(name, low, high, step, log, default)` | Integer hyperparameter |
| `FloatDimension(name, low, high, log, default)` | Float hyperparameter |
| `CategoricalDimension(name, choices, default)` | Categorical hyperparameter |

### Inference Spaces

| Class | BayesFlow Network |
|-------|-------------------|
| `CouplingFlowSpace(include_optional=False)` | `bf.networks.CouplingFlow` |
| `FlowMatchingSpace(include_optional=False)` | `bf.networks.FlowMatching` |
| `DiffusionModelSpace(include_optional=False)` | `bf.networks.DiffusionModel` |
| `ConsistencyModelSpace(epochs, batches_per_epoch, include_optional=False)` | `bf.networks.ConsistencyModel` |
| `StableConsistencyModelSpace(include_optional=False)` | `bf.networks.StableConsistencyModel` |

### Summary Spaces

| Class | BayesFlow Network |
|-------|-------------------|
| `DeepSetSpace(include_optional=False)` | `bf.networks.DeepSet` |
| `SetTransformerSpace(include_optional=False)` | `bf.networks.SetTransformer` |
| `TimeSeriesNetworkSpace(include_optional=False)` | `bf.networks.TimeSeriesNetwork` |
| `TimeSeriesTransformerSpace(include_optional=False)` | `bf.networks.TimeSeriesTransformer` |
| `FusionTransformerSpace(include_optional=False)` | `bf.networks.FusionTransformer` |

### Training Space

| Class | Controls |
|-------|----------|
| `TrainingSpace(include_optional=False)` | `initial_lr`, `batch_size`, `decay_rate` |

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
build_continuous_approximator(hparams, adapter, search_space) -> ContinuousApproximator
```

Builds an **uncompiled** `ContinuousApproximator` from sampled hyperparameters. Handles inference + optional summary network construction.

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
| `validation_data` | `None` | Pre-generated `ValidationDataset` |
| `epochs` | `200` | Max training epochs per trial |
| `batches_per_epoch` | `50` | Online batches per epoch |
| `early_stopping_patience` | `5` | Moving-average patience |
| `early_stopping_window` | `7` | Moving-average window |
| `max_param_count` | `1_000_000` | Pre-training param budget |
| `max_memory_mb` | `None` | Peak-memory budget (disabled) |
| `n_posterior_samples` | `500` | Posterior draws for final validation |
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
create_study(study_name, directions, storage, load_if_exists, sampler, pruner,
             metric_names, warm_start_from, warm_start_top_k) -> optuna.Study
optimize_until(study, objective, n_trained, max_total_trials, show_progress_bar) -> None
warm_start_study(target_study, source_study, top_k, metric_index) -> int
```

### Callbacks

```python
OptunaReportCallback(trial, monitor="loss", report_frequency=1)
MovingAverageEarlyStopping(monitor="loss", window=5, patience=3, restore_best_weights=True)
PeriodicValidationCallback(trial, approximator, validation_data, ...)
```

### Constraints

```python
estimate_peak_memory_mb(params, batch_size=None, dtype_bytes=4) -> float
exceeds_memory_budget(params, max_memory_mb, batch_size=None) -> bool
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

---

## Results

```python
get_pareto_trials(study) -> list[optuna.trial.FrozenTrial]
trials_to_dataframe(study, trained_only=True, include_pruned=False, extra_attrs=None) -> pd.DataFrame
summarize_study(study, select_by=0, top_k=5) -> str
plot_pareto_front(study, ax=None) -> matplotlib.axes.Axes
plot_optimization_history(study, ax=None) -> matplotlib.axes.Axes
plot_metric_scatter(study, x_metric, y_metric, ax=None, show_iso_lines=None) -> matplotlib.axes.Axes
plot_metric_panels(study, metrics=None, ax=None) -> matplotlib.axes.Axes | np.ndarray
plot_param_importance(study, top_k=10, ax=None, target_name=None) -> matplotlib.axes.Axes
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
extract_objective_values(metrics: dict, param_count: int) -> tuple[float, float]
extract_multi_objective_values(metrics, param_count, metric_keys, mode, cost_metric, ...) -> tuple[float, ...]
compute_inference_time_ratio(inference_time_s, sim_time_per_sim, n_sims) -> float
loguniform_int(low, high, alpha=1.0, rng=None) -> int
loguniform_float(low, high, alpha=1.0, rng=None) -> float
```
