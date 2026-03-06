# API Reference

Complete list of public symbols exported from `bayesflow_hpo`.

## High-Level API

### `optimize(...) -> optuna.Study`

One-call convenience function that creates a search space, generates validation data, builds an objective, and runs the study.

```python
def optimize(
    simulator, adapter, param_keys, data_keys,
    validation_data=None, validation_conditions=None, sims_per_condition=200,
    search_space=None, inference_conditions=None,
    n_trials=100, epochs=200, batches_per_epoch=50,
    max_param_count=2_000_000, max_memory_mb=None,
    metrics=None, objective_metric="calibration_error",
    train_fn=None,
    storage=None, study_name="bayesflow_hpo",
    directions=None, warm_start_from=None,
    warm_start_top_k=20, warm_start_metric_index=0,
    show_progress_bar=True,
) -> optuna.Study
```

Defaults to `CouplingFlowSpace + DeepSetSpace + TrainingSpace` when no search space is provided.

| Parameter | Description |
|-----------|-------------|
| `metrics` | List of metric names for validation (resolved via registry). Defaults to `DEFAULT_METRICS`. |
| `objective_metric` | Key in the validation summary used as the HPO objective. |
| `train_fn` | Custom training function `(workflow, params, callbacks) -> None`. Defaults to `fit_online`. |

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
build_inference_network(params, search_space) -> bf.networks.InferenceNetwork
build_summary_network(params, search_space) -> bf.networks.SummaryNetwork | None
build_workflow(simulator, adapter, inference_network, summary_network, params, config) -> bf.BasicWorkflow
```

### WorkflowBuildConfig

```python
@dataclass
class WorkflowBuildConfig:
    inference_conditions: list[str] | None = None
    checkpoint_name: str = "bayesflow_hpo_trial"
    batches_per_epoch: int = 50
    optimizer: Any | None = None
```

---

## Optimization

### ObjectiveConfig

See [Optimization docs](optimization.md) for full field listing. Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `metrics` | `None` | Metric names for validation (defaults to `DEFAULT_METRICS`) |
| `objective_metric` | `"calibration_error"` | Summary key used as HPO objective |
| `train_fn` | `None` | Custom training function (defaults to `fit_online`) |
| `training_failure_penalty` | `(1.0, 1.5)` | Return value when training fails |

### GenericObjective

```python
objective = GenericObjective(config: ObjectiveConfig)
quality_metric, param_score = objective(trial: optuna.Trial)
```

### Study Management

```python
create_study(study_name, directions, storage, load_if_exists, sampler, pruner,
             warm_start_from, warm_start_top_k, warm_start_metric_index) -> optuna.Study
resume_study(study_name, storage) -> optuna.Study
warm_start_study(target_study, source_study, top_k, metric_index) -> int
```

### Callbacks

```python
OptunaReportCallback(trial, monitor="loss", report_frequency=1)
MovingAverageEarlyStopping(monitor="loss", window=5, patience=3, restore_best_weights=True)
```

### Constraints

```python
estimate_param_count(params) -> int
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
trials_to_dataframe(study, include_pruned=False) -> pd.DataFrame
plot_pareto_front(study, ax=None) -> matplotlib.axes.Axes
plot_param_importance(study, ax=None, top_k=10) -> matplotlib.axes.Axes
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
loguniform_int(low, high, alpha=1.0, rng=None) -> int
loguniform_float(low, high, alpha=1.0, rng=None) -> float
```
