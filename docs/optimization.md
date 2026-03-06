# Optimization

## Objective Function

### ObjectiveConfig

Central configuration dataclass for the objective:

```python
@dataclass
class ObjectiveConfig:
    simulator: bf.simulators.Simulator
    adapter: bf.adapters.Adapter
    search_space: CompositeSearchSpace
    inference_conditions: list[str] | None = None
    validation_data: ValidationDataset | None = None
    epochs: int = 200
    batches_per_epoch: int = 50
    early_stopping_patience: int = 15
    early_stopping_window: int = 15
    max_param_count: int = 2_000_000
    param_budget_penalty: tuple[float, float] = (1.0, 1.5)
    max_memory_mb: float | None = None
    memory_budget_penalty: tuple[float, float] = (1.0, 1.5)
    training_failure_penalty: tuple[float, float] = (1.0, 1.5)
    n_posterior_samples: int = 500
    metrics: list[str] | None = None
    objective_metric: str = "calibration_error"
    train_fn: Callable[[bf.BasicWorkflow, dict, list], None] | None = None
```

#### Configurable Validation Metrics

`metrics` controls which metrics are computed on the fixed validation dataset. It accepts a list of metric names resolved via the [metric registry](validation.md#metric-registry). Defaults to `DEFAULT_METRICS = ["calibration_error", "coverage", "rmse", "contraction", "sbc"]`.

`objective_metric` selects which key from the validation summary drives the HPO objective. Defaults to `"calibration_error"`.

#### Custom Training Function

`train_fn` allows users to override the default training loop. The signature is `(workflow, params, callbacks) -> None`. When `None`, the objective uses `workflow.fit_online(...)`. Example:

```python
def my_train_fn(workflow, params, callbacks):
    workflow.fit_offline(data=my_data, epochs=100, callbacks=callbacks)

config = ObjectiveConfig(..., train_fn=my_train_fn)
```

#### Training Failure Handling

When training raises an exception, the objective:
1. Logs the error via `logging.warning`
2. Stores the message as `trial.set_user_attr("training_error", str(exc))`
3. Returns `training_failure_penalty` (default: `(1.0, 1.5)`)

### GenericObjective

Callable that implements the Optuna trial loop:

1. **Sample hyperparameters** from the composite search space via `trial.suggest_*`
2. **Pre-filter** by estimated parameter count and peak memory
3. **Build** inference network, summary network, and workflow
4. **Train** using `train_fn` (default: `fit_online`) with early stopping and pruning callbacks
5. **Validate** on the fixed validation dataset using the metric registry
6. **Return** `(objective_metric_value, normalized_param_score)`

Trials that exceed budgets or crash during training return penalty values without wasting GPU time.

## Constraints

### Parameter Budget

`estimate_param_count(params)` provides a heuristic parameter count from the sampled hyperparameters, dispatching by network-specific prefixes (`cf_`, `fm_`, `dm_`, etc.). Trials exceeding `max_param_count` are rejected before any network is built.

### Memory Budget

`estimate_peak_memory_mb(params)` estimates peak training memory as:

```
memory ≈ (4 × param_count × dtype_bytes)     # weights + grads + Adam states
        + (3 × activation_elements × dtype_bytes)  # activations
```

Trials exceeding `max_memory_mb` are rejected before training.

### Penalty Values

When a trial is rejected or crashes, the objective returns:
- **Quality metric** = 1.0 (worst possible)
- **Param score** = 1.5 (above the normalized range [0, 1])

This ensures rejected trials are dominated by any successful trial in the Pareto front.

## Callbacks

### OptunaReportCallback

Keras callback that reports a monitored metric (default: `loss`) to Optuna after each epoch. Enables Optuna's median pruner to terminate unpromising trials early.

```python
OptunaReportCallback(trial, monitor="loss", report_frequency=1)
```

### MovingAverageEarlyStopping

Stops training when the moving average of a metric stops improving:

```python
MovingAverageEarlyStopping(
    monitor="loss",
    window=5,       # Smoothing window size
    patience=3,     # Epochs without improvement before stopping
    restore_best_weights=True,
)
```

The moving average prevents noisy loss curves from triggering premature stops.

## Study Management

### Creating a Study

```python
study = create_study(
    study_name="my_hpo",
    directions=["minimize", "minimize"],  # quality metric, param_score
    storage="sqlite:///hpo.db",           # Persistent storage (optional)
    load_if_exists=True,                  # Resume interrupted runs
    sampler=optuna.samplers.TPESampler(multivariate=True),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)
```

Defaults: `TPESampler(multivariate=True)` and `MedianPruner(n_startup_trials=10, n_warmup_steps=20)`.

### Resuming a Study

```python
study = resume_study("my_hpo", storage="sqlite:///hpo.db")
```

### Warm-Starting

Seed a new study with the best trials from a previous study:

```python
study = create_study(
    study_name="new_study",
    warm_start_from=old_study,
    warm_start_top_k=20,          # Top 20 trials by metric
    warm_start_metric_index=0,    # Sort by first objective
)
```

Or manually:

```python
n_added = warm_start_study(target_study, source_study, top_k=20)
```

## Trial Cleanup

After each trial, `cleanup_trial()` runs:
1. `gc.collect()` — Python garbage collection
2. `torch.cuda.empty_cache()` — Free CUDA memory (if PyTorch available)
3. `keras.backend.clear_session()` — Reset TensorFlow/Keras state (if available)

This prevents memory leaks between trials in long HPO runs.

## Workflow Construction

`build_workflow` delegates to `bf.BasicWorkflow`:

```python
workflow = build_workflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_net,
    summary_network=summary_net,
    params=params,
    config=WorkflowBuildConfig(
        batches_per_epoch=50,
        optimizer=None,  # or a custom keras optimizer
    ),
)
```

When `config.optimizer` is `None`, an `ExponentialDecay + Adam` schedule is created:
- `decay_steps = max(1, config.batches_per_epoch)` — decays once per epoch
- `decay_rate` from `params.get("decay_rate", 0.95)`
- `staircase=True`

When a custom optimizer is provided, it is passed directly to `bf.BasicWorkflow`.
