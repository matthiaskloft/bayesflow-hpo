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
    validation_data: ValidationDataset
    epochs: int = 200
    num_batches: int = 50
    early_stopping_patience: int = 5
    early_stopping_window: int = 7
    max_param_count: int = 1_000_000
    max_memory_mb: float | None = None
    metric_constraints_hard: list[MetricConstraintSpec] | None = None
    n_posterior_samples: int = 500
    n_intermediate_posterior_samples: int = 250
    intermediate_validation_interval: int = 10
    intermediate_validation_warmup: int = 10
    pruning_strategy: str | tuple[str, str] = "dominance"
    pruning_n_startup_trials: int | None = None
    objective_metrics: list[str] = field(default_factory=lambda: ["calibration_error", "nrmse"])
    objective_mode: str = "pareto"
    cost_metric: str = "inference_time"
    checkpoint_pool: CheckpointPool | None = None
    report_frequency: int = 10
    build_approximator_fn: BuildApproximatorFn | None = None
    train_fn: TrainFn | None = None
    validate_fn: ValidateFn | None = None
```

#### Configurable Validation Metrics

`objective_metrics` controls which validation metrics are optimized. It accepts a list of metric names resolved via the [metric registry](validation.md#metric-registry). Defaults to `["calibration_error", "nrmse"]`.

`objective_mode` chooses whether metrics are aggregated (`"mean"`) or optimized jointly (`"pareto"`). `cost_metric` selects the cost objective (`"inference_time"` or `"param_count"`).

#### Pruning Strategy

`pruning_strategy` controls how multi-objective trials are pruned during training. Optuna does not support `trial.report()` for multi-objective studies, so `PeriodicValidationCallback` dispatches to a custom strategy:

| Value | Behavior |
|-------|----------|
| `"dominance"` (default) | Per-objective normalized median check; prunes only if worse than median on ALL objectives. Adapted from MO-ASHA (Schmucker et al., 2021). |
| `"mo-sha"` | Non-dominated sorting at each step; prunes trials outside the top 1/η fraction. Full MO-ASHA Algorithm 2 (Schmucker et al., 2021). |
| `("primary", "metric_name")` | Single-metric median pruning on the named metric. Equivalent to Optuna's MedianPruner (Akiba et al., 2019). |
| `"none"` | No intermediate validation or pruning. |

`pruning_n_startup_trials` controls how many completed trials are needed before pruning activates. When `None` (default), it is auto-detected from the sampler: `_n_startup_trials` for TPE/GP (25/10), `population_size` for NSGA-II/III (50), fallback 10 for others.

#### Custom Training Function

`train_fn` allows users to override the default training loop. The signature is `(approximator, simulator, hparams, callbacks) -> None`. When `None`, the objective uses `approximator.fit(simulator=..., epochs=..., num_batches=..., ...)`. Example:

```python
def my_train_fn(approximator, simulator, hparams, callbacks):
    approximator.fit(
        simulator=simulator,
        epochs=int(hparams["epochs"]),
        num_batches=int(hparams["num_batches"]),  # for BF versions expecting num_batches
        batch_size=int(hparams.get("batch_size", 256)),
        callbacks=callbacks,
    )

config = ObjectiveConfig(..., train_fn=my_train_fn)
```

#### Custom Validation Function

`validate_fn` allows users to override the default validation pipeline. The signature is `(approximator, validation_data, n_posterior_samples) -> dict[str, float]`. The returned dict must include all keys listed in `objective_metrics`. When provided, `validate_fn` is also used for **intermediate validation** during training (via `PeriodicValidationCallback`), enabling mid-training pruning for custom approximator architectures.

```python
def my_validate_fn(approximator, validation_data, n_posterior_samples):
    """Custom validation for per-item (IRT-style) posteriors."""
    # ... sample from approximator, compute metrics ...
    return {"calibration_error": cal_err, "correlation": corr}

study = hpo.optimize(
    ...,
    validate_fn=my_validate_fn,
    objective_metrics=["calibration_error", "correlation"],
)
```

Without `validate_fn`, the default `run_validation_pipeline` is used for both final and intermediate validation. If the default pipeline cannot handle your approximator's output shapes (e.g., 3D per-item posteriors), intermediate validation will fail silently and pruning will be ineffective — providing a custom `validate_fn` is the fix.

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
4. **Train** using `train_fn` (default: `approximator.fit(simulator=...)`) with early stopping and pruning callbacks
5. **Validate** on the fixed validation dataset using the metric registry
6. **Return** `(objective_metric_value, normalized_param_score)`

Trials that exceed budgets or crash during training return penalty values without wasting GPU time.

For downstream packages, a customized FlowMatching search space plugs into the same composition API:

```python
import bayesflow_hpo as hpo

inference_space = hpo.FlowMatchingSpace.balanced()
search_space = hpo.CompositeSearchSpace(inference_space=inference_space)
```

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

`optimize(max_memory_mb="auto")` enables GPU-memory auto-detection:

- Uses `torch.cuda.mem_get_info()` and takes **free** VRAM (not total).
- Applies `memory_safety_margin` (default 0.2).
- Resolved formula:
  `free_bytes * (1 - safety_margin) / (1024.0 ** 2)`.
- If CUDA is unavailable, a warning is logged and memory budget is disabled.

### Metric Constraints

Metric constraints use tuple specs:

```python
(metric_name, threshold, "above" | "below")
```

- `"above"` means violation when `metric_value > threshold` (useful for metrics to minimize)
- `"below"` means violation when `metric_value < threshold` (useful for metrics to maximize like coverage)

Two layers are supported:

**Hard constraints** (`metric_constraints_hard`): Post-validation hard rejection in the objective. Trials violating hard constraints return penalty values and are marked with `rejected_reason="metric_constraint"`.

**Soft constraints** (`metric_constraints_soft`): Optuna `constraints_func` feasibility guidance for sampler presets that support constraints. Provides numerical feasibility scores to guide sampling without hard rejection.

```python
study = hpo.optimize(
    ...,
    metric_constraints_hard=[
        ("calibration_error", 0.1, "below"),  # reject if cal_error > 0.1
        ("coverage_90", 0.8, "above"),        # reject if coverage < 0.8
    ],
    metric_constraints_soft=[
        ("nrmse", 0.15, "below"),  # guide sampling away from high NRMSE
    ],
)
```

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
    directions=["minimize", "minimize"],
    storage="sqlite:///hpo.db",
    load_if_exists=True,
    sampler="gp",       # String preset or BaseSampler
    pruner="median",    # String preset or BasePruner
)
```

Both `sampler` and `pruner` accept string presets or Optuna instances.

**Sampler presets** (all use `seed=42`; budget constraints auto-wired when `budget_aware=True`):

| Preset | Sampler | Notes |
|--------|---------|-------|
| `"tpe"` | `TPESampler(multivariate=True, n_startup_trials=25)` | Default |
| `"gp"` | `GPSampler(n_startup_trials=10)` | Gaussian process BO |
| `"botorch"` | `BoTorchSampler(n_startup_trials=10)` | Requires `optuna-integration[botorch]` |
| `"nsga2"` | `NSGAIISampler(population_size=50)` | Evolutionary multi-objective |
| `"nsga3"` | `NSGAIIISampler(population_size=50)` | Reference-point-based, for >3 objectives |
| `"auto"` | `AutoSampler()` | Requires newer Optuna version |
| `"random"` | `RandomSampler()` | Baseline |
| `None` | Same as `"tpe"` | Backwards compatible |

**Pruner presets** (single-objective only):

| Preset | Pruner | Notes |
|--------|--------|-------|
| `"median"` | `MedianPruner(n_startup_trials=5, n_warmup_steps=1)` | Default-like behavior |
| `"hyperband"` | `HyperbandPruner(min_resource=1, reduction_factor=3)` | Better with TPE (Li et al., 2018) |
| `"none"` | `NopPruner()` | Disable single-objective pruning |
| `None` | Default `MedianPruner` | Existing behavior |

Note: For multi-objective studies, `trial.report()` is unsupported so the study pruner is ignored; pruning is handled by the `pruning_strategy` parameter on `optimize()`.

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
    warm_start_top_k=20,          # Top 20 trials by mean objective
)
```

Or manually:

```python
n_added = warm_start_study(target_study, source_study, top_k=20)
```

### QMC Warm-up

Replace the main sampler's random startup phase with a Sobol quasi-random sequence for better space-filling coverage:

```python
study = create_study(
    sampler="tpe",
    qmc_startup_trials=16,  # 16 Sobol trials, then TPE takes over
)
```

Or via `optimize()`:

```python
study = hpo.optimize(
    ...,
    sampler="gp",
    qmc_startup_trials=16,
)
```

**Key properties:**
- Sobol's low-discrepancy guarantee is optimal at n = 2^m (8, 16, 32, 64, ...)
- A warning is logged for non-power-of-2 values
- Only non-rejected completions count toward the QMC quota
- QMC warm-up composes with warm-start and all sampler presets:

```python
study = create_study(
    sampler="tpe",
    warm_start_from=old_study,
    warm_start_top_k=10,
    qmc_startup_trials=16,  # QMC exploration + warm-start exploitation
)
```

**References:** Sobol' (1967) introduced Sobol sequences; Joe & Kuo (2008) improved the direction numbers used by SciPy's `scipy.stats.qmc.Sobol` (and thus Optuna's `QMCSampler`).

## Trial Cleanup

After each trial, `cleanup_trial()` runs:
1. `gc.collect()` — Python garbage collection
2. `torch.cuda.empty_cache()` — Free CUDA memory (if PyTorch available)
3. `keras.backend.clear_session()` — Reset TensorFlow/Keras state (if available)

This prevents memory leaks between trials in long HPO runs.

## Approximator Construction

`build_continuous_approximator()` constructs an uncompiled `ContinuousApproximator`:

```python
from bayesflow_hpo import build_continuous_approximator

approximator = build_continuous_approximator(
    hparams=trial_params,
    adapter=adapter,
    search_space=search_space,
    checkpoint_dir=None,  # optional: load weights from checkpoint
)
```

The function:
1. Delegates to `search_space.build(hparams)` to construct inference and summary networks
2. Wraps them in a `bf.ContinuousApproximator`
3. Optionally loads weights from a checkpoint directory

The returned approximator is **uncompiled**. The trial lifecycle in `GenericObjective.__call__()` compiles it separately with an `Adam + CosineDecay` schedule:
- `CosineDecay(initial_learning_rate=initial_lr, decay_steps=max(1, total_steps))`
- `total_steps = epochs * num_batches`
