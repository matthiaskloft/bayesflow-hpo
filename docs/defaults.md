# Default Configuration Reference

This document lists every default value used by `optimize()` and the
subsystems it orchestrates.  All defaults can be overridden by passing
explicit arguments to `optimize()` or by constructing lower-level
objects (e.g. `ObjectiveConfig`, `create_study`) directly.

---

## High-level API (`optimize()`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_trials` | **50** | Number of *trained* trials to collect (budget-rejected trials don't count). |
| `max_total_trials` | **3 &times; n_trials** | Hard cap on total trials including budget-rejected. |
| `epochs` | **200** | Maximum training epochs per trial. |
| `num_batches` | **50** | Online simulation batches per epoch. |
| `max_param_count` | **1 000 000** | Trials exceeding this estimated param count are rejected. |
| `max_memory_mb` | **None** (disabled) | Peak-memory budget in MB, or `"auto"` for CUDA free-memory auto-detection. |
| `metric_constraints_hard` | **None** | Hard metric constraints (post-validation rejection). |
| `metric_constraints_soft` | **None** | Soft metric constraints (feasibility guidance for sampler presets). |
| `memory_safety_margin` | **0.2** | Safety margin for `max_memory_mb="auto"`. |
| `objective_metrics` | **`["calibration_error", "nrmse"]`** | List of metric keys to optimize. |
| `objective_mode` | **`"pareto"`** | `"pareto"` gives each metric its own Pareto direction; `"mean"` averages metrics into one scalar. |
| `cost_metric` | **`"inference_time"`** | Cost objective (`"inference_time"` or `"param_count"`). |
| `pruning_strategy` | **`"dominance"`** | Multi-objective pruning strategy (`"dominance"`, `"mo-sha"`, `("primary", metric)`, `"none"`). |
| `pruning_n_startup_trials` | **None** (auto-detect) | Min completed trials before pruning. Auto-detects from sampler when None. |
| `sampler` | **`None`** (= `"tpe"`) | Sampler preset or instance. |
| `pruner` | **`None`** | Pruner preset or instance. |
| `resume` | **`False`** | Continue a previously persisted study instead of starting fresh. |
| `sims_per_condition` | **200** | Simulations per condition grid point in validation data. |
| `storage` | **`"sqlite:///bayesflow_hpo.db"`** | Optuna storage for persistence & crash recovery. |
| `study_name` | **`"bayesflow_hpo"`** | Optuna study name. |
| `directions` | **`None`** (auto-derived) | Auto-derives `["minimize"] * n_objectives` from `objective_mode`. In mean mode: 2 directions; in pareto mode with N metrics: N+1 directions. |
| `warm_start_top_k` | **25** | Best trials to copy when warm-starting from another study. |
| `qmc_startup_trials` | **0** (disabled) | Number of initial Sobol QMC trials before the main sampler takes over. |
| `show_progress_bar` | **True** | Show Optuna's tqdm progress bar. |
| `report_frequency` | **10** | Optuna report frequency (epochs). |

---

## Default Search Space

When `search_space=None`, `optimize()` creates:

```python
CompositeSearchSpace(
    inference_space=NetworkSelectionSpace({
        "coupling_flow": CouplingFlowSpace(),
        "flow_matching": FlowMatchingSpace(),
    }),
    summary_space=SummarySelectionSpace({
        "deep_set": DeepSetSpace(),
        "set_transformer": SetTransformerSpace(),
    }),
    training_space=TrainingSpace(),
)
```

Optuna selects the network type as a categorical hyperparameter, then
samples the corresponding network-specific dimensions.

### CouplingFlowSpace

| Dimension | Range | Tuned | Constant |
|-----------|-------|-------|----------|
| `cf_depth` | 2--8 | yes | — |
| `cf_subnet_width` | 32--256, step 32 | yes | — |
| `cf_subnet_depth` | 1--3 | yes | — |
| `cf_dropout` | 0.0--0.3 | yes | — |
| `cf_activation` | silu, relu, mish | no | `"silu"` |
| `cf_transform` | affine, spline | no | `"affine"` |
| `cf_permutation` | random, orthogonal | no | `"random"` |
| `cf_use_actnorm` | True, False | no | `True` |

### FlowMatchingSpace

| Dimension | Range | Tuned | Constant |
|-----------|-------|-------|----------|
| `fm_subnet_width` | 32--256, step 32 | yes | — |
| `fm_subnet_depth` | 1--6 | yes | — |
| `fm_dropout` | 0.0--0.2 | yes | — |
| `fm_activation` | — | no | `"mish"` |
| `fm_use_optimal_transport` | — | no | `False` |
| `fm_time_power_law_alpha` | — | no | `0.0` |
| `fm_time_embedding_dim` | — | no | `32` |
| `fm_integrate_method` | — | no | `"tsit5"` |
| `fm_integrate_steps` | — | no | `"adaptive"` |
| `fm_merge` | — | no | `"concat"` |
| `fm_norm` | — | no | `"layer"` |
| `fm_residual` | — | no | `True` |
| `fm_spectral_normalization` | — | no | `False` |
| `fm_kernel_initializer` | — | no | `"he_normal"` |

Profile helpers:
- `FlowMatchingSpace.fast()`
- `FlowMatchingSpace.balanced()`
- `FlowMatchingSpace.quality()`
- `FlowMatchingSpace.preset("default" | "fast" | "balanced" | "quality")`

### DeepSetSpace

| Dimension | Range | Tuned | Constant |
|-----------|-------|-------|----------|
| `ds_summary_dim` | 4--64, step 4 | yes | — |
| `ds_depth` | 1--4 | yes | — |
| `ds_width` | 32--256, step 32 | yes | — |
| `ds_dropout` | 0.0--0.3 | yes | — |
| `ds_activation` | silu, mish | no | `"silu"` |
| `ds_spectral_norm` | True, False | no | `False` |

Architecture: the `invariant_outer` MLP uses `(width, summary_dim)`
as a bottleneck, matching BayesFlow's default architecture.  All other
MLPs use `(width, width)`.  `inner_pooling="mean"` and `output_pooling="mean"`
are hardcoded in `build()`.

### SetTransformerSpace

| Dimension | Range | Tuned | Constant |
|-----------|-------|-------|----------|
| `st_summary_dim` | 8--64, step 8 | yes | — |
| `st_embed_dim` | 32--256, step 32 | yes | — |
| `st_num_heads` | 1, 2, 4, 8 | yes | — |
| `st_num_layers` | 1--4 | yes | — |
| `st_dropout` | 0.0--0.3 | yes | — |
| `st_mlp_width` | 64--512, step 64 | no | `128` |
| `st_mlp_depth` | 1--4 | no | `2` |
| `st_num_inducing` | 8--64, step 8 | no | `None` |

### TrainingSpace

| Dimension | Range | Tuned | Constant |
|-----------|-------|-------|----------|
| `initial_lr` | 1e-4 -- 5e-3 (log) | yes | — |
| `batch_size` | 32--1024, step 32 | no | `256` |
| `decay_rate` | 0.8--0.99 | no | `0.95` |

Constant dimensions can be made tunable by setting `constant=_UNSET`
on individual dimensions or creating the space with overridden fields.

---

## Training Loop

| Setting | Default | Location |
|---------|---------|----------|
| Optimizer | **Adam + CosineDecay** | `build_workflow()` |
| LR schedule | `CosineDecay(initial_lr, total_steps)` | `build_workflow()` |
| Batch size (when not tuned) | **256** | `_default_train_fn()` |
| Early stopping window | **7** | `ObjectiveConfig` |
| Early stopping patience | **5** | `ObjectiveConfig` |
| Early stopping monitor | `"loss"` | `GenericObjective` |
| Restore best weights | `True` | `MovingAverageEarlyStopping` |
| Stagnation detection | ~12 epochs | window + patience |

---

## Optuna Study

| Setting | Default | Location |
|---------|---------|----------|
| Sampler | `TPESampler(seed=42, multivariate=True, n_startup_trials=25)` | `create_study()` |
| Budget-aware constraints | Enabled | `_budget_constraints_func` |
| Pruner | `MedianPruner(n_startup=5, n_warmup=1, interval=1)` | `create_study()` |
| Pruning strategy | `"dominance"` (per-objective normalized median) | `PeriodicValidationCallback` |
| Pruning schedule | Every 10 epochs after 10-epoch warmup | `PeriodicValidationCallback` |
| Intermediate posterior samples | **250** | `ObjectiveConfig` |
| Batch loop size | `max(1, n_trials // 4)` | `optimize_until()` |
| GC after trial | `True` | `optimize_until()` |
| Warm-start ranking | Arithmetic mean of objective values (excl. param_score) | `_mean_ranking_key` |
| `load_if_exists` | `True` | `create_study()` |

---

## Validation

| Setting | Default | Location |
|---------|---------|----------|
| `DEFAULT_METRICS` | calibration_error, nrmse, correlation, coverage, rmse, contraction | `registry.py` |
| Coverage levels | `[0.9, 0.95, 0.975, 0.99]` | `DEFAULT_COVERAGE_LEVELS` |
| Coverage weights | Uniform | `make_coverage_metric()` |
| Posterior samples (final) | **500** | `ObjectiveConfig` |
| Posterior samples (intermediate) | **250** | `ObjectiveConfig` |
| Validation dataset seed | **42** | `generate_validation_dataset()` |

---

## Budget & Penalties

| Setting | Default | Location |
|---------|---------|----------|
| `max_param_count` | **1 000 000** | `optimize()` |
| `max_memory_mb` | **None** (disabled) | `optimize()` |
| Failed-trial calibration error | **1.0** | `FAILED_TRIAL_CAL_ERROR` |
| Failed-trial cost penalty | **1e6** | `FAILED_TRIAL_COST` |
| Failed-trial param score | **1.01** | `FAILED_TRIAL_PARAM_SCORE` |
| Param normalization | `log10(count/1K) / log10(1M/1K)` (0--1) | `normalize_param_count()` |
| Min param reference | **1 000** | `MIN_PARAM_COUNT` |
| Max param reference | **1 000 000** | `MAX_PARAM_COUNT` |

---

## Checkpoint Pool

| Setting | Default | Location |
|---------|---------|----------|
| Pool size | **5** | `CheckpointPool` |
| Pool directory | `checkpoints/` | `CheckpointPool` |
| Behavior | Keep best 5 trial weights, auto-evict worst | `CheckpointPool.maybe_save()` |

---

## Results Table

`DEFAULT_RESULT_ATTRS` controls which trial user-attributes appear as
columns in `trials_to_dataframe()`:

- `param_count`, `training_time_s`, `inference_time_s`
- `calibration_error`, `nrmse`, `correlation`
- `training_error`, `rejected_reason`
