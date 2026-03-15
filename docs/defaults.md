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
| `batches_per_epoch` | **50** | Online simulation batches per epoch. |
| `max_param_count` | **1 000 000** | Trials exceeding this estimated param count are rejected. |
| `max_memory_mb` | **None** (disabled) | Peak-memory budget in MB (disabled by default). |
| `objective_metric` | **`"calibration_error"`** | Key in the validation summary used as the first HPO objective. Ignored when `objective_metrics` is set. |
| `objective_metrics` | **`None`** | List of metric keys for multi-metric optimization (overrides `objective_metric`). |
| `objective_mode` | **`"mean"`** | `"mean"` averages metrics into one scalar; `"pareto"` gives each metric its own direction. |
| `resume` | **`False`** | Continue a previously persisted study instead of starting fresh. |
| `sims_per_condition` | **200** | Simulations per condition grid point in validation data. |
| `storage` | **`"sqlite:///bayesflow_hpo.db"`** | Optuna storage for persistence & crash recovery. |
| `study_name` | **`"bayesflow_hpo"`** | Optuna study name. |
| `directions` | **`None`** (auto-derived) | Auto-derives `["minimize"] * n_objectives` from `objective_mode`. In mean mode: 2 directions; in pareto mode with N metrics: N+1 directions. |
| `warm_start_top_k` | **25** | Best trials to copy when warm-starting from another study. |
| `show_progress_bar` | **True** | Show Optuna's tqdm progress bar. |

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

| Dimension | Range | Sampled? | Fallback |
|-----------|-------|----------|----------|
| `cf_depth` | 2--8 | Always | -- |
| `cf_subnet_width` | 32--256, step 32 | Always | -- |
| `cf_subnet_depth` | 1--3 | Always | -- |
| `cf_dropout` | 0.0--0.3 | Always | -- |
| `cf_activation` | silu, relu, mish | Optional | `"mish"` |
| `cf_transform` | affine, spline | Optional | `"affine"` |
| `cf_permutation` | random, orthogonal | Optional | `"random"` |
| `cf_actnorm` | True, False | Optional | `True` |

### FlowMatchingSpace

| Dimension | Range | Sampled? | Fallback |
|-----------|-------|----------|----------|
| `fm_subnet_width` | 32--256, step 32 | Always | -- |
| `fm_subnet_depth` | 1--6 | Always | -- |
| `fm_dropout` | 0.0--0.2 | Always | -- |
| `fm_activation` | mish, silu | Optional | `"mish"` |
| `fm_use_ot` | True, False | Optional | `False` |
| `fm_time_alpha` | 0.0--2.0 | Optional | `0.0` |

### DeepSetSpace

| Dimension | Range | Sampled? | Fallback |
|-----------|-------|----------|----------|
| `ds_summary_dim` | 4--64, step 4 | Always | -- |
| `ds_depth` | 1--4 | Always | -- |
| `ds_width` | 32--256, step 32 | Always | -- |
| `ds_dropout` | 0.0--0.3 | Always | -- |
| `ds_activation` | silu, mish | Optional | `"silu"` |
| `ds_spectral_norm` | True, False | Optional | `False` |
| `ds_inner_pooling` | mean, max | Optional | `"mean"` |
| `ds_output_pooling` | mean, max | Optional | `"mean"` |

Architecture: the `invariant_outer` MLP uses `(width, summary_dim)`
as a bottleneck, matching BayesFlow's default architecture.  All other
MLPs use `(width, width)`.

### SetTransformerSpace

| Dimension | Range | Sampled? | Fallback |
|-----------|-------|----------|----------|
| `st_summary_dim` | 8--64, step 8 | Always | -- |
| `st_embed_dim` | 32--256, step 32 | Always | -- |
| `st_num_heads` | 1, 2, 4, 8 | Always | -- |
| `st_num_layers` | 1--4 | Always | -- |
| `st_dropout` | 0.0--0.3 | Always | -- |
| `st_mlp_width` | 64--512, step 64 | Optional | `2 * embed_dim` |
| `st_mlp_depth` | 1--4 | Optional | `2` |
| `st_num_inducing` | 8--64, step 8 | Optional | `None` |

### TrainingSpace

| Dimension | Range | Sampled? | Fallback |
|-----------|-------|----------|----------|
| `initial_lr` | 1e-4 -- 5e-3 (log) | Always | -- |
| `batch_size` | 32--1024, step 32 | Optional | `256` |
| `decay_rate` | 0.8--0.99 | Optional | `0.95` (only used with ExponentialDecay) |

Optional dimensions are enabled via `include_optional=True` on each
search space or by passing a custom search space.

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
| Pruning metric | `sqrt(nrmse * calibration_error)` (geometric mean) | `PeriodicValidationCallback` |
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
