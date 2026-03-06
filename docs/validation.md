# Validation

## Fixed Validation Datasets

To ensure fair comparison across architectures, `bayesflow-hpo` generates a fixed validation dataset once and reuses it for every trial.

### Generation

```python
from bayesflow_hpo import generate_validation_dataset

val_data = generate_validation_dataset(
    simulator=simulator,
    param_keys=["theta"],
    data_keys=["x"],
    condition_grid={"N": [50, 100, 200]},  # Conditions to cross
    sims_per_condition=200,
    seed=42,
)
```

This produces `len(condition_grid_product) x sims_per_condition` total simulations. Each simulation is a dict mapping keys to NumPy arrays.

### Grid Helpers

Build condition grids from convenience specs:

```python
from bayesflow_hpo import make_condition_grid

# Linear spacing
grid = make_condition_grid(linspace={"N": (10, 100, 5)})
# {"N": [10.0, 32.5, 55.0, 77.5, 100.0]}

# Log spacing (raw values, not exponents)
grid = make_condition_grid(logspace={"lr": (1e-4, 1e-1, 4)})

# Explicit values
grid = make_condition_grid(values={"method": ["A", "B", "C"]})

# Combined
grid = make_condition_grid(
    linspace={"N": (10, 50, 3)},
    values={"group": [1, 2]},
)
```

One-step dataset creation:

```python
from bayesflow_hpo import make_validation_dataset

val_data = make_validation_dataset(
    simulator=simulator,
    param_keys=["theta"],
    data_keys=["x"],
    linspace={"N": (50, 200, 4)},
    sims_per_condition=200,
    seed=42,
)
```

### The ValidationDataset Dataclass

```python
@dataclass(frozen=True)
class ValidationDataset:
    simulations: list[dict[str, np.ndarray]]
    condition_labels: list[dict[str, float | int]]
    param_keys: list[str]
    data_keys: list[str]
    seed: int
```

### Persistence

```python
from bayesflow_hpo import save_validation_dataset, load_validation_dataset

save_validation_dataset(val_data, "val_data/")
val_data = load_validation_dataset("val_data/")
```

Saves `metadata.json` (keys, seed, condition labels) and `arrays.npz` (all simulation arrays).

## Metric Registry

The validation pipeline uses a registry to map string names to metric functions. All metrics share one signature:

```python
MetricFn = Callable[[np.ndarray, np.ndarray], dict[str, float]]
# (draws: [n_sims, n_samples], true_values: [n_sims]) -> {"key": value, ...}
```

### Built-in Metrics

#### BayesFlow Diagnostic Wrappers

These wrap `bf.diagnostics.*` functions, reshaping `(n_sims, n_samples)` to the BF-expected `(n_sims, n_samples, 1)` format:

| Name | Wraps | Output Keys |
|------|-------|-------------|
| `calibration_error` | `bf.diagnostics.calibration_error` | `calibration_error` |
| `rmse` | `bf.diagnostics.root_mean_squared_error` | `rmse` |
| `nrmse` | `bf.diagnostics.root_mean_squared_error(normalize="range")` | `nrmse` |
| `contraction` | `bf.diagnostics.posterior_contraction` | `contraction` |
| `z_score` | `bf.diagnostics.posterior_z_score` | `mean_abs_z_score`, `mean_z_score` |
| `log_gamma` | `bf.diagnostics.calibration_log_gamma` | `log_gamma` |

#### Native Metrics

| Name | Description | Output Keys |
|------|-------------|-------------|
| `sbc` | SBC rank uniformity tests (KS, chi-squared, C2ST) | `sbc_ks_pvalue`, `sbc_chi2_pvalue`, `sbc_c2st_accuracy` |
| `coverage` | Two-sided SBC rank-based calibration | `coverage_50`, ..., `coverage_99`, `mean_cal_error` |
| `coverage_left` | Left-sided coverage (efficiency for RCTs) | `left_coverage_50`, ..., `left_mean_cal_error` |
| `coverage_right` | Right-sided coverage (futility for RCTs) | `right_coverage_50`, ..., `right_mean_cal_error` |
| `bias` | Mean signed error of posterior mean | `bias` |
| `mae` | Mean absolute error of posterior mean | `mae` |

Aliases: `cal_error` -> `calibration_error`, `coverage_two_sided` -> `coverage`.

Default set: `DEFAULT_METRICS = ["calibration_error", "coverage", "rmse", "contraction", "sbc"]`

### SBC Rank-Based Coverage

Coverage is computed via SBC rank statistics. For each simulation, the rank is `sum(draws < true_value)`, normalized to `[0, 1]` by dividing by `n_samples + 1`.

- **Two-sided** (standard calibration): fraction of normalized ranks in `[alpha/2, 1 - alpha/2]`
- **Left-sided** (efficiency): fraction where `normalized_rank <= level`
- **Right-sided** (futility): fraction where `normalized_rank >= 1 - level`

The **calibration error** per level is `|empirical_coverage - nominal_level|`, averaged (optionally weighted) across levels.

### Custom Coverage Metrics

Use the factory for custom level sets, sidedness, or weighting:

```python
from bayesflow_hpo import make_coverage_metric

# Custom levels with emphasis on tails
fn = make_coverage_metric(
    levels=[0.9, 0.95, 0.99],
    weights=[1.0, 2.0, 3.0],
    side="two-sided",
)
result = fn(draws, true_values)
# {"coverage_90": 0.88, "coverage_95": 0.93, "coverage_99": 0.98, "mean_cal_error": 0.027}
```

### Custom Metrics

Register user-defined metrics:

```python
from bayesflow_hpo import register_metric

def my_metric(draws, true_values):
    return {"my_key": float(...)}

register_metric("my_metric", my_metric)
```

List available metrics:

```python
from bayesflow_hpo import list_metrics
print(list_metrics())
# ['bias', 'calibration_error', 'contraction', 'coverage', 'coverage_left', ...]
```

## Validation Pipeline

`run_validation_pipeline` orchestrates the full evaluation:

```python
from bayesflow_hpo import run_validation_pipeline

result = run_validation_pipeline(
    approximator=workflow.approximator,
    validation_data=val_data,
    n_posterior_samples=1000,
    metrics=["calibration_error", "coverage", "rmse"],  # or None for defaults
)
```

### Pipeline Steps

1. **Resolve metrics** — maps metric names to functions via the registry
2. **Inference** — `make_bayesflow_infer_fn` wraps the approximator to produce posterior draws
3. **Per-condition metrics** — for each condition batch, run all metric functions
4. **Aggregation** — average numeric values across conditions
5. **GPU cleanup** — free memory after each condition via `cleanup_trial()`

### Return Type: ValidationResult

```python
@dataclass(frozen=True)
class ValidationResult:
    condition_metrics: pd.DataFrame       # one row per condition
    summary: dict[str, float]             # mean across conditions
    per_parameter: dict[str, ValidationResult] | None  # multi-param models
    timing: dict[str, float]              # "inference" and "metrics" seconds
    n_conditions: int
    n_posterior_samples: int
    metric_names: list[str]
```

### Table Methods

```python
# Single-row overall summary
result.summary_table()

# Per-condition DataFrame, optionally filtered
result.condition_table()                  # all columns
result.condition_table(metric="coverage") # only columns containing "coverage"

# Per-parameter summary (multi-parameter models)
result.parameter_table()
```

### Objective Extraction

```python
# For HPO: extract a single scalar
result.objective_scalar("calibration_error")  # default key
result.objective_scalar("mean_cal_error")     # or any summary key
```

Falls back to `mean_cal_error` then `1.0` if the key is missing.

### Multi-Parameter Support

For models with multiple inference parameters (e.g., `param_keys=["mu", "sigma"]`), the pipeline computes metrics per parameter and stores sub-results in `per_parameter`:

```python
result.per_parameter["mu"].summary     # {"calibration_error": 0.02, ...}
result.per_parameter["sigma"].summary  # {"calibration_error": 0.05, ...}
result.summary                         # average across parameters
```

## Dry-Run Validation

Catch shape mismatches and key errors before a full HPO run:

```python
from bayesflow_hpo import validate_once

result = validate_once(
    approximator=workflow.approximator,
    validation_data=val_data,
    n_sims=2,                 # just 2 simulations
    n_posterior_samples=10,   # just 10 draws
    metrics=["calibration_error"],
)
```

Slices the first condition to `n_sims` rows and wraps any error with a descriptive message including `param_keys` and `data_keys`.

## SBC Tests

### Rank Uniformity

If the posterior is well-calibrated, SBC ranks should be uniform over `[0, n_posterior_samples]`.

- **KS test** — Kolmogorov-Smirnov test against `Uniform(0, n_posterior_samples)`
- **Chi-squared test** — Binned chi-squared test of rank histogram

Both return p-values; low p-values indicate miscalibration.

### Classifier Two-Sample Test (C2ST)

Requires `scikit-learn`. Trains a random forest to distinguish SBC ranks from uniform samples. Returns accuracy — values near 0.5 indicate good calibration.

```python
from bayesflow_hpo.validation.sbc_tests import compute_sbc_c2st

result = compute_sbc_c2st(ranks, n_posterior_samples, n_folds=5)
# {"sbc_c2st_accuracy": 0.52, "sbc_c2st_std": 0.03}
```
