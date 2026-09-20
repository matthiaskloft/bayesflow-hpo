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
    sim_time_per_sim: float | None = None
```

### Persistence

```python
from bayesflow_hpo import save_validation_dataset, load_validation_dataset

save_validation_dataset(val_data, "val_data/")
val_data = load_validation_dataset("val_data/")
```

Saves `metadata.json` (keys, seed, condition labels) and `arrays.npz` (all simulation arrays).

## Reducing the validation grid

`optimize()`, `validate_once()`, and `run_validation_pipeline()` accept
`aggregate`. The default, `"mean"`, preserves the arithmetic mean across
conditions followed by the arithmetic mean across parameter types.
A scalar `"worst"` or `"geometric"` changes the condition reduction within
each parameter, while retaining the mean across parameter types.

Use a mapping to retain both axes for selected metrics:

```python
result = run_validation_pipeline(
    approximator,
    val_data,
    aggregate={"nrmse": "geometric", "calibration_error": "worst"},
)
```

Each named metric is reduced over all parameter-type × condition cells,
with equal weight per cell. Unspecified metrics retain the default mean.
Keys name metric outputs (or their registered aliases), such as `nrmse`,
not metric groups such as `coverage`. Joint metrics have one value per
condition and are reduced over conditions only, without parameter duplication.
The per-parameter and per-condition tables remain available for inspection.

`"worst"` takes the maximum for lower-is-better metrics and the minimum for
registered higher-is-better metrics. `"geometric"` computes
`exp(mean(log(values)))`, following [SciPy's geometric mean definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gmean.html).
It requires **strictly positive** values: zero or negative values raise
`ValueError`; no clipping or epsilon is applied. Prefer a per-metric mapping
because signed diagnostics and metrics that can be zero do not satisfy this
domain. All reductions omit NaNs; all-NaN cells yield NaN, following the
[NumPy `nanmean` convention](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html).
A failed joint metric remains absent rather than being averaged over its
successful conditions.

The chosen reduction also applies to built-in intermediate validation used
for pruning and early stopping. Custom `validate_fn` hooks return already
reduced scores and must implement their own aggregation; non-default
`aggregate` settings cannot be combined with them.

The study records its aggregation settings. Resuming or warm-starting through
`optimize()` with different settings raises an error before trial training.
Studies without this metadata are treated as using the historical mean.

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
| `calibration_error` | `bf.diagnostics.calibration_error` (default `aggregation=np.median`) | `calibration_error` |
| `mean_calibration_error` | `bf.diagnostics.calibration_error(aggregation=np.mean)` | `mean_calibration_error` |
| `rmse` | `bf.diagnostics.root_mean_squared_error` | `rmse` |
| `nrmse` | `bf.diagnostics.root_mean_squared_error(normalize="range")` | `nrmse` |
| `contraction` | `bf.diagnostics.posterior_contraction` | `contraction` |
| `z_score` | `bf.diagnostics.posterior_z_score` (diagnostic) | `mean_abs_z_score`, `mean_z_score` |
| `log_gamma` | `bf.diagnostics.calibration_log_gamma` | `log_gamma` |

#### Native Metrics

| Name | Description | Output Keys |
|------|-------------|-------------|
| `sbc_ks` | SBC KS statistic (minimize → 0 = uniform ranks) | `sbc_ks` |
| `sbc_chi2` | SBC chi-squared statistic (minimize → 0 = uniform ranks) | `sbc_chi2` |
| `coverage` | Two-sided SBC rank-based calibration (diagnostic) | `coverage_50`, ..., `coverage_99`, `mean_cal_error` |
| `coverage_left` | Left-sided coverage, efficiency for RCTs (diagnostic) | `left_coverage_50`, ..., `left_mean_cal_error` |
| `coverage_right` | Right-sided coverage, futility for RCTs (diagnostic) | `right_coverage_50`, ..., `right_mean_cal_error` |
| `bias` | Mean signed error of posterior mean (diagnostic) | `bias` |
| `mae` | Mean absolute error of posterior mean | `mae` |
| `correlation` | Pearson association of posterior means and truth; diagnostic only, not recovery error | `correlation` |
| `sbc` | Deprecated; delegates to `sbc_ks` + `sbc_chi2` (diagnostic) | `sbc_ks`, `sbc_chi2` |

Aliases: `cal_error` -> `calibration_error`, `corr` -> `correlation`, `coverage_two_sided` -> `coverage`.

Rows marked **(diagnostic)** are registered `kind="diagnostic"`: they are
computed and reported, but passing one in `objective_metrics` raises. The
authoritative list is `describe_metrics()`; `tests/test_metric_kinds.py`
fails if this table drifts from it.

#### Joint Metrics

These see a whole condition batch — draws, true values, and the data behind
them — rather than one parameter's marginal, so they are registered through
`register_joint_metric()` and passed via `joint_metrics=` rather than
`metrics=`.  They receive a `JointMetricInputs`.

| Name | Kind | Description |
|------|------|-------------|
| `tarp_error` | objective | TARP expected-coverage error against **supplied** reference points. Cannot run at its registered default — build it with `make_tarp_joint_metric(reference_points=...)`. |
| `tarp_error_random` | diagnostic | TARP with reference points drawn here rather than supplied — see `reference=` below for the two distributions. Diagnostic because Lemos et al. (2023, Sec. 4.3) show an `x`-independent reference cannot detect a posterior that ignores its data, so it is rejected in `objective_metrics`. |
| `lc2st` | objective | L-C2ST on the full joint posterior. Requires the `sklearn` extra, and costs roughly 700x a TARP evaluation. |

```python
from bayesflow_hpo import run_validation_pipeline
from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

result = run_validation_pipeline(
    approximator, val_data,
    joint_metrics={"tarp_error": make_tarp_joint_metric(reference_points=refs)},
)
```

Joint metrics are excluded from *intermediate* validation unless
`optimize(include_joint_metrics=True)`; see
[optimization.md](optimization.md#pruning-strategy).

Call `describe_metrics()` for the live registry, with each metric's kind,
aliases, description, and extra dependency.

#### Choosing TARP reference points

Which key a TARP metric emits is decided by whether *you* supply the
reference points, not by how they are distributed:

- **`reference_points=...`** — a callable on `JointMetricInputs`, or one
  array per condition, emitting `tarp_error` (objective). Derive these from
  the conditioning data. Only an `x`-dependent reference detects a posterior
  that ignores its data, and that is the whole reason this key is an
  objective. Deriving them from the posterior under test looks
  data-dependent, is not, and nothing can detect the difference. Passing
  `reference_points=` together with a non-default `reference=` is rejected
  rather than silently resolved, since both name the reference.
- **`reference_points=None`** — drawn for you, emitting `tarp_error_random`
  (diagnostic). `reference=` picks the distribution:

| `reference` | Draw | When |
|-------------|------|------|
| `"uniform_box"` (default) | Uniform over the box spanned by the 1st/99th percentiles of the standardized truths. | The default; keeps existing studies' numbers comparable. |
| `"prior_derangement"` | Each simulation references another simulation's truth, so each reference is marginally a draw from `p(theta)`. Lemos et al. (2023, Sec. 4.1) make this choice, and BayesFlow's `accuracy_random_points` follows it. | A correlated or non-box-shaped prior, where the box puts reference mass in corners no truth or draw occupies. |

Both are `x`-independent, so switching does not turn the diagnostic into an
objective. Sec. 4.2 finds the coverage curve robust across reference
distributions: the choice can move the number without moving the verdict, so
values from the two modes are not interchangeable even though they agree on
whether a posterior is calibrated.

Because the references are a permutation of the truths, they are drawn
without replacement and so are jointly dependent; only the marginal is the
prior. The permutation is redrawn rather than taken as one cyclic shift
precisely to keep that dependence from concentrating in a single offset.

The setting is recorded in the study's joint-metric pin **when it is not the
default** — absence of the key means `"uniform_box"`, which is what pins
written before this option existed meant, so those studies still resume.
Either way, switching distributions changes the pin, so a resume reports it.
Read the pin's `reference` key, not its `reference_mode`: the latter records
only supplied-versus-drawn, so a `prior_derangement` run pins
`reference_mode="random"` while the *result* reports
`reference_mode="prior_derangement"`.

`"prior_derangement"` needs the truths to be distinct: it rejects any
assignment that would hand a simulation a reference equal to its own truth
(which would pin that simulation's coverage fraction at 0), and raises if it
cannot draw a valid one within a bounded number of attempts — the expected
outcome for a prior concentrated on few atoms. It also requires at least two
simulations, since with one the only available reference is that
simulation's own truth.

```python
make_tarp_joint_metric(reference="prior_derangement")
```

### `calibration_error` vs `mean_calibration_error`

Both form the absolute deviation between nominal and empirical
central-interval coverage at 20 nominal levels
(`alpha = linspace(0.005, 0.995, 20)`) and differ only in how those 20
deviations are aggregated:

- `calibration_error` uses BayesFlow's default `np.median`. Despite the
  name it carried until 0.2.x, **it is not an Expected Calibration
  Error** -- an ECE is a mean. Its computation is frozen so that values
  recorded by earlier studies stay comparable.
- `mean_calibration_error` uses `np.mean`. Prefer it for new work: a
  median discards half the calibration curve, so a posterior that behaves
  near the centre can hide badly miscalibrated tails.

Neither is named `ece`, deliberately.
`bf.diagnostics.expected_calibration_error` already exists and is a
*different* statistic -- a bin-size-weighted calibration error over one-hot
model indices, for model comparison, after Naeini et al. (2015). The
Expected Calibration Error of that literature is a weighted mean over bins
of predicted probability, not an unweighted mean over equally spaced
nominal coverage levels, so this package does not claim the term.

Both are *marginal* statistics computed per parameter, ignoring the data
behind each posterior, so a posterior that returns the prior regardless
of its input scores perfectly on either. A good value is necessary, not
sufficient.

Default set: `DEFAULT_METRICS = ["calibration_error", "nrmse", "correlation", "coverage", "rmse", "contraction"]`

`correlation` is retained in validation reports for exploratory recovery plots, but is registered as diagnostic-only and cannot be used in `objective_metrics`. Pearson correlation measures linear association rather than agreement: additive or multiplicative bias can leave it equal to 1. Its posterior-mean summary is retained to match RMSE/NRMSE and their squared-error loss. Use NRMSE as the point-recovery objective and inspect bias and recovery plots alongside it. A posterior median is appropriate when the intended loss is absolute error, in which case a median-based MAE should be used instead of silently changing correlation (Gneiting, 2011).

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
2. **Inference** — `make_bayesflow_infer_fn` wraps the approximator to produce posterior draws,
   sampling the condition batch in slices of at most `max_samples_per_call` draws
   (default `20_000`). A condition holds `sims_per_condition x n_posterior_samples`
   draws — 100,000 at the `optimize()` defaults. Neither factor is a
   search-space hyperparameter, so the training estimate cannot see them;
   `estimate_validation_memory_mb()` budgets this chunk separately before
   training (see [optimization.md](optimization.md#memory-budget)). Pass
   `max_samples_per_call=None` to sample each condition in one call. Note that
   the cap also moves `inference_time`: several smaller `sample()` calls carry
   more fixed per-call cost than one large one, so cost values are not
   comparable across a change to this setting — relevant when warm-starting or
   resuming a study, where old and new trials share one Pareto front.
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
    failed_joint_metrics: dict[str, str]  # joint metric name -> error message
    joint_metric_settings: dict[str, dict[str, Any]]  # configuration actually used
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

## Custom Validation for Structured Posteriors

The default `run_validation_pipeline` expects flat 2D posteriors `(batch, param_dim)`. For models with structured (e.g., per-item) posteriors of shape `(batch, n_samples, items)`, the default pipeline will fail because `bf.diagnostics.calibration_error` cannot broadcast the shapes.

**Solution**: provide a custom `validate_fn` to `optimize()` that flattens the structured posteriors before computing metrics:

```python
from bayesflow_hpo.validation.registry import resolve_metrics


def validate_irt(approximator, validation_data, n_posterior_samples):
    metric_fns = resolve_metrics(["calibration_error", "correlation"])
    all_rows = []

    for raw_batch in validation_batches:
        posterior = approximator.sample(conditions=raw_batch, num_samples=n_posterior_samples)

        for param_key in ["a", "b"]:
            true = np.asarray(raw_batch[param_key])            # (B, I)
            draws = np.asarray(posterior[param_key])            # (B, n_samples, I)

            # Flatten per-item: (B*I,) and (B*I, n_samples)
            true_flat = true.reshape(-1)
            draws_flat = np.moveaxis(draws, 1, -1).reshape(-1, draws.shape[1])

            row = {}
            for _name, fn in metric_fns.items():
                row.update(fn(draws_flat, true_flat))
            all_rows.append(row)

    # Average across items and parameters
    return {key: float(np.mean([r[key] for r in all_rows])) for key in ["calibration_error", "correlation"]}
```

This `validate_fn` is also used for intermediate validation during training (via `PeriodicValidationCallback`), enabling mid-training pruning for structured approximators.

## C2ST Metrics

Classifier two-sample tests for posterior validation. Requires `scikit-learn>=1.3` (`pip install bayesflow-hpo[sklearn]`).

### Local C2ST (L-C2ST)

Reference-free local posterior diagnostic using joint `(theta, x)` samples (Linhart et al., 2023). No reference posterior needed — uses the amortized approximator's own samples.

```python
from bayesflow_hpo import lc2st

result = lc2st(
    posterior_samples,   # (n_sims, n_samples, n_params)
    true_params,         # (n_sims, n_params)
    observations,        # (n_sims, ...) observation data
    n_folds=5,           # cross-validation folds
    n_null_trials=0,     # permutation null trials (0 = skip)
    seed=42,
)
# result.statistic: float              (mean single-class MSE_0; near 0 = calibrated)
# result.p_value: float | None         (None unless n_null_trials > 0)
# result.null_statistics: np.ndarray   (empty unless n_null_trials > 0)
# result.per_observation_stats: np.ndarray
```

### Global C2ST

Standard classifier two-sample test (Lopez-Paz & Oquab, 2017). Requires reference posterior samples.

```python
from bayesflow_hpo import global_c2st

result = global_c2st(
    samples_p,  # (n, d) reference posterior samples
    samples_q,  # (n, d) approximate posterior samples
    seed=42,
)
# result.accuracy: float  (0.5 = indistinguishable, 1.0 = fully separable)
# result.p_value: float
# result.n_test: int
```

### ValidateFn Factory

`make_lc2st_validate_fn()` returns a `ValidateFn` compatible with `optimize(validate_fn=...)` that computes standard per-parameter metrics and L-C2ST from a single inference pass:

```python
from bayesflow_hpo import make_lc2st_validate_fn

validate_fn = make_lc2st_validate_fn(
    base_metrics=["calibration_error", "nrmse"],
    n_folds=5,
)

study = hpo.optimize(
    ...,
    validate_fn=validate_fn,
    objective_metrics=["calibration_error", "nrmse"],
)
```

## Possible Future Extensions

- **Prior-scale NRMSE:** evaluate replacing validation-sample range normalization with a fixed prior-scale normalization. This could make scores less sensitive to the realized validation sample, but requires a compatibility and aggregation design.
- **Held-out posterior NLL for density-evaluable NPE:** average `-log q(theta | x)` over a large prior-predictive validation set. Lueckmann et al. (2021) describe this as appropriate when evaluated across many observations; it should not be inferred from a handful of cases.

These are prospective features, not currently available metrics.  TARP, which
this list used to name among them, is implemented — see
[Joint Metrics](#joint-metrics).

## SBC Tests

### Rank Uniformity

If the posterior is well-calibrated, SBC ranks should be uniform over `[0, n_posterior_samples]`.

```python
from bayesflow_hpo.validation.sbc_tests import compute_sbc_uniformity_tests

results = compute_sbc_uniformity_tests(ranks, n_posterior_samples, n_bins=20)
# {"ks_statistic": ..., "ks_pvalue": ..., "chi2_statistic": ..., "chi2_pvalue": ...}
```

- **KS test** — Kolmogorov-Smirnov test against `Uniform(0, n_posterior_samples)`
- **Chi-squared test** — Binned chi-squared test of rank histogram

Both return p-values; low p-values indicate miscalibration.

