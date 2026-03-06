# Changelog

Changes implemented during the v0.2.0 workover (quality fixes, BayesFlow decoupling, validation redesign).

---

## Bug Fixes

### `decay_steps` used `batch_size` instead of `batches_per_epoch`

**File:** `builders/workflow.py`

The LR schedule decayed every `batch_size` optimizer steps — far too aggressive. Fixed by adding `batches_per_epoch: int = 50` to `WorkflowBuildConfig` and using `decay_steps = max(1, config.batches_per_epoch)`.

### `estimate_param_count` included learning rate

**File:** `optimization/constraints.py`

Removed the `regularization = int(1000 * lr)` term that incorrectly added learning rate to the parameter count estimate.

### Silent exception swallowing in `GenericObjective`

**File:** `optimization/objective.py`

Training exceptions were caught with a bare `except Exception` and no logging. Now:
- Exceptions are logged via `logging.warning`
- Error message stored as `trial.set_user_attr("training_error", str(exc))`
- Separate `training_failure_penalty` field on `ObjectiveConfig`

### `assert` used for runtime validation

**File:** `builders/adapter.py`

Replaced `assert loc is not None` with `raise ValueError(...)`. (Module was subsequently removed — see decoupling below.)

### `loguniform_int/float` accepted non-positive bounds

**File:** `utils.py`

Added `if low <= 0: raise ValueError(...)` at the top of both functions.

---

## BayesFlow Decoupling

### Removed `PriorStandardize` (deep coupling to BF internals)

**File:** `builders/adapter.py`

Subclassed the non-public `bayesflow.adapters.transforms.transform.Transform` and used the internal `@serializable` decorator. Deleted entirely. Users should use `bf.Adapter` fluent API.

### Removed `AdapterSpec` + `create_adapter`

**File:** `builders/adapter.py`

Reimplemented BayesFlow's Adapter builder behind a declarative dataclass. Deleted entirely. Users provide their own `bf.Adapter`.

### Simplified `build_workflow`

**File:** `builders/workflow.py`

Removed manual `compile()` call. Now delegates to `bf.BasicWorkflow(optimizer=..., initial_learning_rate=...)`. Added `optimizer: Any | None = None` to `WorkflowBuildConfig` so users can provide a custom optimizer; when `None`, an `ExponentialDecay + Adam` schedule is created with the corrected `decay_steps`.

### Added `train_fn` hook to `ObjectiveConfig`

**File:** `optimization/objective.py`

The objective previously hardcoded `workflow.fit_online(...)`. Added optional `train_fn: Callable[[BasicWorkflow, dict, list], None]` field. When `None`, falls back to the default `fit_online` call. Users with pre-simulated data can plug in `fit_offline` or `fit_disk`.

### Removed duplicated GPU cleanup

**File:** `validation/pipeline.py`

Replaced `_cleanup_gpu_memory()` with `from bayesflow_hpo.optimization.cleanup import cleanup_trial`.

### Fixed `Sequence` import

**File:** `validation/pipeline.py`

Changed `from typing import Sequence` to `from collections.abc import Sequence`.

### Fixed tabs in `search_spaces/summary/__init__.py`

Converted tabs to 4-space indentation.

### Removed single-choice categoricals

- `FlowMatchingSpace.fm_loss` (only `"mse"`) — removed dimension, hardcoded in `build()`
- `DeepSetSpace.ds_pooling` (only `"mean"`) — removed dimension, hardcoded `inner_pooling="mean"`, `output_pooling="mean"`

### Cleaned up `__init__.py` exports

Removed `PriorStandardize`, `AdapterSpec`, `create_adapter`. Added `extract_objective_values`, `ValidationResult`, `validate_once`, `make_condition_grid`, `make_coverage_metric`, `make_validation_dataset`, `register_metric`, `list_metrics`, `DEFAULT_METRICS`. Sorted `__all__` alphabetically.

---

## Validation Module Redesign

### New: `validation/registry.py` — Metric registry

Plugin-based metric registry mapping string names to callables. Metric function signature: `(draws: ndarray[n_sims, n_samples], true_values: ndarray[n_sims]) -> dict[str, float]`.

**Built-in metrics wrapping BF diagnostics:**

| Name | Wraps | Output Keys |
|------|-------|-------------|
| `calibration_error` | `bf.diagnostics.calibration_error` | `calibration_error` |
| `rmse` | `bf.diagnostics.root_mean_squared_error` | `rmse` |
| `nrmse` | `bf.diagnostics.root_mean_squared_error(normalize="range")` | `nrmse` |
| `contraction` | `bf.diagnostics.posterior_contraction` | `contraction` |
| `z_score` | `bf.diagnostics.posterior_z_score` | `mean_abs_z_score`, `mean_z_score` |
| `log_gamma` | `bf.diagnostics.calibration_log_gamma` | `log_gamma` |

**Native metrics (not in BayesFlow):**

| Name | Description | Output Keys |
|------|-------------|-------------|
| `sbc` | KS + chi-squared + C2ST on SBC ranks | `sbc_ks_pvalue`, `sbc_chi2_pvalue`, `sbc_c2st_accuracy` |
| `coverage` | Two-sided SBC rank-based calibration | `coverage_50`, ..., `coverage_99`, `mean_cal_error` |
| `coverage_left` | Left-sided coverage (efficiency) | `left_coverage_50`, ..., `left_mean_cal_error` |
| `coverage_right` | Right-sided coverage (futility) | `right_coverage_50`, ..., `right_mean_cal_error` |
| `bias` | Mean signed error of posterior mean | `bias` |
| `mae` | Mean absolute error of posterior mean | `mae` |

**Coverage metric factory:**
```python
make_coverage_metric(
    levels=[0.5, 0.8, 0.9, 0.95, 0.99],
    side="two-sided",  # or "left", "right"
    weights=None,
    prefix="",
) -> MetricFn
```

**Public API:**
```python
register_metric(name, fn, aliases=None, overwrite=False)
get_metric(name) -> MetricFn
resolve_metrics(names) -> dict[str, MetricFn]
list_metrics() -> list[str]
```

### New: `validation/result.py` — Structured results

```python
@dataclass(frozen=True)
class ValidationResult:
    condition_metrics: pd.DataFrame
    summary: dict[str, float]
    per_parameter: dict[str, ValidationResult] | None = None
    timing: dict[str, float]
    n_conditions: int
    n_posterior_samples: int
    metric_names: list[str]
```

Table methods: `summary_table()`, `condition_table(metric=None)`, `parameter_table()`.
HPO extraction: `objective_scalar(key="calibration_error")`.

### New: `validation/dry_run.py` — Quick compatibility check

```python
validate_once(approximator, validation_data, n_sims=2, n_posterior_samples=10, metrics=None)
```

Slices the first condition, runs the full pipeline, wraps errors with descriptive messages including `param_keys`/`data_keys`.

### New: Grid helpers in `validation/data.py`

```python
make_condition_grid(linspace=..., logspace=..., values=...)
make_validation_dataset(simulator, param_keys, data_keys, linspace=..., logspace=..., values=..., sims_per_condition=200, seed=42)
```

### Rewritten: `validation/metrics.py`

Replaced the 215-line module with two functions:
- `compute_condition_metrics(draws, true_values, cond_id, metric_fns)` — dispatches to registry
- `aggregate_condition_rows(condition_rows)` — averages numeric values across conditions

### Rewritten: `validation/pipeline.py`

- Accepts `metrics: list[str]` parameter (resolved via registry)
- Returns `ValidationResult` instead of raw `dict`
- Supports multi-parameter inference (per-parameter sub-results)
- Uses `cleanup_trial` from `optimization.cleanup`

### Integrated into objective

- `ObjectiveConfig` gained `metrics: list[str] | None`, `objective_metric: str` fields
- `api.optimize()` gained `metrics`, `objective_metric`, `train_fn` parameters
- Objective extracts scalar via `validation_result.objective_scalar(self.config.objective_metric)`

---

## Test Improvements

- Extracted shared `FakeTrial` and `DummySimulator` to `tests/conftest.py`
- Updated all existing tests for new APIs
- Added `tests/test_validation/test_metric_registry.py` (16 tests: registry CRUD, all built-in metrics, coverage variants)
- Added `tests/test_validation/test_result.py` (8 tests: table methods, objective extraction, repr)
- Total: 91 tests passing
