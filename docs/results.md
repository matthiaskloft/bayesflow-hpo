# Results & Export

## Pareto Front Extraction

After optimization, extract the Pareto-optimal trials (non-dominated in both objectives):

```python
from bayesflow_hpo import get_pareto_trials

pareto = get_pareto_trials(study)
for trial in pareto:
    cal_error, param_score = trial.values
    print(f"Trial {trial.number}: cal={cal_error:.4f}, params={param_score:.3f}")
```

## Trials DataFrame

Convert all trials to a pandas DataFrame for analysis:

```python
from bayesflow_hpo import trials_to_dataframe

df = trials_to_dataframe(study, include_pruned=False)
# Columns: trial_number, state, value_0, value_1, param_*, user_attr_*
```

## Visualization

### Pareto Front Plot

```python
from bayesflow_hpo import plot_pareto_front

fig, ax = plot_pareto_front(study)
# Scatter: calibration error (x) vs. parameter score (y)
# Pareto-optimal points highlighted in red
```

### Parameter Importance

```python
from bayesflow_hpo import plot_param_importance

fig, ax = plot_param_importance(study, top_k=10)
# Horizontal bar chart of top-k hyperparameter importances
```

Uses Optuna's built-in `get_param_importances()` with fANOVA.

## Model Export

### Save with Metadata

After selecting a trial, retrain and export the model with full reproducibility metadata:

```python
from bayesflow_hpo import save_workflow_with_metadata, get_workflow_metadata

metadata = get_workflow_metadata(
    config=best_trial.params,
    model_type="coupling_flow",
    validation_results=validation_result,
    extra={"study_name": "my_hpo", "trial_number": best_trial.number},
)

path = save_workflow_with_metadata(
    approximator=workflow.approximator,
    path="best_model/",
    metadata=metadata,
)
# Creates: best_model/model.keras + best_model/metadata.json
```

### Load

```python
from bayesflow_hpo import load_workflow_with_metadata

approximator, metadata = load_workflow_with_metadata("best_model/")
```

### Metadata Contents

The metadata JSON includes:
- `config` — All hyperparameters
- `model_type` — Network type name
- `saved_at` — ISO timestamp
- `bayesflow_hpo_version` — Package version
- `validation_results` — Full validation metrics (if provided)
- Any `extra` fields you pass

## Parameter Count Utilities

```python
from bayesflow_hpo import get_param_count, normalize_param_count, denormalize_param_count

count = get_param_count(workflow.approximator)        # e.g. 150_000
score = normalize_param_count(count)                  # log10(150000) / 6.0 ≈ 0.86
count_back = denormalize_param_count(score)           # ≈ 150_000
```
