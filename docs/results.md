# Results & Export

## Pareto Front Extraction

After optimization, extract the Pareto-optimal trials (non-dominated in all objectives):

```python
from bayesflow_hpo import get_pareto_trials

pareto = get_pareto_trials(study)
for trial in pareto:
    print(f"Trial {trial.number}: values={trial.values}")
```

## Trial Selection

### Lexicographic-Pareto Selection

Select the best trial using a two-phase algorithm (Deb et al., 2002):
1. **Satisficing phase** — filter candidates by priority thresholds in order; unmet priorities promote to Phase 2
2. **Pareto phase** — apply non-dominated sorting over remaining study objectives with mean-rank tiebreak

```python
from bayesflow_hpo import select_best_trial

trial, result = select_best_trial(
    study,
    priorities=[
        ("calibration_error", 0.01),   # infer direction from study
        ("nrmse", 0.05),               # infer direction from study
    ],
    # inference_time (3rd objective) has no threshold → Pareto selection
)

print(f"Selected trial #{trial.number}")
print(f"Thresholds met: {result.thresholds_met}")
print(f"Pareto front size: {len(result.pareto_front)}")
```

For user attributes (non-objective metrics), specify direction explicitly:

```python
trial, result = select_best_trial(
    study,
    priorities=[
        ("calibration_error", 0.01),
        ("coverage_90", 0.85, "above"),  # user_attr, explicit direction
    ],
)
```

### Convenience Functions

```python
from bayesflow_hpo import best_config, trial_table, compare_trials

# Best trial's hyperparameters (with optional priority-based selection)
config = best_config(
    study,
    priorities=[("calibration_error", 0.01), ("nrmse", 0.05)],
)

# Formatted trial table (top-k by select_by objective)
table = trial_table(study, top_k=10, select_by=0, metrics=["calibration_error", "nrmse"])

# Side-by-side comparison of specific trials
comparison = compare_trials(study, trial_numbers=[3, 7, 12])
```

## Trials DataFrame

Convert all trials to a pandas DataFrame for analysis:

```python
from bayesflow_hpo import trials_to_dataframe

df = trials_to_dataframe(study, include_pruned=False, include_ranks=True)
# Columns: trial_number, state, value_0, value_1, param_*, user_attr_*, rank_*
```

## Study Summary

```python
from bayesflow_hpo import summarize_study

print(summarize_study(study, select_by=0))
```

## Visualization

### Study Dashboard

`plot_study()` creates a 3-row GridSpec figure with Pareto front, optimization history, and parameter importance. Supports 2--3 objectives.

```python
from bayesflow_hpo import plot_study

fig = plot_study(study, third_dim="color")
```

### Individual Plots

```python
from bayesflow_hpo import (
    plot_pareto_front,
    plot_optimization_history,
    plot_param_importance,
    plot_metric_scatter,
    plot_metric_panels,
    plot_pareto_3d,
    plot_pareto_projections,
    plot_parallel_coordinates,
)

# Pairwise 2D Pareto projections with third-dim encoding
plot_pareto_front(study, third_dim="color", max_cols=3)

# Per-objective direction-aware step lines
plot_optimization_history(study, max_cols=3)

# Per-objective parameter importance bar charts
plot_param_importance(study, top_k=10, max_cols=3)

# Scatter of any two metrics
plot_metric_scatter(study, "calibration_error", "nrmse", show_iso_lines=True)

# Per-trial metric distribution panels
plot_metric_panels(study, metrics=["calibration_error", "nrmse"], max_cols=3)

# 3D Pareto front (3-objective studies)
plot_pareto_3d(study, cost_display="color")

# Pairwise 2D projections (3-objective studies)
plot_pareto_projections(study, cost_display="color", max_cols=3)

# Parallel coordinates for top-k trials
plot_parallel_coordinates(study, top_k=20, select_by=0)
```

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
