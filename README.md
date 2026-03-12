# bayesflow-hpo

[![CI](https://github.com/matthiaskloft/bayesflow-hpo/actions/workflows/ci.yml/badge.svg)](https://github.com/matthiaskloft/bayesflow-hpo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/matthiaskloft/bayesflow-hpo/branch/main/graph/badge.svg)](https://codecov.io/gh/matthiaskloft/bayesflow-hpo)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated hyperparameter optimization for [BayesFlow](https://github.com/bayesflow-org/bayesflow) 2.x neural posterior estimation (NPE) models, powered by [Optuna](https://optuna.org/).

Tunes inference networks, summary networks, and training hyperparameters — so you can focus on the model, not the plumbing.

## Installation

```bash
pip install bayesflow-hpo            # core
pip install bayesflow-hpo[c2st]      # + C2ST metric (requires scikit-learn)
pip install bayesflow-hpo[dashboard] # + Optuna dashboard
```

Requires Python >= 3.11 and a Keras 3 backend (PyTorch recommended: `export KERAS_BACKEND=torch`).

## Quick start

```python
import bayesflow_hpo as hpo

# One call does it all: search space, study, training, validation
study = hpo.optimize(
    simulator=simulator,          # your BayesFlow simulator
    adapter=adapter,              # your BayesFlow adapter
    validation_conditions={"N": [50, 100, 200]},
    n_trials=50,
)
```

`optimize()` automatically infers `param_keys` and `data_keys` from adapter transforms, so in most cases you don't need to pass them explicitly.

## How it works

```
                              ┌──────────────────────────┐
                              │   CompositeSearchSpace   │
                              │  ┌────────────────────┐  │
 Your simulator + adapter ──► │  │ Inference networks │  │
                              │  │ Summary networks   │  │ ──► Optuna Study
 Validation conditions ─────► │  │ Training params    │  │      (n_trials)
                              │  └────────────────────┘  │
                              └──────────────────────────┘
                                          │
                                 Each trial:
                                 1. Sample hyperparameters
                                 2. Budget check (reject if too large)
                                 3. Build & train workflow
                                 4. Validate on fixed dataset
                                 5. Report metrics to Optuna
```

## Features

### Search spaces

Declarative, dataclass-based search spaces for **10 network architectures**:

| Inference networks | Summary networks |
|---|---|
| CouplingFlow | DeepSet |
| FlowMatching | SetTransformer |
| DiffusionModel | FusionTransformer |
| ConsistencyModel | TimeSeriesNetwork |
| StableConsistencyModel | TimeSeriesTransformer |

Use `NetworkSelectionSpace` to let Optuna choose the best network type, or fix a single architecture:

```python
from bayesflow_hpo import CouplingFlowSpace, CompositeSearchSpace, TrainingSpace

space = CompositeSearchSpace(inference_space=CouplingFlowSpace(), training_space=TrainingSpace())
study = hpo.optimize(..., search_space=space)
```

### Multi-objective optimization

Single-metric (default), mean-aggregated, or full Pareto-front optimization:

```python
# Mean of multiple metrics
study = hpo.optimize(..., objective_metrics=["calibration_error", "nrmse"], objective_mode="mean")

# Full Pareto front: each metric becomes its own Optuna objective
study = hpo.optimize(..., objective_metrics=["calibration_error", "nrmse"], objective_mode="pareto")
```

### Validation metrics

14 built-in metrics with a registry for custom ones:

| Category | Metrics |
|---|---|
| Calibration | `calibration_error`, `mean_cal_error`, `coverage`, `coverage_left`, `coverage_right` |
| Accuracy | `rmse`, `nrmse`, `mae`, `bias`, `correlation` |
| Diagnostics | `contraction`, `z_score`, `log_gamma` |
| SBC | `sbc` (KS test, chi-squared, C2ST) |

```python
# Register a custom metric
hpo.register_metric("my_metric", my_metric_fn)
study = hpo.optimize(..., metrics=["calibration_error", "my_metric"])
```

### Budget constraints

Reject infeasible architectures *before* training. Budget-rejected trials don't count toward `n_trials`.

```python
study = hpo.optimize(..., max_param_count=500_000, max_memory_mb=4096)
```

### Custom networks

```python
from bayesflow_hpo import register_custom_inference_network

register_custom_inference_network(
    name="my_flow", network_cls=MyCustomFlow, space_cls=MyCustomFlowSpace, builder_fn=my_builder,
)
```

### Study management & results

```python
# Resume / warm-start
study = hpo.optimize(..., study_name="my_study", storage="sqlite:///hpo.db", resume=True)
study = hpo.optimize(..., warm_start_from=previous_study, warm_start_top_k=10)

# Save / load the best workflow
hpo.save_workflow_with_metadata(workflow, study, path="best_model/")
workflow = hpo.load_workflow_with_metadata("best_model/")

# Analyze
df = hpo.trials_to_dataframe(study)
hpo.summarize_study(study)
hpo.plot_pareto_front(study)
hpo.plot_param_importance(study)
```

## Examples

| Notebook | Description |
|---|---|
| [`quickstart.ipynb`](examples/quickstart.ipynb) | Minimal end-to-end HPO run |
| [`custom_summary_network.ipynb`](examples/custom_summary_network.ipynb) | Custom summary network registration + HPO |
| [`multi_objective.ipynb`](examples/multi_objective.ipynb) | Two-objective optimization + warm start |
| [`optuna_dashboard.md`](examples/optuna_dashboard.md) | Optuna dashboard integration guide |

## Contributing

```bash
git clone git@github.com:matthiaskloft/bayesflow-hpo.git
cd bayesflow-hpo
export KERAS_BACKEND=torch
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## License

MIT
