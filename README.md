# bayesflow-hpo

[![CI](https://github.com/matthiaskloft/bayesflow-hpo/actions/workflows/ci.yml/badge.svg)](https://github.com/matthiaskloft/bayesflow-hpo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/matthiaskloft/bayesflow-hpo/branch/main/graph/badge.svg)](https://codecov.io/gh/matthiaskloft/bayesflow-hpo)

Generic hyperparameter optimization for BayesFlow neural posterior estimation (NPE).

## Features

- **Declarative search spaces** for 5 inference networks (CouplingFlow, FlowMatching, Diffusion, Consistency, StableConsistency) and 5 summary networks (DeepSet, SetTransformer, FusionTransformer, TimeSeriesNetwork, TimeSeriesTransformer)
- **Multi-metric objectives** — single metric, arithmetic mean, or full Pareto-front mode (`objective_mode="mean"` | `"pareto"`)
- **13+ built-in validation metrics** — calibration error, NRMSE, correlation, coverage, SBC, bias, MAE, and more, with a `register_metric()` API for custom metrics
- **Fixed validation datasets** — generate, save, and reload condition-grid validation data for reproducible trial comparison
- **Parameter & memory budgets** — `max_param_count` and `max_memory_mb` pre-checks to reject infeasible trials before training
- **Intermediate pruning** — periodic validation during training for early stopping of unpromising trials
- **Custom network registration** — plug in user-defined inference/summary networks with their own search spaces
- **Warm-start study seeding** from prior Optuna studies
- **Workflow persistence** — save/load trained models with full study metadata
- **Pareto-front extraction and plotting helpers**

## Examples

- `examples/quickstart.ipynb` — minimal end-to-end BayesFlow HPO run
- `examples/custom_summary_network.ipynb` — custom summary-space registration + HPO
- `examples/multi_objective.ipynb` — two-objective optimization + warm start
- `examples/optuna_dashboard.md` — Optuna dashboard integration guide

## Quick start

```python
import bayesflow_hpo as hpo

study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    param_keys=["theta"],
    data_keys=["x"],
    validation_conditions={"N": [50, 100, 200]},
    n_trials=50,
)
```
