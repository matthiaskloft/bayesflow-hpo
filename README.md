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
                                 3. Build approximator
                                 4. Compile with Adam + CosineDecay
                                 5. Train
                                 6. Validate on fixed dataset
                                 7. Report metrics to Optuna
```

## Features

- **10 network search spaces** — CouplingFlow, FlowMatching, DiffusionModel, ConsistencyModel, StableConsistencyModel, DeepSet, SetTransformer, FusionTransformer, TimeSeriesNetwork, TimeSeriesTransformer
- **Network selection** — let Optuna choose the best architecture via `NetworkSelectionSpace` / `SummarySelectionSpace`
- **Custom hooks** — replace build, train, or validate steps while reusing the full trial lifecycle
- **Pre-flight validation** — `check_pipeline()` catches interface errors before GPU hours are wasted
- **Multi-objective** — single-metric, mean-aggregated, or full Pareto-front optimization
- **13 built-in validation metrics** — calibration, accuracy, SBC diagnostics, plus a registry for custom metrics
- **Budget constraints** — reject infeasible architectures before training (`max_param_count`, `max_memory_mb`)
- **Custom network registration** — plug in your own inference/summary networks
- **Study management** — resume, warm-start, save/load workflows, analyze results

## Examples

See the [examples/](examples/) directory for complete walkthroughs:

| Notebook | Description |
|---|---|
| [`quickstart.ipynb`](examples/quickstart.ipynb) | End-to-end HPO: search, train, validate, select & retrain the best model |
| [`custom_summary_network.ipynb`](examples/custom_summary_network.ipynb) | Custom summary network registration + HPO |
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
