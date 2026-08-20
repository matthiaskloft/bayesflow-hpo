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
                                 4. Compile with the mode's LR schedule
                                 5. Train
                                 6. Validate on fixed dataset
                                 7. Report metrics to Optuna
```

## Features

- **10 network search spaces** — CouplingFlow, FlowMatching, DiffusionModel, ConsistencyModel, StableConsistencyModel, DeepSet, SetTransformer, FusionTransformer, TimeSeriesNetwork, TimeSeriesTransformer
- **FlowMatching solver controls** — tune or fix `fm_integrate_method` / `fm_integrate_steps` plus TimeMLP kwargs (`fm_merge`, `fm_norm`, `fm_residual`, `fm_spectral_normalization`) with BayesFlow-default constants when untuned
- **Network selection** — let Optuna choose the best architecture via `NetworkSelectionSpace` / `SummarySelectionSpace`
- **Sampler presets** — TPE, GP, BoTorch, NSGA-II/III, Auto, Random with auto-wired budget constraints
- **QMC warm-up** — Sobol sequences for better space-filling startup coverage
- **Multi-objective pruning** — dominance, MO-SHA, primary-median strategies
- **Metric constraints** — hard post-validation rejection, soft feasibility guidance
- **Coherent training modes** — equal-budget cosine runs, plus open-ended inverse-sqrt runs with validation early stopping
- **Derived hyperparameters** — express exact simulation-budget relationships without redundant sampler dimensions
- **Custom hooks** — replace build, train, or validate steps while reusing the full trial lifecycle
- **Pre-flight validation** — `check_pipeline()` catches interface errors before GPU hours are wasted
- **Multi-objective** — single-metric, mean-aggregated, or full Pareto-front optimization (2-3 objectives)
- **Rich visualization** — `plot_study()` adaptive dashboard with pairwise Pareto projections, per-objective history, and parameter importance; plus standalone `plot_pareto_3d()`, `plot_parallel_coordinates()`, and more
- **13 built-in validation metrics** — calibration, accuracy, SBC diagnostics, C2ST, plus a registry for custom metrics
- **Budget constraints** — reject infeasible architectures before training (`max_param_count`, `max_memory_mb="auto"`)
- **Custom network registration** — plug in your own inference/summary networks
- **Study management** — resume, warm-start, save/load workflows, analyze results
- **Lexicographic-Pareto selection** — `select_best_trial()` with satisficing thresholds

## Examples

See the [examples/](examples/) directory for complete walkthroughs:

```python
# Speed-sensitive FlowMatching search space
import bayesflow_hpo as hpo

inference_space = hpo.FlowMatchingSpace.fast()
search_space = hpo.CompositeSearchSpace(inference_space=inference_space)
```

| Notebook | Description |
|---|---|
| [`getting_started.ipynb`](examples/getting_started.ipynb) | End-to-end HPO: search, train, validate, select & retrain the best model |
| [`two_moons_optimization.ipynb`](examples/two_moons_optimization.ipynb) | Network selection (CouplingFlow vs FlowMatching) on the Two Moons benchmark |
| [`custom_summary_network.ipynb`](examples/custom_summary_network.ipynb) | Custom summary network registration + HPO |
| [`qmc_warmup_benchmark.ipynb`](examples/qmc_warmup_benchmark.ipynb) | QMC warm-up effectiveness benchmark (TPE vs GP, convergence analysis) |
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
