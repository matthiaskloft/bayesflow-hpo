# bayesflow-hpo Documentation

Generic hyperparameter optimization for [BayesFlow 2.x](https://github.com/bayesflow-org/bayesflow) neural posterior estimation.

## Overview

`bayesflow-hpo` automates the search for optimal neural network architectures and training hyperparameters for amortized Bayesian inference models built with BayesFlow. It uses [Optuna](https://optuna.org/) for multi-objective optimization, balancing **calibration quality** against **model complexity**.

### Key Capabilities

- **Declarative search spaces** for all BayesFlow inference and summary network types
- **Multi-objective Optuna integration** (configurable quality metric vs. normalized parameter count)
- **Fixed validation datasets** with condition grid helpers for fair comparison across architectures
- **Metric registry** with built-in BF diagnostic wrappers and native SBC/coverage/bias metrics
- **SBC rank-based coverage** with two-sided, left-sided (efficiency), and right-sided (futility) variants
- **Custom metrics** via a plugin registry (`register_metric`)
- **Structured validation results** with per-condition, per-parameter, and summary tables
- **Dry-run validation** to catch shape mismatches before a full HPO run
- **Memory/parameter budget** pre-checks to avoid OOM trials
- **Configurable training** (default `fit_online`, or user-provided `train_fn` for `fit_offline`/`fit_disk`)
- **Warm-start** from prior Optuna studies
- **Custom network registration** for user-defined architectures
- **Pareto front extraction** and importance plotting

## Quick Start

```python
import bayesflow as bf
import bayesflow_hpo as hpo

# Your simulator + adapter (BayesFlow standard setup)
simulator = bf.simulators.make_simulator(...)
adapter = bf.adapters.Adapter(...)

# Run HPO with sensible defaults (CouplingFlow + DeepSet)
study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    param_keys=["theta"],
    data_keys=["x"],
    validation_conditions={"N": [50, 100, 200]},
    n_trials=50,
    epochs=100,
    metrics=["calibration_error", "coverage", "rmse"],
    objective_metric="calibration_error",
)

# Analyze results
pareto = hpo.get_pareto_trials(study)
hpo.plot_pareto_front(study)
```

## Documentation Contents

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Package structure, module responsibilities, data flow |
| [Search Spaces](search_spaces.md) | All network search spaces, dimensions, and customization |
| [Optimization](optimization.md) | Objective function, constraints, study management, callbacks |
| [Validation](validation.md) | Validation datasets, metric registry, coverage, SBC tests, result tables |
| [Results & Export](results.md) | Pareto extraction, visualization, model export |
| [API Reference](api_reference.md) | Complete public API with signatures and descriptions |
| [Changelog](quality_report.md) | Changes implemented in the v0.2.0 workover |

## Installation

```bash
pip install bayesflow-hpo

# With optional dependencies
pip install bayesflow-hpo[c2st]       # SBC classifier two-sample test
pip install bayesflow-hpo[dashboard]  # Optuna dashboard
pip install bayesflow-hpo[dev]        # Development tools
```

## Requirements

- Python >= 3.11
- BayesFlow >= 2.0.0
- Optuna >= 3.0.0
- Keras >= 3.9, < 3.13 (PyTorch backend recommended)
