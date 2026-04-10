# bayesflow-hpo Documentation

Generic hyperparameter optimization for [BayesFlow 2.x](https://github.com/bayesflow-org/bayesflow) neural posterior estimation.

## Overview

`bayesflow-hpo` automates the search for optimal neural network architectures and training hyperparameters for amortized Bayesian inference models built with BayesFlow. It uses [Optuna](https://optuna.org/) for multi-objective optimization, balancing **calibration quality** against **model complexity**.

### Key Capabilities

- **Declarative search spaces** for all BayesFlow inference and summary network types
- **Multi-objective Optuna integration** (configurable quality metrics vs. cost)
- **Sampler presets** (TPE, GP, BoTorch, NSGA-II/III, Auto, Random) with auto-wired constraints
- **QMC warm-up** using Sobol sequences for better space-filling startup coverage
- **Fixed validation datasets** with condition grid helpers for fair comparison across architectures
- **Metric registry** with built-in BF diagnostic wrappers and native SBC/coverage/C2ST metrics
- **SBC rank-based coverage** with two-sided, left-sided (efficiency), and right-sided (futility) variants
- **Metric constraints** (hard post-validation rejection, soft feasibility guidance for samplers)
- **Custom metrics** via a plugin registry (`register_metric`)
- **Structured validation results** with per-condition, per-parameter, and summary tables
- **Dry-run validation** to catch shape mismatches before a full HPO run
- **Memory/parameter budget** pre-checks with GPU-memory auto-detection
- **Multi-objective pruning** (dominance, MO-SHA, primary-metric median)
- **Configurable training** (default `approximator.fit(simulator=...)`, or user-provided `train_fn`)
- **Warm-start** from prior Optuna studies
- **Custom network registration** for user-defined architectures
- **Pareto front extraction** and lexicographic-Pareto trial selection
- **Rich visualization** (Pareto fronts, optimization history, parameter importance)

## Quick Start

```python
import bayesflow as bf
import bayesflow_hpo as hpo

# Your simulator + adapter (BayesFlow standard setup)
simulator = bf.simulators.make_simulator(...)
adapter = bf.adapters.Adapter(...)

# Run HPO with sensible defaults
study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    search_space=hpo.CompositeSearchSpace(
        inference_space=hpo.CouplingFlowSpace(),
        summary_space=hpo.DeepSetSpace(),
    ),
    validation_conditions={"N": [50, 100, 200]},
    n_trials=50,
    epochs=100,
    objective_metrics=["calibration_error", "nrmse"],
    sampler="tpe",  # or "gp", "botorch", "nsga2", "nsga3", "auto", "random"
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
| [Optimization](optimization.md) | Objective function, constraints, study management, sampler presets, QMC warm-up |
| [Validation](validation.md) | Validation datasets, metric registry, coverage, C2ST, SBC tests, result tables |
| [Results & Export](results.md) | Pareto extraction, lexicographic-Pareto selection, visualization, model export |
| [API Reference](api_reference.md) | Complete public API with signatures and descriptions |
| [Changelog](quality_report.md) | Changes implemented in the v0.2.0 workover and post-v0.2.0 enhancements |

## Installation

```bash
pip install bayesflow-hpo

# With optional dependencies
pip install bayesflow-hpo[dashboard]  # Optuna dashboard
pip install bayesflow-hpo[dev]        # Development tools
```

## Requirements

- Python >= 3.11
- BayesFlow >= 2.0.0
- Optuna >= 4.0.0
- Keras >= 3.9, < 3.13 (PyTorch backend recommended)
