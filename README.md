# bayesflow-hpo

Generic hyperparameter optimization for BayesFlow neural posterior estimation (NPE).

## Features

- Declarative search spaces for BayesFlow inference + summary networks
- Optuna multi-objective optimization (`calibration_error`, `param_score`)
- Fixed validation dataset generation and serialization
- Per-parameter validation metrics with aggregated calibration summaries
- Memory budget pre-check (`max_memory_mb`) to avoid likely OOM trials
- Custom network registration API for user-defined search spaces
- Warm-start study seeding from prior Optuna studies
- Generic objective and study helpers
- Pareto/front extraction and plotting helpers

## Advanced examples

- `examples/optuna_dashboard.md`

## Runnable notebooks

- `examples/quickstart.ipynb` — minimal end-to-end BayesFlow HPO run
- `examples/custom_summary_network.ipynb` — custom summary-space registration + HPO
- `examples/multi_objective.ipynb` — two-objective optimization + warm start

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
