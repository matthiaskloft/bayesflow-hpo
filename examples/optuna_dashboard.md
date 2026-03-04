# Optuna Dashboard Integration

`bayesflow-hpo` supports Optuna's persistent storage backends and can be monitored with `optuna-dashboard`.

## 1. Install dashboard extras

```bash
pip install -e ".[dashboard]"
```

## 2. Run optimization with persistent storage

```python
import bayesflow_hpo as hpo

study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    param_keys=["theta"],
    data_keys=["x"],
    validation_conditions={"N": [50, 100, 200]},
    n_trials=200,
    storage="sqlite:///hpo_study.db",
    study_name="gaussian_hpo",
)
```

## 3. Start the dashboard

```bash
optuna-dashboard sqlite:///hpo_study.db
```

Then open <http://127.0.0.1:8080> and select `gaussian_hpo`.

## 4. Warm-start a larger study from a smaller one

```python
import optuna
import bayesflow_hpo as hpo

small = optuna.load_study(study_name="gaussian_hpo", storage="sqlite:///hpo_study.db")

large = hpo.create_study(
    study_name="gaussian_hpo_large",
    storage="sqlite:///hpo_study.db",
    warm_start_from=small,
    warm_start_top_k=30,
)
```
