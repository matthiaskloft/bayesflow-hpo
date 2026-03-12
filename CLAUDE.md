# bayesflow-hpo


## Scope

- Generic HPO package for BayesFlow 2.x models
- Focus on reusable search spaces, objective, validation, and result analysis
- Defaults for otoptional search space parameters MUST follow bayesflow defaults, but users can override them
- Keep domain-specific simulator logic out of this package



## Conventions

- Python >= 3.11
- Keras backend: PyTorch preferred
- src/ layout + setuptools
- pytest + ruff
