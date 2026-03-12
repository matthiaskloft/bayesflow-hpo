---
name: test-hpo
description: Run bayesflow-hpo tests with the correct Keras backend. Pass an optional path filter as args (e.g., "test_validation/").
---

Run the bayesflow-hpo test suite with the PyTorch backend.

## Steps

1. Set the environment: `KERAS_BACKEND=torch`
2. Determine the test target:
   - If args are provided, run: `KERAS_BACKEND=torch python -m pytest tests/{args} -v --tb=short`
   - If no args, run the full suite: `KERAS_BACKEND=torch python -m pytest tests/ -v --tb=short`
3. Report results:
   - On success: confirm all tests passed with count
   - On failure: list failing tests with `file:line` references and the assertion message
