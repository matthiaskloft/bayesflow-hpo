---
name: test-hpo
description: Run bayesflow-hpo tests with the correct Keras backend. Pass an optional path filter as args (e.g., "test_validation/").
---

Run the bayesflow-hpo test suite with the PyTorch backend.

## Steps

1. Resolve the interpreter. This package is never installed globally — use the
   checkout's own virtualenv:
   - Windows: `.venv/Scripts/python.exe`
   - macOS/Linux: `.venv/bin/python`

   If it does not exist, create it before running anything (this applies to each
   `git worktree` separately, since an editable install resolves to the source
   tree it was installed from):

   ```bash
   python -m venv .venv
   .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   .venv/Scripts/python -m pip install -e ".[dev]"    # .venv/bin/python elsewhere
   ```

   Both steps are required: `[dev]` pulls no Keras backend, so installing it
   alone yields a venv that has `pytest` but fails at `import bayesflow`.

   Never fall back to a bare `python` — a system interpreter has neither
   `pytest` nor `bayesflow_hpo`, and a venv belonging to another worktree would
   test that worktree's source.

2. Set the environment: `KERAS_BACKEND=torch`
3. Determine the test target, with `PY` the interpreter from step 1:
   - If args are provided, run: `KERAS_BACKEND=torch $PY -m pytest tests/{args} -v --tb=short`
   - If no args, run the full suite: `KERAS_BACKEND=torch $PY -m pytest tests/ -v --tb=short`
4. Report results:
   - On success: confirm all tests passed with count
   - On failure: list failing tests with `file:line` references and the assertion message
   - Never report a pass you did not observe. If the suite could not run, say so.
