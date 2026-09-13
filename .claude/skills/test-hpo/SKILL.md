---
name: test-hpo
description: Run bayesflow-hpo tests with the correct Keras backend. Pass an optional path filter as args (e.g., "test_validation/").
---

Run the bayesflow-hpo test suite with the PyTorch backend.

## Steps

1. Resolve the interpreter and assign it to `PY` for the shell you are using.
   This package is never installed globally — use the checkout's own virtualenv:

   ```bash
   PY=.venv/bin/python           # macOS / Linux
   PY=.venv/Scripts/python.exe   # Windows (Git Bash)
   ```

   In PowerShell the same assignment is `$PY = ".venv\Scripts\python.exe"`, and
   every `$PY` below is then used as `& $PY`.

   Never fall back to a bare `python` — a system interpreter has neither
   `pytest` nor `bayesflow_hpo`, and a venv belonging to another worktree would
   test that worktree's source.

2. If `.venv` does not exist, create it before running anything. This applies to
   each `git worktree` separately, since an editable install resolves to the
   source tree it was installed from:

   ```bash
   python -m venv .venv
   $PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   $PY -m pip install -e ".[dev]"
   ```

   Both installs are required: `[dev]` pulls no Keras backend, so installing it
   alone yields a venv that has `pytest` but fails at `import bayesflow`.

3. Determine the test target. `KERAS_BACKEND=torch` must be set for every run:
   - If args are provided:
     `KERAS_BACKEND=torch $PY -m pytest tests/{args} -v --tb=short`
   - If no args, run the full suite:
     `KERAS_BACKEND=torch $PY -m pytest tests/ -v --tb=short`

   In PowerShell, set the backend first: `$env:KERAS_BACKEND = 'torch'`.

4. Report results:
   - On success: confirm all tests passed with count
   - On failure: list failing tests with `file:line` references and the assertion message
   - Never report a pass you did not observe. If the suite could not run, say so.
