---
name: lint
description: Run ruff linting matching CI configuration. Pass "--fix" as args to auto-fix issues.
---

Run ruff check matching the CI pipeline for bayesflow-hpo.

## Steps

1. Resolve the interpreter and assign it to `PY` — the checkout's own
   virtualenv, never a bare `python` (see the `test-hpo` skill for how to create
   it if missing):

   ```bash
   PY=.venv/bin/python           # macOS / Linux
   PY=.venv/Scripts/python.exe   # Windows (Git Bash)
   ```

   In PowerShell: `$PY = ".venv\Scripts\python.exe"`, then invoke as `& $PY`.

2. Determine mode:
   - If args contain `--fix`: run `$PY -m ruff check --fix src/ tests/`
   - Otherwise: run `$PY -m ruff check src/ tests/`
3. Report results:
   - On clean: confirm no issues found
   - On issues: list each with `file:line` references and the rule code
