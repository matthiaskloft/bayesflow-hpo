---
name: lint
description: Run ruff linting matching CI configuration. Pass "--fix" as args to auto-fix issues.
---

Run ruff check matching the CI pipeline for bayesflow-hpo.

## Steps

1. Determine mode:
   - If args contain `--fix`: run `ruff check --fix src/ tests/`
   - Otherwise: run `ruff check src/ tests/`
2. Report results:
   - On clean: confirm no issues found
   - On issues: list each with `file:line` references and the rule code
