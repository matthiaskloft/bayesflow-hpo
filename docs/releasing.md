# Publishing releases

Release preparation stays in a reviewed PR. The `Release` workflow in
`.github/workflows/release.yml` runs automatically on a pushed `vX.Y.Z` tag. It
checks (`scripts/release_notes.py`) that the tag matches the `pyproject.toml`
version, that the `__version__` fallback in `src/bayesflow_hpo/__init__.py`
agrees, that `CITATION.cff` carries the same version and the changelog's
release date, and that `CHANGELOG.md` has a dated `## X.Y.Z -- YYYY-MM-DD` entry; it
refuses commits that are not on `main`. Only stable three-part versions are
supported.

The workflow then reruns the full CI (`ci.yml`: lint, citations, docstrings,
mypy, tests on 3.11--3.13) against the tagged commit, builds the wheel and sdist
with `python -m build`, checks metadata with `twine check --strict`, and runs
`scripts/check_distribution.py`, which inspects both archives for local
runtime content and installs the wheel (with the CPU torch backend) into a
clean venv outside the checkout. Publishing downloads those same checked
artifacts; it does not rebuild them.

## One-time setup

1. Create GitHub environments `pypi` and `testpypi` under repository
   Settings → Environments. Require a reviewer for `pypi` and restrict it to
   `v*` **tags**; allow only the `main` **branch** for `testpypi`. Declaring an
   environment in YAML does not create reviewer rules.
2. On [PyPI](https://pypi.org/manage/account/publishing/) add a Trusted
   Publisher to `bayesflow-hpo` (or a pending publisher if the project does not
   exist yet): owner `matthiaskloft`, repository `bayesflow-hpo`, workflow
   `release.yml`, environment `pypi`.
3. Register the same on [TestPyPI](https://test.pypi.org/manage/account/publishing/)
   with environment `testpypi`. The two indexes have separate accounts.
4. After this workflow is merged, rehearse once on TestPyPI:

   ```console
   gh workflow run release.yml --ref main -f destination=testpypi
   ```

   `validate`, `ci`, `build` and `testpypi` must pass; `pypi` and
   `github-release` are skipped.

No API token belongs in repository secrets: only the upload jobs get
`id-token: write`, and they neither check out nor build code.

## Each release

1. In a PR, bump `version` in `pyproject.toml` and the `__version__` fallback
   in `src/bayesflow_hpo/__init__.py`, add a dated
   `## X.Y.Z -- YYYY-MM-DD` section to `CHANGELOG.md`, and set `version` and
   `date-released` in `CITATION.cff` to the same version and date.
2. Merge after review and wait for green `main` CI.
3. Tag the merged commit and push the tag:
   `git tag vX.Y.Z <merged-commit> && git push origin vX.Y.Z`. A tag pushed
   with a workflow's `GITHUB_TOKEN` does not start the workflow.
4. In the run, choose **Review deployments** → `pypi` → approve. After PyPI
   accepts the upload, the workflow creates the GitHub release from the
   changelog entry with the wheel and sdist attached.
5. Verify a fresh `pip install bayesflow-hpo==X.Y.Z` and `pip check`.

## Failure recovery

| Failure | Action |
| --- | --- |
| `validate` rejects the tag or metadata | Fix it in a new PR and tag a new version; never move a published tag. Rehearsals must run from `main`. |
| CI, `twine` or distribution checks fail | Fix the cause in a reviewed PR; upload jobs stay blocked. |
| Trusted Publishing authentication fails | Check owner, repository, `release.yml` filename and environment on the index, then rerun the failed upload job if no files were accepted. |
| Index rejects an existing version | Uploads are immutable; release a new version rather than re-uploading changed bytes. |
| PyPI succeeded, GitHub release failed | `gh run rerun RUN_ID --failed`, or create the release manually from the same tag, notes and artifacts. |
| A draft GitHub release exists | The job fails rather than treat it as published; finish the draft's assets and publish it manually. |

References: [PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/),
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/),
[GitHub workflow triggers](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).
