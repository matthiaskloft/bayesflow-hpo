"""The two version declarations must agree.

``pyproject.toml`` is the release version; ``bayesflow_hpo.__version__``
prefers installed distribution metadata and falls back to a literal when
there is none. That fallback is what an import from an unpacked source
checkout -- no ``pip install`` -- reports, so a release that bumps only
``pyproject.toml`` makes such a checkout identify itself as the *previous*
release. This test is the reason that cannot happen twice.

Parsed rather than imported: reading ``__init__.py`` as text needs no
backend, so the check runs even where ``import bayesflow_hpo`` cannot.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _fallback_version() -> str:
    """Return the ``__version__`` literal assigned in the except handler."""
    source = (ROOT / "src" / "bayesflow_hpo" / "__init__.py").read_text(
        encoding="utf-8"
    )
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            for statement in ast.walk(handler):
                if not isinstance(statement, ast.Assign):
                    continue
                targets = [
                    t.id for t in statement.targets if isinstance(t, ast.Name)
                ]
                if "__version__" in targets and isinstance(
                    statement.value, ast.Constant
                ):
                    return str(statement.value.value)
    raise AssertionError(
        "no `__version__ = <literal>` fallback found in "
        "src/bayesflow_hpo/__init__.py -- if the fallback was removed or "
        "restructured, update this test rather than deleting it"
    )


def test_fallback_version_matches_pyproject() -> None:
    assert _fallback_version() == _pyproject_version()


def test_changelog_documents_the_released_version() -> None:
    """The version being shipped must have a changelog section."""
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    version = _pyproject_version()
    headings = [
        line for line in changelog.splitlines() if line.startswith("## ")
    ]
    assert any(
        heading[3:].strip().split()[0] == version for heading in headings
    ), f"CHANGELOG.md has no `## {version}` section; headings: {headings[:5]}"
