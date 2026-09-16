"""The metric tables in ``docs/validation.md``, against the registry.

A metric registered ``kind="diagnostic"`` is computed and reported but
*raises* if passed in ``objective_metrics``. That makes its kind the single
most actionable fact in the table -- a reader who trusts an unmarked row
writes a call that cannot run.

The table has been wrong about this twice. Before the docs audit it marked
one of eight diagnostics; the audit's own rewrite still marked only four,
and an independent review caught the other four. Hence this test rather than
a third round of careful reading.

The table is not generated because ``outputs`` is registered for only 4 of
the 22 metrics, so its "Output Keys" column has no machine-readable source.
Pinning the one column that can drift dangerously is the affordable half.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bayesflow_hpo.validation.registry import _JOINT, _KINDS, _REGISTRY

DOC = Path(__file__).resolve().parent.parent / "docs" / "validation.md"


def _table_rows() -> dict[str, str]:
    """Map metric name -> its markdown table row in validation.md."""
    rows: dict[str, str] = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*`([a-z0-9_]+)`\s*\|", line)
        if match:
            rows.setdefault(match.group(1), line)
    return rows


def _kind(name: str) -> str:
    return _KINDS.get(name, "objective")


#: The explicit marker, not the substring. Half these rows cite
#: ``bf.diagnostics.root_mean_squared_error``, whose module path contains
#: "diagnostic" while the metric is objective-ready.
_MARKED = re.compile(
    r"\(diagnostic\)|diagnostic only|\|\s*diagnostic\s*\|", re.IGNORECASE
)


def _is_marked_diagnostic(row: str) -> bool:
    return bool(_MARKED.search(row))


@pytest.mark.parametrize(
    "name",
    sorted(n for n in list(_REGISTRY) + list(_JOINT) if _kind(n) == "diagnostic"),
)
def test_diagnostic_metrics_are_marked_in_the_docs(name: str) -> None:
    rows = _table_rows()
    assert name in rows, (
        f"{name!r} is a registered metric with no row in docs/validation.md"
    )
    assert _is_marked_diagnostic(rows[name]), (
        f"{name!r} is registered kind='diagnostic' -- passing it in "
        f"objective_metrics raises -- but its row in docs/validation.md does "
        f"not say so:\n  {rows[name]}"
    )


@pytest.mark.parametrize(
    "name",
    sorted(n for n in list(_REGISTRY) + list(_JOINT) if _kind(n) == "objective"),
)
def test_objective_metrics_are_not_marked_diagnostic(name: str) -> None:
    """The converse: an objective-ready metric must not be warned off."""
    row = _table_rows().get(name)
    if row is None:
        pytest.skip(f"{name!r} has no row in validation.md")
    assert not _is_marked_diagnostic(row), (
        f"{name!r} is registered kind='objective' but its row in "
        f"docs/validation.md calls it diagnostic:\n  {row}"
    )


def test_the_test_sees_the_table() -> None:
    """Vacuity floor: a regex that matches nothing would pass everything."""
    rows = _table_rows()
    assert len(rows) >= 15, (
        f"only {len(rows)} metric rows found in {DOC.name}; the row regex is "
        f"broken, not the documentation"
    )
