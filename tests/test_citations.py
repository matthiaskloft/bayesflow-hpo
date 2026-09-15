"""The citation-consistency check, as a test.

``scripts/check_citations.py`` is runnable on its own; running it here too
means a citation added without a matrix entry fails the same suite as a broken
import, rather than waiting for someone to remember the script exists.

See the "Audit status" section of ``docs/references.md`` for what this guards
against and, more importantly, what it cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

check_citations = pytest.importorskip("check_citations")


def test_every_src_citation_is_backed_by_the_matrix() -> None:
    """Every author/year in ``src/`` has an entry, with its locators stated."""
    assert check_citations.main(["--root", str(REPO_ROOT)]) == 0


def test_matrix_parses_into_entries() -> None:
    """A guard on the checker itself.

    If the heading format of ``docs/references.md`` ever changes, the parser
    silently returns nothing and the check above passes vacuously -- every
    citation would be "missing" only if entries existed to miss. Anchor it on
    a couple of entries that certainly do exist.
    """
    references = (REPO_ROOT / "docs" / "references.md").read_text(encoding="utf-8")
    entries = check_citations.parse_matrix_entries(references)

    assert len(entries) > 20
    assert ("talts", "2018") in entries
    assert ("deb", "2002") in entries


def test_locator_is_attributed_to_the_nearer_citation() -> None:
    """A locator between two citations belongs to whichever is closer.

    Regression test for the reading that pinned "Algorithm 1" on Deb et al.
    (2002) in ``pruning_strategies.py`` when the sentence attributes it to
    MO-ASHA, i.e. to the Schmucker citation that immediately follows it.
    """
    text = (
        "(Deb et al., 2002), confirmed via MO-ASHA Algorithm 1 "
        "(Schmucker et al., 2021)"
    )
    citations = list(check_citations._CITATION_RE.finditer(text))
    assigned = check_citations._assign_locators(
        text, [(m.start(), m.end()) for m in citations]
    )

    assert ("algorithm", "1") not in assigned[0]
    assert ("algorithm", "1") in assigned[1]


def test_full_author_list_keys_on_the_first_surname() -> None:
    """"Lemos, Coogan, Hezaveh and Perreault-Levasseur (2023)" is one citation."""
    text = "Lemos, Coogan, Hezaveh and Perreault-Levasseur (2023)"
    matches = list(check_citations._CITATION_RE.finditer(text))

    assert len(matches) == 1
    assert check_citations._author_key(matches[0].group("authors")) == "lemos"
