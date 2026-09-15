"""The citation-consistency check, as a test.

``scripts/check_citations.py`` is runnable on its own; running it here too
means a citation added without a matrix entry fails the same suite as a broken
import, rather than waiting for someone to remember the script exists.

Several tests below exercise the checker against synthetic trees rather than
the repository. A checker with no failing-case test is a checker nobody has
proven can fail, and the first version of this one passed a real violation
because it could not see the citation at all.

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

STUB_MATRIX_HEADER = "# References\n\n"

#: Padding sizes for the synthetic trees, at the checker's own vacuity floors
#: so that a test meaning to exercise one rule is never failed by the other.
_MIN_ENTRIES = check_citations._MIN_EXPECTED_ENTRIES
_MIN_CITATIONS = check_citations._MIN_EXPECTED_CITATIONS


def _write_tree(root: Path, references: str, source: str) -> Path:
    """Lay out a minimal ``docs/references.md`` + ``src/`` pair under ``root``."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "references.md").write_text(references, encoding="utf-8")
    (root / "src" / "module.py").write_text(source, encoding="utf-8")
    return root


def _filler_name(index: int) -> str:
    """A distinct, purely alphabetic surname.

    Alphabetic because the citation pattern requires a surname to be letters:
    a digit in the name stops it matching, which silently produced zero
    citations the first time these fixtures were written.
    """
    return "Aa" + chr(ord("a") + index // 26) + chr(ord("a") + index % 26)


def _matrix_with(entries: int, body: str = "") -> str:
    """A stub matrix carrying enough entries to clear the vacuity floor.

    Covers at least as many filler names as ``_many_citations`` emits, so the
    padding needed to clear the floors never itself trips rule 1 and turns a
    targeted test into a test of the padding.
    """
    count = max(entries, _MIN_CITATIONS)
    filler = "".join(
        f"### {_filler_name(index)}, A. (19{index:02d})\n\nPadding.\n\n"
        for index in range(count)
    )
    return STUB_MATRIX_HEADER + filler + body


def _many_citations(count: int) -> str:
    """Source text with enough citations to clear the vacuity floor."""
    return "\n".join(
        f'"""{_filler_name(index)} et al. (19{index:02d})."""' for index in range(count)
    )


# --- The real repository -------------------------------------------------


def test_every_src_citation_is_backed_by_the_matrix() -> None:
    """Every author/year in ``src/`` has an entry, with its locators stated."""
    assert check_citations.main(["--root", str(REPO_ROOT)]) == 0


def test_matrix_parses_into_entries() -> None:
    """A guard on the checker's matrix-side parser."""
    references = (REPO_ROOT / "docs" / "references.md").read_text(encoding="utf-8")
    entries = check_citations.parse_matrix_entries(references)

    assert len(entries) >= check_citations._MIN_EXPECTED_ENTRIES
    assert ("talts", "2018") in entries
    assert ("deb", "2002") in entries


def test_source_parses_into_citations() -> None:
    """A guard on the checker's source-side parser.

    The counterpart to the test above, and the one that was missing when a
    regex gap made ``search_spaces/training.py``'s Shallue citation invisible.
    An empty result here would make every other check pass vacuously.
    """
    citations = check_citations.collect_source_citations(REPO_ROOT)
    keys = {citation.key for citation in citations}

    assert len(citations) >= check_citations._MIN_EXPECTED_CITATIONS
    assert ("schmucker", "2021") in keys
    assert ("linhart", "2023") in keys
    assert ("shallue", "2019") in keys


# --- Failure paths -------------------------------------------------------


def test_missing_entry_fails(tmp_path: Path) -> None:
    """A citation with no matrix entry is reported."""
    root = _write_tree(
        tmp_path,
        _matrix_with(check_citations._MIN_EXPECTED_ENTRIES),
        _many_citations(check_citations._MIN_EXPECTED_CITATIONS)
        + '\n"""Backed by nothing: Nobody et al. (2001)."""\n',
    )

    assert check_citations.main(["--root", str(root)]) == 1


def test_missing_locator_fails(tmp_path: Path) -> None:
    """A locator absent from an otherwise-present entry is reported."""
    entry = "### Talts, S. (2018)\n\nAn entry that mentions Theorem 1 only.\n\n"
    root = _write_tree(
        tmp_path,
        _matrix_with(check_citations._MIN_EXPECTED_ENTRIES, entry),
        _many_citations(check_citations._MIN_EXPECTED_CITATIONS)
        + '\n"""Talts et al. (2018), Theorem 4."""\n',
    )

    assert check_citations.main(["--root", str(root)]) == 1


def test_stated_locator_passes(tmp_path: Path) -> None:
    """The same citation passes once the entry states the locator."""
    entry = "### Talts, S. (2018)\n\nAn entry that mentions Thm. 4 in an alias.\n\n"
    root = _write_tree(
        tmp_path,
        _matrix_with(check_citations._MIN_EXPECTED_ENTRIES, entry),
        _many_citations(check_citations._MIN_EXPECTED_CITATIONS)
        + '\n"""Talts et al. (2018), Theorem 4."""\n',
    )

    assert check_citations.main(["--root", str(root)]) == 0


def test_unparseable_matrix_is_an_error_not_a_pass(tmp_path: Path) -> None:
    """An empty parse exits 2 rather than reporting a clean run.

    The failure mode this forecloses: the heading format changes, the matrix
    parses into nothing, every citation trivially has no entry to contradict,
    and CI goes green on a check that read nothing.
    """
    root = _write_tree(tmp_path, "no headings here", _many_citations(30))

    assert check_citations.main(["--root", str(root)]) == 2


def test_citationless_source_is_an_error_not_a_pass(tmp_path: Path) -> None:
    """A source tree with no citations found exits 2, not 0."""
    root = _write_tree(
        tmp_path,
        _matrix_with(check_citations._MIN_EXPECTED_ENTRIES),
        "x = 1\n",
    )

    assert check_citations.main(["--root", str(root)]) == 2


# --- Parsing details -----------------------------------------------------


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


@pytest.mark.parametrize(
    "text",
    [
        "Talts et al. (2018), Theorem 1",
        "(Schmucker et al., 2021)",
        "Deb & Jain (2014)",
        "Lemos, Coogan, Hezaveh and Perreault-Levasseur (2023)",
        "Shallue et al.'s (2019, Sec. 4)",
        "(Gneiting, 2011)",
    ],
)
def test_citation_shapes_are_recognised(text: str) -> None:
    """Every way this repository actually writes a citation is seen.

    The possessive case is the one that went missing: the checker passed on a
    docstring reading "Shallue et al.'s (2019, Sec. 4)" only because it could
    not see the citation, and so never checked the locator.
    """
    assert _recognised(text)


@pytest.mark.parametrize(
    "text",
    [
        "# Copyright 2024 The BayesFlow Developers",
        "see PyTorch 2019 release notes",
        "Accepted at AISTATS 2021",
        "raise ValueError('Expected 2020 samples')",
        "Verified 2026-09-11 against arXiv",
        "Published as a conference paper at ICLR 2017",
    ],
)
def test_prose_that_is_not_a_citation_is_ignored(text: str) -> None:
    """A capitalised word before a year is not by itself a citation.

    These all passed the original denylist approach and would have hard-failed
    CI on perfectly ordinary text -- including, pointedly, this package's own
    verification stamp.
    """
    assert not _recognised(text)


def _recognised(text: str) -> bool:
    """Whether the checker would treat ``text`` as containing a citation."""
    return any(
        check_citations._author_key(match.group("authors"))
        not in check_citations._NON_AUTHOR_TOKENS
        and check_citations._looks_like_a_citation(text, match)
        for match in check_citations._CITATION_RE.finditer(text)
    )


@pytest.mark.parametrize(
    ("entry", "kind", "number", "expected"),
    [
        ("Thm. 1 is the result", "theorem", "1", True),
        ("Theorem 1 is the result", "theorem", "1", True),
        ("Theorem 10 is the result", "theorem", "1", False),
        ("Section 3.1 is the locator", "section", "3", True),
        ("Sections 4 and 5", "section", "5", True),
        ("Algs. 1--2 are the procedure", "algorithm", "1", True),
        ("no locator at all", "theorem", "1", False),
    ],
)
def test_entry_locator_matching(
    entry: str, kind: str, number: str, expected: bool
) -> None:
    """Alias spellings resolve; a longer number is not a prefix match."""
    assert check_citations._entry_states_locator(entry, kind, number) is expected
