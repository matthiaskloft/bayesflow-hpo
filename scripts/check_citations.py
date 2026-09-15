#!/usr/bin/env python
"""Check that citations in ``src/`` are stated in ``docs/references.md``.

This is a *consistency* check, not a correctness check. It cannot tell whether
a locator is true; it can only tell whether the same claim is recorded in the
one place this project designates as the record. That is a real limit, and it
is the point: the failure mode it exists to catch is twin drift -- two files
making the same claim, one corrected and one not -- which is how
``validation_callback.py`` kept a stale "MO-ASHA's dominance-based promotion"
wording for a full PR after ``pruning_strategies.py`` had it corrected.

Two rules are enforced:

1. **Every author/year cited in ``src/`` has an entry in the matrix.** A
   citation nobody recorded is a citation nobody checked.
2. **Every locator attached to such a citation appears in that entry.** If a
   docstring says "Talts et al. (2018), Theorem 1", the Talts entry must
   mention Theorem 1. A locator that exists only in a docstring is one nobody
   can audit against the source without re-deriving where it came from.

Run it directly, or via ``pytest tests/test_citations.py``.

Usage
-----
``python scripts/check_citations.py [--root .]``

Exits non-zero, listing every violation, if either rule fails.

References
----------
The failure modes this guards against are recorded in the "Audit status"
section of ``docs/references.md`` and in
``docs/contributing-references.md``.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Citation", "collect_source_citations", "main", "parse_matrix_entries"]

#: Words that look like a surname to the citation regex but never are. Without
#: this, "Published as a conference paper at ICLR 2017" reads as a citation.
_NON_AUTHOR_TOKENS = frozenset(
    {
        "april",
        "august",
        "december",
        "february",
        "icml",
        "iclr",
        "january",
        "july",
        "june",
        "march",
        "may",
        "neurips",
        "november",
        "october",
        "optuna",
        "pmlr",
        "september",
    }
)

#: Locator kinds worth pinning. Each maps to the canonical word used when
#: looking the locator up in an entry, so "Thm. 1", "Theorem 1" and "Thms. 1--2"
#: all resolve to the same claim.
_LOCATOR_KINDS = {
    "alg": "algorithm",
    "algorithm": "algorithm",
    "algorithms": "algorithm",
    "algs": "algorithm",
    "def": "definition",
    "definition": "definition",
    "definitions": "definition",
    "defs": "definition",
    "eq": "equation",
    "equation": "equation",
    "equations": "equation",
    "eqs": "equation",
    "prop": "proposition",
    "proposition": "proposition",
    "propositions": "proposition",
    "props": "proposition",
    "sec": "section",
    "section": "section",
    "sections": "section",
    "secs": "section",
    "thm": "theorem",
    "theorem": "theorem",
    "theorems": "theorem",
    "thms": "theorem",
}

_SURNAME = r"[A-Z][A-Za-zÀ-ɏ'’-]*"

#: "Talts et al. (2018)" / "(Schmucker et al., 2021)" / "Deb & Jain (2014)" /
#: "Lemos, Coogan, Hezaveh and Perreault-Levasseur (2023)". Spelled-out author
#: lists must be matched whole, or the trailing surnames read as a citation of
#: their own and the entry lookup fails on an author who is not first.
_CITATION_RE = re.compile(
    rf"(?P<authors>{_SURNAME}(?:,\s*{_SURNAME})*"
    rf"(?:\s*(?:&|and)\s*{_SURNAME})?"
    r"(?:\s+et\s+al\.?)?)"
    r"[,\s]*\(?(?P<year>(?:19|20)\d{2})\)?"
)

#: "Thm. 1", "Section 3.2", "Algs. 1--2", "Equation (7)".
_LOCATOR_RE = re.compile(
    r"\b(?P<kind>" + "|".join(sorted(_LOCATOR_KINDS, key=len, reverse=True)) + r")"
    r"\.?\s*\(?(?P<number>\d+(?:\.\d+)*)\)?",
    re.IGNORECASE,
)

#: How far *after* a citation a locator still counts as attached to it. Long
#: enough to span "Talts et al. (2018), Theorem 1 (Sec. 4.1)", short enough not
#: to swallow the next sentence.
_LOCATOR_WINDOW_AFTER = 120

#: How far *before* a citation a locator counts as attached to it. Much tighter,
#: because a locator that precedes its citation does so immediately -- "MO-ASHA
#: Algorithm 1 lines 1--6 (Schmucker et al., 2021)". Give this the same reach as
#: the forward window and any algorithm number mentioned earlier in a paragraph
#: gets pinned on whichever citation happens to come next.
_LOCATOR_WINDOW_BEFORE = 25


def _fold(text: str) -> str:
    """Return ``text`` lowercased with diacritics stripped.

    ``src/`` writes "Lopez-Paz" and "Modrak" in some places and the accented
    forms in others; the matrix uses one spelling. Folding makes those the same
    key, so the check reports genuine absences rather than typography.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.replace("’", "'").lower()


def _author_key(authors: str) -> str:
    """Reduce an author string to its first surname, folded."""
    head = re.split(r"\s*(?:,|&|\sand\s)\s*|\s+et\s+al", authors)[0]
    return _fold(head.strip())


@dataclass(frozen=True)
class Citation:
    """One author/year mention, with any locators attached to it."""

    author: str
    year: str
    locators: frozenset[tuple[str, str]]
    path: Path
    line: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.author, self.year)

    def __str__(self) -> str:
        return f"{self.author.title()} ({self.year})"


def parse_matrix_entries(references: str) -> dict[tuple[str, str], str]:
    """Map ``(folded first surname, year)`` to the body of its matrix entry.

    Entries are the ``### Author, A., & Other, B. (YYYY)`` headings of
    ``docs/references.md``; the body runs to the next heading.
    """
    entries: dict[tuple[str, str], str] = {}
    pattern = re.compile(r"^### (?P<heading>.+?\((?P<year>(?:19|20)\d{2})\))\s*$", re.M)
    matches = list(pattern.finditer(references))
    for index, match in enumerate(matches):
        is_last = index + 1 == len(matches)
        end = len(references) if is_last else matches[index + 1].start()
        surname = match.group("heading").split(",")[0].strip()
        entries[(_fold(surname), match.group("year"))] = references[match.start() : end]
    return entries


def _assign_locators(
    text: str, spans: list[tuple[int, int]]
) -> list[set[tuple[str, str]]]:
    """Assign each locator in ``text`` to the nearest citation in ``spans``.

    Nearest on *either* side, because a locator is as often introduced before
    its citation as after it: in "MO-ASHA Algorithm 1 lines 1--6 (Schmucker et
    al., 2021)" the algorithm belongs to the citation that follows, and
    attaching it to whatever citation happened to come earlier in the sentence
    invents a claim the docstring never made.
    """
    assigned: list[set[tuple[str, str]]] = [set() for _ in spans]
    if not spans:
        return assigned
    for match in _LOCATOR_RE.finditer(text):
        locator = (_LOCATOR_KINDS[match.group("kind").lower()], match.group("number"))
        best: int | None = None
        best_distance = float("inf")
        for index, (start, end) in enumerate(spans):
            if match.start() >= end:
                distance = match.start() - end
                limit = _LOCATOR_WINDOW_AFTER
            elif match.end() <= start:
                distance = start - match.end()
                limit = _LOCATOR_WINDOW_BEFORE
            else:
                distance, limit = 0, _LOCATOR_WINDOW_AFTER
            if distance <= limit and distance < best_distance:
                best, best_distance = index, distance
        if best is not None:
            assigned[best].add(locator)
    return assigned


def collect_source_citations(root: Path) -> list[Citation]:
    """Collect every citation in the Python sources under ``root/src``."""
    citations: list[Citation] = []
    for path in sorted((root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [
            match
            for match in _CITATION_RE.finditer(text)
            if _author_key(match.group("authors"))
            and _author_key(match.group("authors")) not in _NON_AUTHOR_TOKENS
        ]
        locators = _assign_locators(text, [(m.start(), m.end()) for m in found])
        for match, attached in zip(found, locators):
            citations.append(
                Citation(
                    author=_author_key(match.group("authors")),
                    year=match.group("year"),
                    locators=frozenset(attached),
                    path=path.relative_to(root),
                    line=text.count("\n", 0, match.start()) + 1,
                )
            )
    return citations


def _entry_states_locator(entry: str, kind: str, number: str) -> bool:
    """Whether ``entry`` mentions the ``kind number`` locator in any spelling."""
    folded = _fold(entry)
    for alias, canonical in _LOCATOR_KINDS.items():
        if canonical != kind:
            continue
        pattern = rf"\b{re.escape(alias)}\.?\s*\(?{re.escape(number)}\b"
        if re.search(pattern, folded):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (default: the checkout this script lives in).",
    )
    args = parser.parse_args(argv)

    matrix_path = args.root / "docs" / "references.md"
    entries = parse_matrix_entries(matrix_path.read_text(encoding="utf-8"))
    citations = collect_source_citations(args.root)

    missing_entries: list[Citation] = []
    missing_locators: list[tuple[Citation, str, str]] = []
    for citation in citations:
        entry = entries.get(citation.key)
        if entry is None:
            missing_entries.append(citation)
            continue
        for kind, number in sorted(citation.locators):
            if not _entry_states_locator(entry, kind, number):
                missing_locators.append((citation, kind, number))

    for citation in missing_entries:
        print(
            f"{citation.path}:{citation.line}: {citation} is cited but has no "
            f"entry in docs/references.md",
            file=sys.stderr,
        )
    for citation, kind, number in missing_locators:
        print(
            f"{citation.path}:{citation.line}: {citation} cites "
            f"{kind} {number}, which its docs/references.md entry does not state",
            file=sys.stderr,
        )

    failures = len(missing_entries) + len(missing_locators)
    if failures:
        print(
            f"\n{failures} citation(s) not backed by docs/references.md. "
            f"Add the claim to the matrix, or correct the citation.",
            file=sys.stderr,
        )
        return 1

    print(f"{len(citations)} citation(s) in src/ all backed by docs/references.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
