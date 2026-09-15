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

#: Venue and publisher names that appear next to a year in exactly the shape a
#: citation takes -- "*ICLR 2017*", "AISTATS 2021, PMLR 130". The positive test
#: below rejects most non-citations on shape alone; these survive it because
#: they are genuinely written the way a citation is written.
_NON_AUTHOR_TOKENS = frozenset(
    {
        "aistats",
        "arxiv",
        "cvpr",
        "iclr",
        "icml",
        "ijcai",
        "jmlr",
        "keras",
        "neurips",
        "nips",
        "optuna",
        "pmlr",
        "pytorch",
        "scipy",
        "tensorflow",
        "uai",
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
#: "Lemos, Coogan, Hezaveh and Perreault-Levasseur (2023)" /
#: "Shallue et al.'s (2019)". Two things this has to get right:
#:
#: - Spelled-out author lists must be matched whole, or the trailing surnames
#:   read as a citation of their own and the entry lookup fails on an author
#:   who is not first.
#: - The possessive is a normal way to write a citation in prose and must not
#:   hide one. `search_spaces/training.py` says "Shallue et al.'s (2019,
#:   Sec. 4)"; without the `'s` branch the whole citation went unseen, and with
#:   it the locator check caught a missing entry on the first run.
_CITATION_RE = re.compile(
    rf"(?P<authors>{_SURNAME}(?:,\s*{_SURNAME})*"
    rf"(?:\s*(?:&|and)\s*{_SURNAME})?"
    r"(?:\s+et\s+al\.?)?)"
    r"(?:'s|\u2019s)?"
    r"[,\s]*(?P<open>\()?(?P<year>(?:19|20)\d{2})(?P<close>\))?"
)

#: Separators inside a locator run. Kept to unambiguous list and range joins:
#: a bare comma would let "Sec. 3, 2018 edition" read the year as a locator.
#: The matrix side adds the comma back (see ``_ENTRY_RUN_SEPARATORS``), because
#: bibliographic prose lists locators that way and the year sits in the
#: heading, not mid-sentence.
_RUN_SEPARATORS = r"--|\u2013|-|and"

#: The matrix writes "Secs. 1, 4--5"; the comma is load-bearing there.
_ENTRY_RUN_SEPARATORS = rf"{_RUN_SEPARATORS}|,"

_NUMBER = r"\d+(?:\.\d+)*"

#: "Thm. 1", "Section 3.2", "Algs. 1--2", "Equation (7)", "Theorems 1 and 3".
#:
#: The ``numbers`` group captures the whole run, not just its head. A plural
#: kind that stopped at the first number let the rest through unchecked:
#: "Theorems 1 and 99" reported only Theorem 1, so an entry stating Theorem 1
#: silently accepted the unsupported Theorem 99.
_LOCATOR_RE = re.compile(
    r"\b(?P<kind>" + "|".join(sorted(_LOCATOR_KINDS, key=len, reverse=True)) + r")"
    rf"\.?\s*\(?(?P<numbers>{_NUMBER}(?:\s*(?:{_RUN_SEPARATORS})\s*{_NUMBER})*)\)?",
    re.IGNORECASE,
)

#: Markdown strikethrough. Entries use it to mark a locator a correction note
#: is *rejecting*, so that recording the history of an error does not quietly
#: license the error. See ``_accepted_text``.
_STRIKETHROUGH_RE = re.compile(r"~~.+?~~", re.DOTALL)

#: Floors below which the check is assumed broken rather than satisfied. Set
#: well under the current counts (~29 entries, ~48 citations) so ordinary
#: editing never trips them, and well above zero so a parser that has stopped
#: matching does. See the guard in ``main()``.
_MIN_EXPECTED_ENTRIES = 20
_MIN_EXPECTED_CITATIONS = 25

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


def _looks_like_a_citation(text: str, match: re.Match[str]) -> bool:
    """Whether a surname-then-year match is written the way a citation is.

    A denylist of words that are not surnames does not scale: every capitalised
    word before a year joins it ("Copyright 2024", "Expected 2020 samples",
    "Verified 2026-09-11"), and a list that has to grow to stay correct fails
    open on whatever was not thought of. Test the shape instead. A citation
    carries at least one of:

    - an ``et al.``, or an ``&``/``and`` joining two surnames;
    - a parenthesised year, ``Talts (2018)``;
    - enclosure in parentheses as a whole, ``(Gneiting, 2011)``.

    Ordinary prose that happens to put a capitalised word before a year has
    none of these.
    """
    authors = match.group("authors")
    if re.search(r"\bet\s+al|&|\sand\s", authors):
        return True
    if match.group("open") and match.group("close"):
        return True
    before = text[: match.start()].rstrip()
    if not before.endswith("("):
        return False
    # The closing bracket may already have been consumed as the year's own,
    # which is how "(Gneiting, 2011)" parses: no opening bracket on the year,
    # but the citation as a whole is parenthesised.
    after = text[match.end() :].lstrip()
    return bool(match.group("close")) or after.startswith(")")


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
        kind = _LOCATOR_KINDS[match.group("kind").lower()]
        numbers = re.split(rf"\s*(?:{_RUN_SEPARATORS})\s*", match.group("numbers"))
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
            assigned[best].update((kind, number) for number in numbers)
    return assigned


def collect_source_citations(root: Path) -> list[Citation]:
    """Collect every citation in the Python sources under ``root/src``."""
    citations: list[Citation] = []
    for path in sorted((root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [
            match
            for match in _CITATION_RE.finditer(text)
            if _author_key(match.group("authors")) not in _NON_AUTHOR_TOKENS
            and _looks_like_a_citation(text, match)
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


def _accepted_text(entry: str) -> str:
    """Return ``entry`` with its rejected locators removed.

    Entries record the history of a corrected citation on purpose -- it is the
    only evidence that a sentence has been contested. But a correction note
    names the locator it is rejecting, and scanning the whole entry therefore
    accepts it: before this, reverting `objectives.py` to "Theorem 3.1" or
    `training.py` to "Sec. 5.1" -- the two errors this check was written to
    catch -- passed, because the entries mention those locators while saying
    they are wrong.

    The convention is markdown strikethrough, which `docs/TODO.md` already
    uses for the same purpose and which renders as the note means it. A
    locator inside ``~~ ~~`` is stated as rejected and never satisfies a
    citation.
    """
    return _STRIKETHROUGH_RE.sub(" ", entry)


def _entry_states_locator(entry: str, kind: str, number: str) -> bool:
    """Whether ``entry`` mentions the ``kind number`` locator in any spelling.

    Matching is deliberately loose in one direction and strict in the other.
    ``Section 3`` is satisfied by an entry that says ``Section 3.1``, since an
    entry that locates a claim more precisely than the docstring has not
    contradicted it. ``Theorem 1`` is *not* satisfied by ``Theorem 10``.

    Ranges and lists count for every number they contain, so ``Secs. 3--4``
    and ``Sections 4 and 5`` satisfy a docstring citing either endpoint. An
    entry writing the range is stating both, and forcing it to spell them out
    separately would make the entry worse to read in order to please the
    check.
    """
    folded = _fold(_accepted_text(entry))
    aliases = [
        alias for alias, canonical in _LOCATOR_KINDS.items() if canonical == kind
    ]
    run = rf"{_NUMBER}(?:\s*(?:{_ENTRY_RUN_SEPARATORS})\s*{_NUMBER})*"
    for alias in aliases:
        for match in re.finditer(rf"\b{re.escape(alias)}\.?\s*\(?({run})", folded):
            stated = re.split(rf"\s*(?:{_ENTRY_RUN_SEPARATORS})\s*", match.group(1))
            # A dotted subsection satisfies its parent: an entry that locates
            # the claim at "Section 3.1" has not contradicted a docstring
            # citing "Section 3", it has been more precise than it.
            if any(one == number or one.startswith(f"{number}.") for one in stated):
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

    # A checker that finds nothing reports success, which is the one failure
    # this tool must not have: a regression in the citation regex, a moved
    # `src/`, or a wrong --root would all degrade to a green run that proves
    # nothing. Neither side of the comparison is ever legitimately empty, so
    # treat an empty one as a broken checker rather than a clean repository.
    if len(entries) < _MIN_EXPECTED_ENTRIES:
        print(
            f"{matrix_path} parsed into {len(entries)} entries, expected at "
            f"least {_MIN_EXPECTED_ENTRIES}. The heading format has probably "
            f"changed and this check is no longer reading the matrix.",
            file=sys.stderr,
        )
        return 2
    if len(citations) < _MIN_EXPECTED_CITATIONS:
        print(
            f"Found {len(citations)} citation(s) in {args.root / 'src'}, "
            f"expected at least {_MIN_EXPECTED_CITATIONS}. The citation "
            f"pattern has probably stopped matching; this check is not "
            f"looking at what it thinks it is.",
            file=sys.stderr,
        )
        return 2

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
