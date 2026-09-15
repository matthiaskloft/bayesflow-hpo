# Citing sources in this package

`CLAUDE.md` requires every implementation to be backed by a full text or by
package documentation. This file says what that looks like in practice, and
why it is shaped the way it is.

The short version:

- **[`docs/references.md`](references.md) is the record.** A claim is stated
  there once. Docstrings cite it; they do not restate it.
- **Every locator names its edition and the date it was checked.**
- **`scripts/check_citations.py` enforces consistency**, and cannot enforce
  truth. Read the limits below before trusting a green run.

## The four failure modes this is designed against

These are not hypothetical. PR #86 corrected thirteen citation errors, every
one found by hand and only because someone asked directly whether the
implementations had been checked against full texts.

1. **Inherited-and-unchecked.** A claim copied from a commit message, an
   issue, or a neighbouring docstring, never read back. This produced the
   original gamma/Modrák misattribution and an "Equation 7" that propagated
   to eight places.
2. **Silent staleness.** A claim true when written and false later. The
   Optuna entries said "4.9.0 is the version installed and verified against
   here" -- correct until the dependency floor moved to 5.0.0, which
   falsified the sentence without anyone touching it.
3. **Edition mismatch.** A locator checked against a different edition than
   the one at hand. The deleted `docs/references/sobol1967_qmc.md` gave
   English-translation page numbers for a reading of the Russian original.
4. **Twin drift.** Two files making the same claim, one corrected and one
   not. `validation_callback.py` kept the "MO-ASHA's dominance-based
   promotion" wording for a full PR after `pruning_strategies.py` had it
   corrected, because nothing linked them.

## The convention

### One record

State a claim in `docs/references.md`. In `src/`, cite author and year, and
the locator if the claim depends on one:

```python
"""...uniform rank statistics (Talts et al., 2018, Thm. 1)."""
```

Do not reproduce the argument in the docstring. If a reader needs the
supporting quotation, it belongs in the matrix entry, where there is one copy
of it to keep correct. This is the direct countermeasure to (4).

### Every locator carries an edition and a date

In the matrix entry, close with a parenthetical naming what was read and
when:

```
(Thm. 1, Sec. 4.1 -- arXiv:1804.06788, verified 2026-09-11.)
```

The edition is not ceremony. Sobol' (1967) has two paginations, Gneiting
(2011) is cited from JASA but verified from arXiv, and Bland & Altman (1986)
is verified from a 1987 corrected reprint. A page or section number means
nothing without saying which of these it came from -- that is (3). The date
is what makes (2) visible: a stamp from before a dependency bump is a prompt
to re-check, where an undated sentence looks equally true forever.

For library behaviour, "verified" means executed against the installed
version, and the version is named. Do not verify an API claim from its
documentation alone when you can run it.

### Say what you did not check

An entry with no stamp is an entry nobody has read back. Leave it visibly
unstamped rather than implying otherwise, and list it in the audit section.
The point of (1) is that an unchecked claim is indistinguishable from a
checked one unless the difference is recorded.

### Corrections stay in the entry, struck through

When a locator turns out to be wrong, fix it and keep a short note saying
what it used to be. Several entries carry these. They cost two lines and they
are the only evidence that a given sentence has been contested -- including
when a *correction* was itself wrong, as happened to Emmerich & Deutz's
Proposition 9, dismissed as non-existent by the second-pass audit and
reinstated by the third.

**Write the rejected locator in `~~strikethrough~~`:**

```
(An earlier version of the `objectives.py` comment cited ~~Theorem 3.1~~,
which does not exist. The statistic is Theorem 3.)
```

This is not cosmetic. The check scans an entry for the locators it states,
and a correction note names the locator it is rejecting -- so before this
convention, an entry saying "cited Theorem 3.1, which does not exist"
*accepted* a docstring citing Theorem 3.1. Reverting either error this
tooling was built to catch would have passed CI. Struck locators are removed
before matching, so the history stays readable and stops licensing the
mistake it records.

Strike only the rejected reading. A locator that is wrong in one role and
right in another stays live in the second: Section 3.6 of Li et al. is not
where the `eta = 3` default lives, but it is where the "3 or 4"
recommendation lives, and the entry states both.

## The check

```bash
python scripts/check_citations.py
```

It also runs as `tests/test_citations.py` and as its own CI job. It asserts
two things:

1. Every author/year cited in `src/` has an entry in `docs/references.md`.
2. Every locator attached to such a citation appears in that entry.

### What it cannot do

It compares two texts. It has no access to the papers, so **a claim that is
wrong in both places passes**. Full-text verification stays a human or agent
judgement; the check only shrinks the surface on which drift goes unnoticed.

It also attributes locators to citations by proximity, which is a heuristic.
A locator is attached to the nearest citation, with a generous window after
and a tight one before, because a locator that precedes its citation does so
immediately. Prose that separates a locator from its citation by a clause can
still be mis-read. If the check flags something you believe is correct,
prefer moving the citation next to the locator over arguing with the tool --
the ambiguity it tripped on is usually real ambiguity for a reader too.

### When it fails

Read the paper. The four findings this check produced on its first run were
one bogus theorem number (`objectives.py` cited Linhart et al. Theorem 3.1;
the paper numbers theorems flat and the statistic is Theorem 3), one citation
of a section that does not exist attached to a claim the paper contradicts
(`training.py` on Shallue et al. Sec. 5.1), and two correct claims the matrix
had simply never recorded. Only the last two are fixed by editing
`docs/references.md`.
