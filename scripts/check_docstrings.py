#!/usr/bin/env python
"""Check that docstrings in ``src/`` agree with the code they document.

Like ``scripts/check_citations.py``, this is a *consistency* check. It cannot
tell whether a description is true; it can only tell whether the docstring
and the signature describe the same thing. That is a real limit, and it is
the point: the failure mode it exists to catch is a docstring that has
quietly stopped matching its function, which nothing else detects because a
wrong docstring renders exactly as well as a right one.

``scripts/gen_param_tables.py`` already forces this for ``optimize()`` and
``ObjectiveConfig``, whose tables are generated. This covers the other 450-odd
parameter entries, and it was written after an audit of them found four
defects: a public parameter with no entry at all, a ``Returns`` paragraph
sitting inside a ``Parameters`` block where numpydoc reads it as a parameter
named after the whole sentence, an undocumented argument beside two
documented ones, and a cross-reference to a symbol in a module that does not
contain it.

Four rules are enforced:

1. **Every entry names a real parameter.** A documented parameter that does
   not exist is a reader following an instruction that cannot work.
2. **Every parameter of a documented function has an entry.** The rule
   applies only to functions that already have a ``Parameters`` section:
   writing one is the choice, completing it is then not optional.
3. **Every asserted default matches the signature.** "(default 200)" beside
   ``= 100`` is the cheapest possible lie to tell and the hardest to notice.
4. **Every internal cross-reference resolves.** ``:class:`~pkg.mod.Name```
   for a ``Name`` that ``mod`` does not define renders as plain text and
   fails nothing.

Resolution for rule 4 is static -- the source tree is parsed, not imported --
so this script needs no backend, no install, and nothing but the standard
library, exactly like ``check_citations.py``.

Run it directly, or via ``pytest tests/test_docstrings.py``.

Usage
-----
``python scripts/check_docstrings.py [--root .]``

Exits non-zero, listing every violation, if any rule fails.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Finding",
    "ModuleIndex",
    "check_tree",
    "default_is_consistent",
    "main",
    "parameter_entries",
]

#: numpydoc section headings. A heading only ends a section at column 0; an
#: indented "References" belongs to the parameter entry above it.
_SECTIONS = frozenset({
    "Parameters", "Returns", "Yields", "Raises", "Warns", "Other Parameters",
    "Attributes", "Methods", "See Also", "Notes", "References", "Examples",
})

#: Vacuity floors. A checker that silently parses nothing reports success it
#: has not earned -- the first draft of this one did exactly that for all 331
#: docstrings in the package, because ``ast.get_docstring`` dedents and its
#: parser expected entries to be indented. These are set well below the
#: current counts so that ordinary edits never trip them, and a parser that
#: breaks outright always does.
_MIN_EXPECTED_ENTRIES = 250
_MIN_EXPECTED_XREFS = 20

#: An assertion about a default: "(default 200)", "default ``True``",
#: "defaults to 5", "Default: 0.05". Two things are deliberate.
#:
#: The trailing ``\b`` on ``defaults?``: without it, "Not defaulted to
#: ``None``" matched, capturing a value of "ed".
#:
#: The value must be backticked, quoted or numeric. Prose mentions a default
#: constantly -- "the default path", "defaults to the registry", "the default
#: pool" -- and treating the next English word as an asserted value produced
#: seven false positives reading "docstring says default via" and the like.
#: An assertion worth checking is written as a literal.
#: A bare number may be written with digit grouping -- "1 000 000" -- so a
#: space is consumed only when a digit follows it. Capturing just the "1"
#: made a correct docstring disagree with ``= 1000000``.
_DEFAULT_CLAIM = re.compile(
    r"\bdefaults?\b\s*(?:to|:|=)?\s*"
    r"(``[^`]+``|`[^`]+`|\"[^\"]*\"|'[^']*'"
    r"|[-+]?\d(?:[\d_,]|\.\d|\s(?=\d))*)",
    re.I,
)

#: ``:role:`target``` -- ``~pkg.mod.Name``, ``pkg.mod.Name`` or ``text <t>``.
_XREF = re.compile(r":(\w+):`([^`]+)`")

_PACKAGE = "bayesflow_hpo"


@dataclass(frozen=True)
class Finding:
    """One disagreement between a docstring and the code."""

    path: str
    lineno: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno} [{self.rule}] {self.message}"


# --------------------------------------------------------------------------
# Docstring parsing
# --------------------------------------------------------------------------

def _sections(doc: str) -> dict[str, list[str]]:
    """Split a *dedented* docstring into its numpydoc sections."""
    lines = doc.split("\n")
    out: dict[str, list[str]] = {}
    current: str | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        at_margin = bool(line) and not line[:1].isspace()
        underlined = (
            i + 1 < len(lines)
            and bool(lines[i + 1].strip())
            and set(lines[i + 1].strip()) == {"-"}
        )
        if at_margin and stripped in _SECTIONS and underlined:
            current = stripped
            out[current] = []
            continue
        if current is not None:
            if stripped and set(stripped) == {"-"}:
                continue
            out[current].append(line)
    return out


def parameter_entries(doc: str) -> dict[str, str]:
    """Map parameter name -> entry body, for the ``Parameters`` section.

    ``ast.get_docstring`` dedents, so an entry heading sits at column 0 and
    its body is indented. numpydoc allows one heading to name several
    parameters (``a, b``) and to carry a type (``a : int``); both are
    handled, and each name gets the shared body.
    """
    body = _sections(doc).get("Parameters")
    if body is None:
        return {}

    entries: dict[str, str] = {}
    names: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        for name in names:
            entries[name] = "\n".join(buf).strip()

    for line in body:
        stripped = line.strip()
        if not stripped:
            buf.append("")
            continue
        if not line[:1].isspace():
            flush()
            names = [
                n.strip() for n in stripped.split(":")[0].split(",") if n.strip()
            ]
            buf = []
        else:
            buf.append(stripped)
    flush()
    return entries


# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------

def _signature_params(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str | None]:
    """Parameter name -> default expression as written, or ``None``."""
    args = node.args
    out: dict[str, str | None] = {}
    positional = args.posonlyargs + args.args
    pad = len(positional) - len(args.defaults)
    for i, arg in enumerate(positional):
        if arg.arg in ("self", "cls"):
            continue
        default = args.defaults[i - pad] if i >= pad else None
        out[arg.arg] = ast.unparse(default) if default is not None else None
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out[arg.arg] = ast.unparse(default) if default is not None else None
    # ``*args`` and ``**kwargs`` are ordinary numpydoc entries. Omitting them
    # made a correct docstring fail the check as a phantom parameter -- a
    # false CI red, which is the one outcome a gate must not produce.
    if args.vararg is not None:
        out[args.vararg.arg] = None
    if args.kwarg is not None:
        out[args.kwarg.arg] = None
    return out


def _class_params(node: ast.ClassDef) -> dict[str, str | None]:
    """Annotated class attributes, merged with ``__init__``'s parameters.

    Both are merged rather than the first non-empty one winning: a plain
    class carrying an unrelated class-level annotation (``_cache: dict =
    {}``) would otherwise shadow its own ``__init__`` and report every
    documented argument as a phantom.
    """
    out: dict[str, str | None] = {}
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out[stmt.target.id] = (
                ast.unparse(stmt.value) if stmt.value is not None else None
            )
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name == "__init__":
                for name, default in _signature_params(stmt).items():
                    out.setdefault(name, default)
    return out


def _has_base(node: ast.ClassDef) -> bool:
    """Whether a class inherits, and so may document an inherited field.

    A dataclass subclass documenting a field declared on its parent is
    correct numpydoc. Only ``node.body`` is parsed here, so the parent's
    fields are invisible and would read as phantoms; the undocumented-
    parameter rule still applies, only the phantom rule is relaxed.
    """
    return bool([b for b in node.bases if not _is_object(b)])


def _is_object(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "object"


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

def _unwrap_field(expr: str) -> str:
    """Reduce ``field(default_factory=lambda: X)`` / ``field(default=X)`` to X.

    A dataclass default written through ``field()`` describes the same value
    the prose does; comparing the wrapper text against it is a false
    positive, and was one of the two that kept the first version of this
    check out of CI.
    """
    if not expr.startswith("field("):
        return expr
    try:
        call = ast.parse(expr, mode="eval").body
    except SyntaxError:  # pragma: no cover - defensive
        return expr
    if not isinstance(call, ast.Call):  # pragma: no cover - defensive
        return expr
    for kw in call.keywords:
        if kw.arg == "default":
            return ast.unparse(kw.value)
        if kw.arg == "default_factory":
            value = kw.value
            if isinstance(value, ast.Lambda):
                return ast.unparse(value.body)
            # `default_factory=list` and friends: the produced value is
            # whatever calling it gives, which is not statically known.
            return expr
    return expr


def _normalize(value: str) -> tuple[str, bool]:
    """Return the bare value and whether it was written as a quoted string.

    The flag matters: ``'tpe'`` with its quotes stripped is a valid Python
    identifier, and the named-constant escape below would then treat every
    string default in the package as unverifiable -- which it did, blinding
    the rule to ``'pareto'``, ``'dominance'``, ``'fixed_budget'`` and the
    rest of ``optimize()``'s string defaults.
    """
    text = value.strip().strip("`").strip().rstrip(").,;:")
    quoted = len(text) > 1 and text[0] in "\"'" and text[-1] in "\"'"
    if quoted:
        text = text[1:-1]
    return text, quoted


def _is_literal(text: str) -> bool:
    """Whether *text* is a Python literal this check can compare."""
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return False
    return True


def default_is_consistent(claimed: str, actual: str | None) -> bool:
    """Whether a prose default claim agrees with the signature."""
    if actual is None:
        # No default in the signature at all, so there is nothing to
        # contradict.
        return True
    actual = _unwrap_field(actual)
    want, want_quoted = _normalize(claimed)
    got, got_quoted = _normalize(actual)
    if not want_quoted and not _is_literal(want):
        # The claim is an expression, not a value: "defaults to
        # ``objective_metrics[0]``" documents which metric the
        # ``("primary", metric)`` tuple falls back to, not what
        # ``pruning_strategy`` itself defaults to. Nothing statically
        # checkable, and reading it as a claim about the parameter is wrong.
        return True
    # Compare as Python values where both sides are literals, so that
    # ``["a", "b"]`` and ``['a', 'b']`` -- the same list written with the
    # other quote character -- agree.
    try:
        if ast.literal_eval(want) == ast.literal_eval(got):
            return True
    except (ValueError, SyntaxError, TypeError, MemoryError):
        pass
    if got == "None" and want != "None":
        # A ``None`` default is a sentinel, and saying what it resolves to
        # -- "defaults to ``3 * n_trials``", "default ``"tpe"``" -- is the
        # documentation's job, not a contradiction of the signature. This
        # is by far the most common shape in the package, and reading it as
        # drift produced 23 false positives on a clean tree. Whether the
        # resolved value is right cannot be decided without running the
        # code, so it is out of scope for a consistency check.
        return True
    if want == got:
        return True
    strip = str.maketrans("", "", "_, ")
    if want.translate(strip) == got.translate(strip):
        return True
    try:
        if float(want.translate(strip)) == float(got.translate(strip)):
            return True
    except ValueError:
        pass
    # A named constant (``MAX_PARAM_COUNT``, ``DEFAULT_STORAGE``): prose may
    # spell the name or the value it holds, and this check cannot evaluate it
    # without importing. A quoted string is NOT a named constant, however much
    # its contents look like an identifier.
    if (
        not got_quoted
        and got.replace(".", "").isidentifier()
        and got not in ("None", "True", "False")
    ):
        return True
    # "0.05" claimed against "0.05 in fixed_budget mode" -- a value the
    # signature qualifies in prose. Restricted to a qualified `got`: without
    # that, "default 1" agreed with `= 100` because "1" is a substring of
    # "100", and "default 5" agreed with `= 0.05`.
    if " " in got:
        return want in got
    return False


# --------------------------------------------------------------------------
# Static cross-reference resolution
# --------------------------------------------------------------------------

class ModuleIndex:
    """Top-level names each module of the package defines or re-exports.

    Built by parsing the source tree rather than importing it, so the check
    runs with no backend and no install.
    """

    def __init__(self, package_root: Path) -> None:
        self.package = package_root.name
        self.modules: dict[str, set[str]] = {}
        self.members: dict[str, set[str]] = {}
        for path in sorted(package_root.rglob("*.py")):
            rel = path.relative_to(package_root).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts.pop()
            name = ".".join([self.package, *parts])
            tree = ast.parse(path.read_text(encoding="utf-8"))
            self.modules[name] = self._top_level_names(tree, name)

    def _top_level_names(self, tree: ast.Module, module: str) -> set[str]:
        names: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(stmt.name)
            elif isinstance(stmt, ast.ClassDef):
                names.add(stmt.name)
                self.members[f"{module}.{stmt.name}"] = {
                    m.name
                    for m in stmt.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                } | {
                    m.target.id
                    for m in stmt.body
                    if isinstance(m, ast.AnnAssign)
                    and isinstance(m.target, ast.Name)
                }
            elif isinstance(stmt, ast.Assign):
                names.update(
                    t.id for t in stmt.targets if isinstance(t, ast.Name)
                )
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name):
                    names.add(stmt.target.id)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                # Re-exports: `from .api import optimize` makes `optimize`
                # a name of THIS module, which is how the package's public
                # API is spelled in docstrings.
                for alias in stmt.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name.split(".")[0])
        return names

    def resolve(self, target: str) -> bool:
        """Whether a dotted target names something in the package."""
        if target in self.modules:
            return True
        for cut in range(len(target.split(".")), 0, -1):
            parts = target.split(".")
            module = ".".join(parts[:cut])
            rest = parts[cut:]
            if module not in self.modules:
                continue
            if not rest:
                return True
            if rest[0] not in self.modules[module]:
                return False
            if len(rest) == 1:
                return True
            qualified = f"{module}.{rest[0]}"
            if qualified in self.members:
                return rest[1] in self.members[qualified]
            # An imported name's members cannot be followed statically.
            return True
        return False


def _xref_targets(doc: str) -> list[str]:
    targets: list[str] = []
    for _role, raw in _XREF.findall(doc):
        target = raw.strip()
        if "<" in target:
            target = target.split("<", 1)[1].rstrip(">")
        target = target.lstrip("~").strip()
        if " " in target or not target:
            continue
        if target.split(".")[0] != _PACKAGE:
            continue  # external -- not ours to verify
        targets.append(target)
    return targets


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------

@dataclass
class Counts:
    """What the run actually inspected, for the vacuity floors."""

    entries: int = 0
    xrefs: int = 0


def check_tree(package_root: Path) -> tuple[list[Finding], Counts]:
    """Check every docstring under *package_root*."""
    index = ModuleIndex(package_root)
    findings: list[Finding] = []
    counts = Counts()

    for path in sorted(package_root.rglob("*.py")):
        rel = path.as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            doc = (
                ast.get_docstring(node)
                if isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef))
                else None
            )
            if not doc:
                continue

            for target in _xref_targets(doc):
                counts.xrefs += 1
                if not index.resolve(target):
                    findings.append(Finding(
                        rel, getattr(node, "lineno", 1), "xref",
                        f"cross-reference does not resolve: `{target}`",
                    ))

            if isinstance(node, ast.Module):
                continue

            entries = parameter_entries(doc)
            if not entries:
                continue
            counts.entries += len(entries)

            params = (
                _class_params(node)
                if isinstance(node, ast.ClassDef)
                else _signature_params(node)
            )
            documented = set(entries)
            actual = set(params)

            malformed = sorted(n for n in documented - actual if not n.isidentifier())
            for name in malformed:
                findings.append(Finding(
                    rel, node.lineno, "malformed-entry",
                    f"{node.name}: {name!r} is not a parameter name. A "
                    f"paragraph at column 0 inside a Parameters block is "
                    f"read as an entry -- indent it, or give it its own "
                    f"section.",
                ))

            phantom = sorted(
                n for n in documented - actual if n.isidentifier()
            )
            if phantom and isinstance(node, ast.ClassDef) and _has_base(node):
                # Could be an inherited field; this parser cannot see the base.
                phantom = []
            if phantom:
                findings.append(Finding(
                    rel, node.lineno, "no-such-parameter",
                    f"{node.name}: documented but not in the signature: "
                    f"{', '.join(phantom)}",
                ))

            missing = sorted(
                n for n in actual - documented if not n.startswith("_")
            )
            if missing:
                findings.append(Finding(
                    rel, node.lineno, "undocumented-parameter",
                    f"{node.name}: in the signature but not documented: "
                    f"{', '.join(missing)}",
                ))

            for name, body in entries.items():
                if name not in params:
                    continue
                claim = _DEFAULT_CLAIM.search(body)
                if claim and not default_is_consistent(
                    claim.group(1), params[name]
                ):
                    findings.append(Finding(
                        rel, node.lineno, "wrong-default",
                        f"{node.name}({name}): docstring says default "
                        f"{claim.group(1)}, signature says {params[name]}",
                    ))

    return findings, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check docstrings against code.")
    parser.add_argument("--root", default=".", type=Path)
    args = parser.parse_args(argv)

    package_root = args.root / "src" / _PACKAGE
    if not package_root.is_dir():
        print(f"no package at {package_root}", file=sys.stderr)
        return 2

    findings, counts = check_tree(package_root)

    # Vacuity floors before the verdict: a clean report from a parser that
    # inspected nothing is worse than a failure, because it is believed.
    if counts.entries < _MIN_EXPECTED_ENTRIES:
        print(
            f"only {counts.entries} parameter entries parsed, expected at "
            f"least {_MIN_EXPECTED_ENTRIES}. The parser is broken, not the "
            f"docstrings.",
            file=sys.stderr,
        )
        return 2
    if counts.xrefs < _MIN_EXPECTED_XREFS:
        print(
            f"only {counts.xrefs} internal cross-references found, expected "
            f"at least {_MIN_EXPECTED_XREFS}. The parser is broken, not the "
            f"docstrings.",
            file=sys.stderr,
        )
        return 2

    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        print(
            f"\n{len(findings)} docstring(s) disagree with the code.",
            file=sys.stderr,
        )
        return 1

    print(
        f"{counts.entries} parameter entries and {counts.xrefs} internal "
        f"cross-references in src/ all agree with the code."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
