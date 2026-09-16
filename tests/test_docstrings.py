"""The docstring-consistency check, as a test.

``scripts/check_docstrings.py`` is runnable on its own; running it here too
means a docstring that stops matching its signature fails the same suite as
a broken import.

Most tests below exercise the checker against synthetic trees rather than the
repository, and several assert it *fails*. That is the important half: the
first draft of this checker reported a clean bill of health for all 331
docstrings in the package while parsing none of them, because
``ast.get_docstring`` dedents and the parser expected entries to be indented.
A checker nobody has proven can fail is a checker nobody should believe --
hence also the vacuity floors, tested here.

The four defects in :func:`test_catches_each_historical_defect` are the real
ones an audit of the package found, reproduced so that a future refactor of
the checker cannot silently stop catching them.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

check_docstrings = pytest.importorskip("check_docstrings")


def _write_pkg(root: Path, modules: dict[str, str]) -> Path:
    """Lay out a synthetic ``src/bayesflow_hpo`` tree."""
    pkg = root / "src" / "bayesflow_hpo"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").touch()
    for name, source in modules.items():
        path = pkg / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return pkg


def _rules(findings) -> set[str]:
    return {f.rule for f in findings}


# --------------------------------------------------------------------------
# The repository itself
# --------------------------------------------------------------------------

def test_repository_docstrings_agree_with_the_code() -> None:
    assert check_docstrings.main(["--root", str(REPO_ROOT)]) == 0


def test_repository_is_actually_inspected() -> None:
    """Guard the guard: the run must see a plausible amount of material."""
    _, counts = check_docstrings.check_tree(REPO_ROOT / "src" / "bayesflow_hpo")
    assert counts.entries >= check_docstrings._MIN_EXPECTED_ENTRIES
    assert counts.xrefs >= check_docstrings._MIN_EXPECTED_XREFS


# --------------------------------------------------------------------------
# Each rule, on a synthetic tree
# --------------------------------------------------------------------------

def test_undocumented_parameter(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(a, b=1):
            """S.

            Parameters
            ----------
            a
                Documented.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert _rules(findings) == {"undocumented-parameter"}
    assert "b" in findings[0].message


def test_parameter_that_does_not_exist(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(a):
            """S.

            Parameters
            ----------
            a
                Documented.
            ghost
                Not a parameter.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert _rules(findings) == {"no-such-parameter"}


def test_paragraph_inside_parameters_is_malformed(tmp_path: Path) -> None:
    """The numpydoc trap: a column-0 line is read as a parameter name."""
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(a):
            """S.

            Parameters
            ----------
            a
                Documented.

            Returns a cleaned copy of the dict.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert _rules(findings) == {"malformed-entry"}


def test_wrong_default(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(a=100):
            """S.

            Parameters
            ----------
            a
                A thing (default 200).
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert _rules(findings) == {"wrong-default"}


def test_unresolvable_cross_reference(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, {"m.py": '''
        """Module.

        See :class:`~bayesflow_hpo.m.Nope` and
        :func:`~bayesflow_hpo.m.real`.
        """


        def real():
            """S."""
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert _rules(findings) == {"xref"}
    assert "Nope" in findings[0].message


def test_clean_tree_has_no_findings(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, {"m.py": '''
        """Module. See :func:`~bayesflow_hpo.m.f`."""


        def f(a, b=100, *, c=None):
            """S.

            Parameters
            ----------
            a
                Documented.
            b
                A thing (default 100).
            c
                Optional; defaults to ``"tpe"`` when ``None``.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


# --------------------------------------------------------------------------
# The false positives that kept this check out of CI
# --------------------------------------------------------------------------

def test_none_sentinel_resolution_is_not_drift(tmp_path: Path) -> None:
    """``= None`` documented by what it resolves to is correct, not wrong."""
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(n_trials=None):
            """S.

            Parameters
            ----------
            n_trials
                Cap. Defaults to ``3 * n_trials``.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


def test_default_factory_is_unwrapped(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, {"m.py": '''
        from dataclasses import dataclass, field


        @dataclass
        class C:
            """S.

            Parameters
            ----------
            metrics
                Keys. Default ``["a", "b"]``.
            """

            metrics: list = field(default_factory=lambda: ["a", "b"])
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


def test_defaulted_is_not_a_default_claim(tmp_path: Path) -> None:
    """"Not defaulted to ``None``" once parsed as a default of "ed"."""
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(seed=42):
            """S.

            Parameters
            ----------
            seed
                Seed. Not defaulted to ``None``: two runs would differ.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


def test_prose_mention_of_a_default_is_not_a_claim(tmp_path: Path) -> None:
    """"the default path", "defaults to the registry" are prose, not values."""
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(metrics=None, pool=5):
            """S.

            Parameters
            ----------
            metrics
                Resolved by the default registry lookup.
            pool
                Unlike the default path, this one is eager.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


def test_quote_style_does_not_matter(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(mode='pareto'):
            """S.

            Parameters
            ----------
            mode
                Mode (default ``"pareto"``).
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


def test_indented_section_heading_does_not_end_parameters(tmp_path: Path) -> None:
    """An indented "References" belongs to the entry it sits under.

    Reading it as a top-level heading truncated the Parameters block and
    reported every parameter after it as undocumented.
    """
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(a, b):
            """S.

            Parameters
            ----------
            a
                Documented.

                References
                ----------
                Someone (2020).
            b
                Also documented.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


# --------------------------------------------------------------------------
# Holes an independent review found in the first version of this checker
# --------------------------------------------------------------------------

def test_string_default_lie_is_caught(tmp_path: Path) -> None:
    """A quoted string is not a named constant.

    ``_normalize`` strips quotes, after which ``'tpe'`` is a valid
    identifier and the named-constant escape hatch swallowed it -- blinding
    the rule to every string default in the package, which is most of
    ``optimize()``'s.
    """
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(kind="tpe"):
            """S.

            Parameters
            ----------
            kind
                Sampler. Defaults to ``"nsga2"``.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert _rules(findings) == {"wrong-default"}


def test_true_string_default_is_not_flagged(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(kind="tpe"):
            """S.

            Parameters
            ----------
            kind
                Sampler. Defaults to ``"tpe"``.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


@pytest.mark.parametrize("default,claimed", [
    ("100", "1"),      # "1" is a substring of "100"
    ("0.05", "5"),     # "5" is a substring of "0.05"
    ("20", "2"),
])
def test_substring_numeric_lie_is_caught(
    tmp_path: Path, default: str, claimed: str
) -> None:
    """The substring fallback used to let a wrong number through."""
    pkg = _write_pkg(tmp_path, {"m.py": f'''
        def f(a={default}):
            """S.

            Parameters
            ----------
            a
                A thing (default {claimed}).
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert _rules(findings) == {"wrong-default"}


def test_grouped_digits_are_read_whole(tmp_path: Path) -> None:
    """"(default 1 000 000)" is one number, not a claim of 1."""
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(n=1000000):
            """S.

            Parameters
            ----------
            n
                Cap (default 1 000 000).
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


def test_nested_option_default_is_not_a_claim(tmp_path: Path) -> None:
    """A default of a sub-option is not a claim about the parameter.

    "For ``"primary"``, the metric defaults to ``objective_metrics[0]``"
    documents the tuple's second element, not ``strategy`` itself.
    """
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(strategy="dominance"):
            """S.

            Parameters
            ----------
            strategy
                One of ``"dominance"`` (default) or ``"primary"``. For
                ``"primary"``, the metric defaults to
                ``objective_metrics[0]``.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


def test_documented_kwargs_is_not_a_phantom(tmp_path: Path) -> None:
    """``**kwargs`` is an ordinary numpydoc entry, not a phantom parameter."""
    pkg = _write_pkg(tmp_path, {"m.py": '''
        def f(a, *args, **kwargs):
            """S.

            Parameters
            ----------
            a
                Thing.
            args
                More.
            kwargs
                Extra.
            """
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


def test_class_annotation_does_not_shadow_init(tmp_path: Path) -> None:
    """An unrelated class-level annotation must not hide ``__init__``."""
    pkg = _write_pkg(tmp_path, {"m.py": '''
        class C:
            """S.

            Parameters
            ----------
            alpha
                Thing.
            """

            _cache: dict = {}

            def __init__(self, alpha=1):
                pass
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


def test_inherited_dataclass_field_is_not_a_phantom(tmp_path: Path) -> None:
    """Only ``node.body`` is parsed, so a base's fields are invisible."""
    pkg = _write_pkg(tmp_path, {"m.py": '''
        from dataclasses import dataclass


        @dataclass
        class Base:
            """B."""

            shared: int = 1


        @dataclass
        class Child(Base):
            """S.

            Parameters
            ----------
            shared
                Inherited.
            own
                Mine.
            """

            own: int = 2
    '''})
    findings, _ = check_docstrings.check_tree(pkg)
    assert findings == []


# --------------------------------------------------------------------------
# Vacuity
# --------------------------------------------------------------------------

def test_empty_tree_fails_the_vacuity_floor(tmp_path: Path) -> None:
    """A parser that inspects nothing must not report success."""
    _write_pkg(tmp_path, {"m.py": '"""Nothing to check."""\n'})
    assert check_docstrings.main(["--root", str(tmp_path)]) == 2


def test_missing_package_is_an_error(tmp_path: Path) -> None:
    assert check_docstrings.main(["--root", str(tmp_path)]) == 2


# --------------------------------------------------------------------------
# The real defects, reproduced
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rule,source", [
    ("undocumented-parameter", '''
        def default_validate_fn(approximator, joint_metrics=None):
            """S.

            Parameters
            ----------
            approximator
                Trained approximator.
            """
    '''),
    ("malformed-entry", '''
        def _validate_metric_keys(raw):
            """S.

            Parameters
            ----------
            raw
                Raw metric dict.

            Returns a cleaned copy of the dict.
            """
    '''),
    ("xref", '''
        """Uses :class:`~bayesflow_hpo.types.CanonicalMetricName`."""
    '''),
    ("wrong-default", '''
        def f(num_batches=50):
            """S.

            Parameters
            ----------
            num_batches
                Batches per epoch (default 999).
            """
    '''),
])
def test_catches_each_historical_defect(
    tmp_path: Path, rule: str, source: str
) -> None:
    pkg = _write_pkg(tmp_path, {"m.py": source})
    findings, _ = check_docstrings.check_tree(pkg)
    assert rule in _rules(findings), f"{rule} no longer caught"
