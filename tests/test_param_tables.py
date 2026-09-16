"""The generated parameter tables, as a test.

The headline test is :func:`test_docs_are_current`, which fails when a
parameter is added, removed, renamed, or re-documented without rerunning
``scripts/gen_param_tables.py``.  That is the drift this generator exists to
make impossible to commit, and asserting it here means the docs are checked
by the same suite as the code rather than by whoever remembers.

The rest exercise the generator against synthetic inputs.  A generator with
no failing-case test is a generator nobody has proven can fail -- and the
tables it replaced were themselves wrong for months while looking perfectly
consistent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# A plain import, not importorskip: if the script under test fails to
# import, that must be a red suite, not a skipped one. The whole point
# of this module is that a vacuous pass is worse than a failure.
import gen_param_tables as gen  # noqa: E402


def test_docs_are_current() -> None:
    """Every generated region matches the code it describes."""
    assert gen.main(["--check", "--root", str(REPO_ROOT)]) == 0, (
        "docs/ is out of date with the code. "
        "Run: python scripts/gen_param_tables.py"
    )


# --------------------------------------------------------------------------
# Docstring parsing
# --------------------------------------------------------------------------

def _sample(a, b=3, *, c="x"):
    """Summary line.

    Parameters
    ----------
    a
        The first one.
    b
        The second one.

        A caveat that should not reach the table.
    c
        The third one.

    Returns
    -------
    None
    """


def test_description_is_the_lead_paragraph_only() -> None:
    params = {p.name: p for p in gen.collect_params(_sample)}
    assert params["b"].description == "The second one."


def test_required_and_defaulted_parameters_are_distinguished() -> None:
    params = {p.name: p for p in gen.collect_params(_sample)}
    assert params["a"].value is None
    assert params["b"].value == "3"
    assert params["c"].keyword_only is True


def test_returns_section_does_not_become_a_row() -> None:
    assert [p.name for p in gen.collect_params(_sample)] == ["a", "b", "c"]


def _undocumented(a, b=1):
    """Summary.

    Parameters
    ----------
    a
        Only this one is documented.
    """


def test_undocumented_parameter_is_an_error() -> None:
    """The gap the generator exists to close must not reappear as a blank cell."""
    with pytest.raises(ValueError, match="no docstring entry for: b"):
        gen.collect_params(_undocumented)


def _shared(a, b=1):
    """Summary.

    Parameters
    ----------
    a, b
        Shared description.
    """


def test_comma_separated_names_each_get_a_row() -> None:
    params = gen.collect_params(_shared)
    assert [p.name for p in params] == ["a", "b"]
    assert all(p.description == "Shared description." for p in params)


def _rst(a=None):
    """Summary.

    Parameters
    ----------
    a
        Uses ``literals``, :class:`~pkg.mod.Thing`, and a \\| pipe.
    """


def test_rst_is_converted_and_pipes_escaped() -> None:
    (param,) = gen.collect_params(_rst)
    assert "`literals`" in param.description
    assert "`Thing`" in param.description
    assert "``" not in param.description
    # An unescaped pipe would silently truncate the table row.
    assert "\\|" in param.description


# --------------------------------------------------------------------------
# Dataclass rendering
# --------------------------------------------------------------------------

@dataclass
class _Config:
    """Summary.

    Parameters
    ----------
    required
        No default.
    listed
        Built by a factory.
    """

    required: int
    listed: list[str] = field(default_factory=lambda: ["a"])


def test_default_factory_is_evaluated() -> None:
    params = {p.name: p for p in gen.collect_params(_Config)}
    assert params["required"].value is None
    assert params["listed"].value == "['a']"


def test_dataclass_block_keeps_the_source_literal() -> None:
    block = gen.render_dataclass_block(_Config, "_Config")
    assert "required: int" in block
    assert "field(default_factory=lambda : ['a'])" in block.replace(
        "lambda:", "lambda :"
    )


# --------------------------------------------------------------------------
# Splicing
# --------------------------------------------------------------------------

def test_splice_replaces_only_the_region() -> None:
    text = (
        "before\n"
        "<!-- BEGIN GENERATED: t -->\n"
        "old\n"
        "<!-- END GENERATED: t -->\n"
        "after\n"
    )
    out = gen.splice(text, "t", "new")
    assert out.startswith("before\n")
    assert out.endswith("after\n")
    assert "old" not in out
    assert "new" in out


@pytest.mark.parametrize("text", [
    "no markers at all",
    "<!-- BEGIN GENERATED: t -->\nunclosed",
    # Two pairs: which one did the author mean?
    "<!-- BEGIN GENERATED: t -->\n<!-- END GENERATED: t -->\n"
    "<!-- BEGIN GENERATED: t -->\n<!-- END GENERATED: t -->",
])
def test_splice_refuses_a_malformed_region(text: str) -> None:
    """A region that cannot be found must fail, not be skipped."""
    with pytest.raises(ValueError, match="exactly one"):
        gen.splice(text, "t", "new")


# --------------------------------------------------------------------------
# Restated defaults
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,rendered,expected", [
    # Restates the column exactly -> dropped.
    ("Stores loss curves. Default 10.", "10", "Stores loss curves."),
    ("Batches per epoch (default 50).", "50", "Batches per epoch."),
    # Says something the column does not -> kept, because for a `None`
    # default this sentence is the only place the real value appears.
    ("Hard cap. Defaults to `3 * n_trials`.", "None",
     "Hard cap. Defaults to `3 * n_trials`."),
    # Trailing gloss the strip would swallow -> left alone.
    ("Safety margin. Default 0.2 (20%).", "0.2",
     "Safety margin. Default 0.2 (20%)."),
])
def test_only_a_restated_default_is_stripped(
    text: str, rendered: str, expected: str
) -> None:
    assert gen._strip_restated_default(text, rendered) == expected


def test_check_mode_does_not_write(tmp_path: Path) -> None:
    doc = tmp_path / "docs" / "api_reference.md"
    doc.parent.mkdir(parents=True)
    original = (
        "<!-- BEGIN GENERATED: optimize-signature -->\n"
        "<!-- END GENERATED: optimize-signature -->\n"
    )
    doc.write_text(original, encoding="utf-8")

    monkeypatched = {"docs/api_reference.md": {"optimize-signature": "body"}}
    real_build = gen.build_regions
    gen.build_regions = lambda: monkeypatched  # type: ignore[assignment]
    try:
        assert gen.main(["--check", "--root", str(tmp_path)]) == 1
        assert doc.read_text(encoding="utf-8") == original
        assert gen.main(["--root", str(tmp_path)]) == 0
        assert "body" in doc.read_text(encoding="utf-8")
    finally:
        gen.build_regions = real_build  # type: ignore[assignment]
