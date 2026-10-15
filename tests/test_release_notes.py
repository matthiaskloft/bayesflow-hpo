"""``scripts/release_notes.py`` gates every release on matching metadata."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_notes", ROOT / "scripts" / "release_notes.py"
)
assert _spec is not None and _spec.loader is not None
release_notes_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_notes_module)
release_notes = release_notes_module.release_notes


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "bayesflow_hpo").mkdir(parents=True)
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copy(
        ROOT / "src" / "bayesflow_hpo" / "__init__.py",
        tmp_path / "src" / "bayesflow_hpo" / "__init__.py",
    )
    shutil.copy(ROOT / "CHANGELOG.md", tmp_path / "CHANGELOG.md")
    shutil.copy(ROOT / "CITATION.cff", tmp_path / "CITATION.cff")
    return tmp_path


def _version(root: Path) -> str:
    import tomllib

    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    return str(tomllib.loads(text)["project"]["version"])


def test_current_version_releasable(repo: Path) -> None:
    version = _version(repo)
    notes = release_notes(repo, f"refs/tags/v{version}", "push")
    assert notes.strip()
    assert not notes.startswith("## ")


def test_rehearsal_from_main(repo: Path) -> None:
    assert release_notes(repo, "refs/heads/main", "workflow_dispatch")


@pytest.mark.parametrize(
    ("ref", "event"),
    [
        ("refs/tags/v0.0.0", "push"),
        ("refs/heads/feature", "workflow_dispatch"),
        ("refs/heads/main", "pull_request"),
    ],
)
def test_wrong_ref_or_event_rejected(repo: Path, ref: str, event: str) -> None:
    with pytest.raises(ValueError):
        release_notes(repo, ref, event)


def test_missing_changelog_entry_rejected(repo: Path) -> None:
    version = _version(repo)
    (repo / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CHANGELOG"):
        release_notes(repo, f"refs/tags/v{version}", "push")


def test_stale_fallback_version_rejected(repo: Path) -> None:
    version = _version(repo)
    init = repo / "src" / "bayesflow_hpo" / "__init__.py"
    init.write_text(
        init.read_text(encoding="utf-8").replace(
            f'__version__ = "{version}"', '__version__ = "0.0.1"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="__version__"):
        release_notes(repo, f"refs/tags/v{version}", "push")


@pytest.mark.parametrize(
    ("field", "stale"), [("version", "0.0.1"), ("date-released", "2000-01-01")]
)
def test_stale_citation_rejected(repo: Path, field: str, stale: str) -> None:
    version = _version(repo)
    cff = repo / "CITATION.cff"
    text = "\n".join(
        f"{field}: {stale}" if line.startswith(f"{field}:") else line
        for line in cff.read_text(encoding="utf-8").splitlines()
    )
    cff.write_text(text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CITATION"):
        release_notes(repo, f"refs/tags/v{version}", "push")
