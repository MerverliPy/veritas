"""Local/code connector behaviour against real temp trees (no network)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from veritas import Source, Surface
from veritas.connectors import build_providers


def make_tree(tmp_path: Path) -> Path:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "alpha.md").write_text(
        "The WidgetBatch tool runs every night at 03:00.\n"
        "WidgetBatch version 4.1 was tagged in May.\n")
    (tmp_path / "notes" / "deep" ).mkdir()
    (tmp_path / "notes" / "deep" / "beta.txt").write_text(
        "WidgetBatch upstream is dormant since 2023.\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text(
        "WidgetBatch secret internals nobody should search\n" * 5)
    (tmp_path / "binary.dat").write_bytes(b"\x00\x01\x02WidgetBatch\x00\x03")
    return tmp_path / "notes"


def test_local_search_returns_anchored_evidence(tmp_path: Path):
    tree = make_tree(tmp_path)
    p = build_providers([Surface.LOCAL], local_root=tree)[0]
    evs = p.search("WidgetBatch nightly version", limit=5)
    locators = {e.source.locator() for e in evs}
    assert any("alpha.md#L" in l for l in locators)   # anchored line range
    assert any("deep/beta.txt#L" in l for l in locators)
    # node_modules and binary junk excluded
    assert not any("node_modules" in l for l in locators)
    assert not any("binary.dat" in l for l in locators)
    # passages contain the matched line text
    passages = " ".join(e.passage for e in evs)
    assert "WidgetBatch" in passages


def test_fetch_returns_exact_anchored_lines(tmp_path: Path):
    tree = make_tree(tmp_path)
    p = build_providers([Surface.LOCAL], local_root=tree)[0]
    evs = p.search("runs every night", limit=1)
    assert len(evs) == 1
    text = p.fetch(evs[0].source)
    assert text and "03:00" in text and "WidgetBatch" in text


def test_fetch_missing_file_returns_none(tmp_path: Path):
    tree = make_tree(tmp_path)
    p = build_providers([Surface.LOCAL], local_root=tree)[0]
    assert p.fetch(Source(path="nope.md", title="nope", surface=Surface.LOCAL)) is None


def test_code_surface_overview_and_git_tracked(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# tiny\n")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "mod.py").write_text("def run():\n    return 42\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "seed"], check=True)

    p = build_providers([Surface.CODE], code_root=repo)[0]
    ov = p.overview()
    assert "codebase:" in ov
    assert "manifest present: README.md" in ov

    evs = p.search("def run return", limit=3)
    assert len(evs) >= 1
    assert evs[0].source.surface is Surface.CODE
    assert "pkg/mod.py" in evs[0].source.locator()
    text = p.fetch(evs[0].source)
    assert "def run" in text


def test_providers_built_per_surface(tmp_path: Path):
    from veritas.connectors.web import WebProvider
    ps = build_providers([Surface.WEB, Surface.LOCAL], local_root=tmp_path)
    assert [type(p).__name__ for p in ps] == ["WebProvider", "LocalProvider"]
    # WebProvider without keys describes keyless engines
    assert isinstance(ps[0], WebProvider)
    assert "wikipedia" in ps[0].describe() or "ddg" in ps[0].describe()
