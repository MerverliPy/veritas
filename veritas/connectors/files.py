"""File-system research (local notes/files and codebase surfaces).

One engine powers both :class:`LocalProvider` (arbitrary directories) and
:class:`CodeProvider` (git-tracked repos with a structural overview):

* discovery — pure-Python walk with sane exclusions (or ``git ls-files`` for
  code), capped file size, text/binary sniffing;
* matching — case-insensitive token search over lines, ranked by hit density,
  quoting the exact line window as the evidence passage with an ``#L#-#``
  anchor so the verifier re-reads the same lines;
* fetch — reads the anchored line range back from disk.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from ..extract import truncate
from ..schema import Evidence, Source, Surface
from .base import Provider

STOPWORDS = {
    "the", "and", "for", "are", "was", "with", "this", "that", "from",
    "what", "how", "why", "where", "which", "when", "your", "you", "has",
    "have", "its", "not", "but", "can", "will", "into", "about", "than",
    "there", "their", "they", "them", "these", "those", "also", "over",
}

MAX_FILE_BYTES = 1_500_000
MAX_LINES_READ = 60_000
_TEXTLIKE = {".md", ".txt", ".py", ".ts", ".tsx", ".js", ".jsx", ".json",
             ".yaml", ".yml", ".toml", ".sh", ".rs", ".go", ".java", ".rb",
             ".c", ".h", ".cpp", ".hpp", ".html", ".css", ".sql", ".ipynb",
             ".rst", ".tex", ".ini", ".cfg", ".conf", ".env", ".csv"}
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build",
              "__pycache__", ".cache", ".pytest_cache", ".mypy_cache",
              ".next", ".nuxt", "target", ".tox", ".nox", "coverage", "out",
              "site-packages", ".idea", ".vscode", ".terraform", "vendor"}
_SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
               "Cargo.lock", "composer.lock", "Pipfile.lock", ".DS_Store"}


def tokenize(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+", query.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS][:12]


def _sniff_text(path: Path) -> bool:
    if path.suffix.lower() in _TEXTLIKE:
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(2048)
    except OSError:
        return False
    if b"\x00" in head:
        return False
    return True


def _iter_files(root: Path) -> list[Path]:
    """Walk a directory returning candidate text files (no size cap here)."""
    files: list[Path] = []
    try:
        it = os.walk(root)
        for dirpath, dirnames, filenames in it:
            dirnames[:] = [d for d in dirnames
                           if d not in _SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                if name in _SKIP_FILES or name.endswith((".min.js", ".min.css")):
                    continue
                p = Path(dirpath) / name
                try:
                    if p.stat().st_size <= MAX_FILE_BYTES and _sniff_text(p):
                        files.append(p)
                except OSError:
                    continue
    except OSError:
        pass
    return files


def _git_tracked(root: Path) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True, timeout=20, check=True,
        ).stdout.decode(errors="replace")
    except (subprocess.SubprocessError, OSError):
        return []
    return [root / f for f in out.split("\0") if f and not f.startswith(".git")]


class _FileEngine:
    """Shared search logic. mode: 'local' | 'code'."""

    def __init__(self, root: Path, mode: str) -> None:
        self.root = root.resolve()
        self.mode = mode

    def candidate_files(self) -> list[Path]:
        if self.mode == "code":
            tracked = _git_tracked(self.root)
            if tracked:
                return [p for p in tracked if p.exists()
                        and p.stat().st_size <= MAX_FILE_BYTES]
            # not a git repo -> fall back to plain walk
            return _iter_files(self.root)
        return _iter_files(self.root)

    def search(self, query: str, limit: int) -> list[Evidence]:
        terms = tokenize(query)
        if not terms:
            return []
        out: list[Evidence] = []
        for path in self.candidate_files():
            try:
                rel = path.relative_to(self.root)
            except ValueError:
                rel = path
            rel_s = str(rel)
            hit_line = _first_matching_line(path, terms)
            if hit_line is None:
                continue
            snippet = _line_window(path, hit_line, terms, span=6)
            if snippet is None:
                continue
            text, start, end = snippet
            out.append(Evidence(
                source=Source(path=rel_s, title=f"{rel_s}:{start}",
                              surface=Surface.CODE if self.mode == "code" else Surface.LOCAL,
                              anchor=f"L{start}-L{end}"),
                passage=truncate(text, 700), kind="file",
            ))
        # rank: passages where more terms co-occur came first via _first_matching_line;
        # cap by limit after a coarse dedupe on path
        seen: set[str] = set()
        final: list[Evidence] = []
        for ev in out:
            if ev.source.path in seen:
                continue
            seen.add(ev.source.path)
            final.append(ev)
            if len(final) >= limit:
                break
        return final

    def fetch(self, source: Source) -> str | None:
        if not source.path:
            return None
        path = (self.root / source.path).resolve()
        if not path.is_file():
            # tolerate locators with #L already stripped by caller, else try suffix match
            return None
        start, end = _parse_anchor(source.anchor)
        try:
            with open(path, "r", errors="replace") as f:
                lines = f.readlines(MAX_LINES_READ + 1)
        except OSError as e:
            self.warnings.append(f"read failed {path}: {e}")
            return None
        if len(lines) > MAX_LINES_READ:
            lines = lines[:MAX_LINES_READ]
        if start is None:
            return "".join(lines)[:80_000]
        lines = lines[max(0, start - 1): min(end, len(lines))]
        return "".join(lines)[:80_000]

    def overview(self, max_entries: int = 14) -> str:
        if self.mode == "local":
            return f"local directory: {self.root}"
        # code overview: manifests, top-level layout, language histogram, git recency
        root = self.root
        lines: list[str] = [f"codebase: {root.name} at {root}"]
        tracked = _git_tracked(root)
        if tracked:
            try:
                log = subprocess.run(
                    ["git", "-C", str(root), "log", "--oneline", "-8"],
                    capture_output=True, timeout=15, check=True,
                ).stdout.decode(errors="replace").strip()
                if log:
                    lines.append("recent commits:\n" + "\n".join("  " + l for l in log.splitlines()))
            except subprocess.SubprocessError:
                pass
            files = tracked
        else:
            files = _iter_files(root)
        exts: dict[str, int] = {}
        for p in files:
            ext = p.suffix.lower().lstrip(".") or "none"
            exts[ext] = exts.get(ext, 0) + 1
        top_exts = ", ".join(f".{e}×{n}" for e, n in
                             sorted(exts.items(), key=lambda kv: -kv[1])[:6])
        lines.append(f"tracked files: {len(files)} | extensions: {top_exts}")
        # top-level entries
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            entries = []
        kept = [e for e in entries if e not in _SKIP_DIRS and not e.startswith(".")]
        lines.append("top-level: " + ", ".join(kept[:max_entries]))
        for manifest in ("README.md", "pyproject.toml", "package.json",
                         "Cargo.toml", "go.mod", "AGENTS.md", "CLAUDE.md", "Makefile"):
            if (root / manifest).exists():
                lines.append(f"manifest present: {manifest}")
        return "\n".join(lines)


def _parse_anchor(anchor: str) -> tuple[int | None, int | None]:
    m = re.fullmatch(r"L(\d+)(?:-L?(\d+))?", anchor or "")
    if not m:
        return None, None
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else start
    return start, end


def _first_matching_line(path: Path, terms: list[str]) -> int | None:
    """Line number (1-based) of the first line containing any term; score by
    count of distinct terms across the file to rank richer hits earlier."""
    best = (0, None)
    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if i > MAX_LINES_READ:
                    break
                lw = line.lower()
                if any(t in lw for t in terms):
                    n = sum(1 for t in terms if t in lw)
                    if n > best[0]:
                        best = (n, i)
                        if n == len(terms):
                            break
    except OSError:
        return None
    return best[1]


def _line_window(path: Path, center: int, terms: list[str], span: int) -> tuple[str, int, int] | None:
    start = max(1, center - span)
    end = center + span
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines(MAX_LINES_READ + 1)
    except OSError:
        return None
    window = lines[start - 1 : min(end, len(lines))]
    text = "".join(window)
    return text, start, min(end, len(lines))


class LocalProvider(Provider):
    surface = Surface.LOCAL

    def __init__(self, root: Path | str) -> None:
        super().__init__()
        self._engine = _FileEngine(Path(root), mode="local")

    def search(self, query: str, limit: int = 8) -> list[Evidence]:
        return self._engine.search(query, limit)

    def fetch(self, source: Source) -> str | None:
        return self._engine.fetch(source)


class CodeProvider(Provider):
    surface = Surface.CODE

    def __init__(self, root: Path | str) -> None:
        super().__init__()
        self._engine = _FileEngine(Path(root), mode="code")

    def search(self, query: str, limit: int = 8) -> list[Evidence]:
        return self._engine.search(query, limit)

    def fetch(self, source: Source) -> str | None:
        return self._engine.fetch(source)

    def overview(self) -> str:
        return self._engine.overview()
