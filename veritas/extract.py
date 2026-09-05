"""HTTP + text extraction helpers shared by connectors.

Deliberately dependency-free (urllib + html.parser). Nothing here may crash a
mission: fetch functions return empty results on failure and connectors report
warnings instead of raising.
"""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from .config import settings

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 veritas-research/0.1"
)

_BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "h5",
    "h6", "blockquote", "pre", "br", "tr", "table", "ul", "ol", "header",
    "footer", "figcaption",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg", "template"):
            self._skip += 1
        elif tag in _BLOCK_TAGS and not self._skip:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg", "template") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str | bytes, max_chars: int = 60_000) -> str:
    """Strip markup, normalise whitespace, cap length."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    parser = _TextExtractor()
    try:
        parser.feed(raw)
    except Exception:
        pass
    text = "".join(parser.parts)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()[:max_chars]


def fetch_url(
    url: str,
    *,
    timeout: float | None = None,
    max_bytes: int = 2_000_000,
) -> tuple[str, str]:
    """GET a URL -> (final_url, html/text payload as str). Raises on failure."""
    timeout = timeout or settings.web_timeout_s
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept-Encoding": "identity",  # keep urllib simple
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        final_url = resp.geturl()
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            data = data[:max_bytes]
        ctype = resp.headers.get("Content-Type", "")
        if "charset=" in ctype:
            enc = ctype.split("charset=")[-1].split(";")[0].strip()
            try:
                return final_url, data.decode(enc, errors="replace")
            except LookupError:
                pass
        return final_url, data.decode("utf-8", errors="replace")


def fetch_readable(url: str, **kw) -> tuple[str, str, str]:
    """Fetch and strip to text -> (final_url, title, text). Empty on failure."""
    try:
        final_url, raw = fetch_url(url, **kw)
    except Exception:
        return url, "", ""
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.I | re.S)
    if m:
        title = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()[:200]
    return final_url, title, html_to_text(raw)


def _window(text: str, lo: int, hi: int, width: int) -> str:
    lo = max(0, lo - width)
    hi = min(len(text), hi + width)
    return text[lo:hi]


def find_passages(text: str, terms: list[str], *, width: int = 240, cap: int = 6) -> list[str]:
    """Return text windows surrounding occurrences of any term (CI).

    Windows are merged when overlapping and de-duplicated by exact text so a
    quoted passage is stable enough for verification to re-find it later.
    """
    text_l = text.lower()
    hits: list[tuple[int, int]] = []
    for term in terms:
        t = term.strip().lower()
        if len(t) < 3:
            continue
        start = 0
        while True:
            i = text_l.find(t, start)
            if i == -1:
                break
            hits.append((i, i + len(t)))
            start = i + len(t)
    if not hits:
        return []
    hits.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in hits:
        if merged and lo - merged[-1][1] <= 120:  # merge nearby hits
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    out: list[str] = []
    seen: set[str] = set()
    for lo, hi in merged[:cap]:
        w = _window(text, lo, hi, width).strip()
        w = re.sub(r"\s+", " ", w)
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def truncate(text: str, n: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def google_style_snippet(text: str, terms: list[str], n: int = 500) -> str:
    """Short snippet containing the most relevant term occurrence."""
    wins = find_passages(text, terms, width=n // 2, cap=1)
    if wins:
        return wins[0][:n]
    return truncate(text, n)
