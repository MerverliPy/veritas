"""Web provider: keyless search engines + direct URL fetch, with optional
paid search (Tavily/Brave/Serper) preferred when keys are present.

Keyless engines used (all public APIs/endpoints):
  * DuckDuckGo HTML results   — general web fallback
  * Wikipedia search + intro extracts
  * arXiv API                 — papers
  * Hacker News (Algolia)     — tech/community discussion + linked pages
  * GitHub repository search  — repos & docs (unauthenticated, rate-limited)

Passages are quoted verbatim so the verifier can re-fetch and re-check the
exact text rather than trusting the search snippet.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from ..config import settings
from ..extract import (UA, fetch_readable, find_passages, google_style_snippet,
                       html_to_text)
from ..schema import Evidence, Source, Surface
from .base import Provider

_NS = {"a": "http://www.w3.org/2005/Atom"}


def _get_json(url: str, *, timeout: float | None = None, headers: dict | None = None) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout or settings.web_timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _get_text(url: str, *, timeout: float | None = None, headers: dict | None = None) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout or settings.web_timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Individual engines
# ---------------------------------------------------------------------------


def duckduckgo(query: str, limit: int = 5) -> list[Evidence]:
    """DDG HTML endpoint. Fragile by nature; returns [] on any failure."""
    out: list[Evidence] = []
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    raw = _get_text(url)
    # results are <a class="result__a" href="...">Title</a> ... <a class="result__snippet">
    for m in list(re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', raw, re.S))[:limit]:
        href, title_html = m.group(1), m.group(2)
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        # DDG wraps real urls in a redirect param
        u = re.search(r"uddg=([^&]+)", href)
        final = urllib.parse.unquote(u.group(1)) if u else href
        if not final.startswith("http"):
            continue
        out.append(Evidence(source=Source(url=final, title=title, surface=Surface.WEB),
                            passage=title, kind="search"))
    return out


def wikipedia(query: str, limit: int = 5) -> list[Evidence]:
    out: list[Evidence] = []
    api = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
           "&srlimit=%d&srsearch=%s" % (min(limit, 8), urllib.parse.quote(query)))
    data = _get_json(api)
    for item in (data or {}).get("query", {}).get("search", [])[:limit]:
        title = item.get("title", "")
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        out.append(Evidence(
            source=Source(url=page_url, title=f"Wikipedia: {title}", surface=Surface.WEB),
            passage=google_style_snippet(item.get("snippet", ""), query.split()),
            kind="search",
        ))
    return out


def arxiv(query: str, limit: int = 4) -> list[Evidence]:
    out: list[Evidence] = []
    url = ("http://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(f'all:"{query}"') + f"&max_results={limit}")
    raw = _get_text(url, timeout=settings.web_timeout_s + 10)
    root = ET.fromstring(raw)
    for entry in root.findall("a:entry", _NS)[:limit]:
        title = (entry.findtext("a:title", "", _NS) or "").strip().replace("\n", " ")
        link = entry.findtext("a:id", "", _NS) or ""
        summary = (entry.findtext("a:summary", "", _NS) or "").strip()
        out.append(Evidence(
            source=Source(url=link, title=f"arXiv: {title[:160]}", surface=Surface.WEB),
            passage=google_style_snippet(summary, query.split(), n=600),
            kind="search",
        ))
    return out


def hackernews(query: str, limit: int = 5) -> list[Evidence]:
    out: list[Evidence] = []
    url = ("https://hn.algolia.com/api/v1/search?query=" + urllib.parse.quote(query)
           + "&tags=story&hitsPerPage=" + str(limit))
    data = _get_json(url)
    for hit in (data or {}).get("hits", [])[:limit]:
        title = hit.get("title") or ""
        link = hit.get("url")
        points = hit.get("points") or 0
        hn_url = "https://news.ycombinator.com/item?id=" + str(hit.get("objectID", ""))
        story_text = (hit.get("story_text") or "")[:600]
        body = title
        if story_text:
            body = google_style_snippet(story_text, query.split(), n=500)
        if link:
            out.append(Evidence(
                source=Source(url=link, title=f"HN: {title}", surface=Surface.WEB),
                passage=f"[HN discussion {hn_url}, {points} points] {body}", kind="search",
            ))
        else:
            out.append(Evidence(
                source=Source(url=hn_url, title=f"HN: {title}", surface=Surface.WEB),
                passage=f"[HN thread, {points} points] {body}", kind="search",
            ))
    return out


def github(query: str, limit: int = 5) -> list[Evidence]:
    out: list[Evidence] = []
    # the search API 422s on natural-language punctuation; send plain keywords
    q = " ".join(re.findall(r"[a-z0-9]+", query.lower()))[:120]
    if not q:
        return out
    url = ("https://api.github.com/search/repositories?q=" + urllib.parse.quote(q)
           + "&per_page=" + str(limit))
    data = _get_json(url, headers={"Accept": "application/vnd.github+json"})
    for repo in (data or {}).get("items", [])[:limit]:
        full = repo.get("full_name", "")
        desc = repo.get("description") or ""
        stars = repo.get("stargazers_count", 0)
        out.append(Evidence(
            source=Source(url=repo.get("html_url", ""), title=f"GitHub: {full}", surface=Surface.WEB),
            passage=f"[{stars}★] {desc}".strip(), kind="search",
        ))
    return out


# ---------------------------------------------------------------------------
# Paid engines (optional)
# ---------------------------------------------------------------------------


def _tavily(query: str, limit: int = 6) -> list[Evidence]:
    if not settings.tavily_key:
        return []
    body = json.dumps({"api_key": settings.tavily_key, "query": query,
                       "search_depth": "basic", "max_results": limit}).encode()
    req = urllib.request.Request("https://api.tavily.com/search", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=settings.web_timeout_s + 10) as resp:
        data = json.loads(resp.read().decode())
    out = []
    for r in data.get("results", [])[:limit]:
        out.append(Evidence(
            source=Source(url=r.get("url", ""), title=r.get("title", ""), surface=Surface.WEB),
            passage=google_style_snippet(r.get("content", ""), query.split(), n=600),
            kind="search",
        ))
    return out


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class WebProvider(Provider):
    surface = Surface.WEB

    def __init__(self) -> None:
        super().__init__()
        self._engines = []
        if settings.tavily_key:
            self._engines.append(("tavily", _tavily))
        self._engines += [
            ("duckduckgo", duckduckgo),
            ("wikipedia", wikipedia),
            ("arxiv", arxiv),
            ("hackernews", hackernews),
            ("github", github),
        ]

    def search(self, query: str, limit: int = 8) -> list[Evidence]:
        seen: set[str] = set()
        out: list[Evidence] = []
        per_engine = max(1, min(4, max(2, limit // 4)))
        for name, engine in self._engines:
            try:
                results = engine(query, per_engine)
            except Exception as e:
                self.warnings.append(f"{name}: {e}")
                continue
            for ev in results:
                loc = ev.source.locator()
                key = urllib.parse.urlsplit(loc).netloc + urllib.parse.urlsplit(loc).path
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(ev)
            if len(out) >= limit:
                break
        return out[:limit]

    def fetch(self, source: Source) -> str | None:
        if not source.url:
            return None
        final_url, title, text = fetch_readable(source.url)
        if not text:
            self.warnings.append(f"could not fetch {source.url}")
            return None
        return text

    def describe(self) -> str:
        engine = "tavily+" if settings.tavily_key else ""
        return f"web provider [{engine}ddg, wikipedia, arxiv, hn, github]"
