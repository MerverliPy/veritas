"""Web connector URL contracts, tested without network access."""

from __future__ import annotations

from veritas.connectors import web


def test_arxiv_uses_https_endpoint(monkeypatch):
    seen: list[str] = []

    def fake_get_text(url, **kwargs):
        seen.append(url)
        return '<feed xmlns="http://www.w3.org/2005/Atom" />'

    monkeypatch.setattr(web, "_get_text", fake_get_text)
    web.arxiv("query", limit=1)

    assert seen and seen[0].startswith("https://export.arxiv.org/")
