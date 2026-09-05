"""Hermetic tests for the A5 relevance-sample collector."""

from __future__ import annotations

import json

from bench.collect_relevance import extract_sample


def _claim(subq: str, *sources: dict, verdict: str = "supported") -> dict:
    return {"id": "c", "statement": "stmt", "subquestion": subq,
            "evidence": [{"kind": "web", "passage": "p " + s["title"],
                          "retrieved_at": "", "source": s}
                         for s in sources],
            "verdict": verdict, "confidence": "medium",
            "crosschecked": False, "conflicts": [], "note": ""}


def _ledger(claims: list[dict]) -> dict:
    return {"query": "q", "created_at": "", "surfaces": ["web"],
            "confidence_counts": {}, "claims": claims, "gaps": [],
            "crosscheck": {}, "conflicts": []}


def _src(url: str, title: str) -> dict:
    return {"url": url, "path": None, "title": title, "surface": "web",
            "anchor": ""}


def test_sample_limits_and_dedupes():
    led = _ledger([
        _claim("subq one", _src("https://a.dev", "A"),
               _src("https://a.dev", "A dup"),        # dedupe by url
               _src("https://b.dev", "B")),
        _claim("subq one", _src("https://c.dev", "C")),  # extra for subq one
        _claim("subq two", _src("https://d.dev", "D"),
               _src("https://e.dev", "E")),
        _claim("subq three", _src("https://f.dev", "F")),  # beyond max_subq
    ])
    s = extract_sample(led, max_subquestions=2, sources_per_subquestion=2)
    subs = [e["subquestion"] for e in s]
    assert subs == ["subq one", "subq one", "subq two", "subq two"]
    assert [e["url"] for e in s] == ["https://a.dev", "https://b.dev",
                                     "https://d.dev", "https://e.dev"]
    assert all(e["passage"] for e in s)  # passage snippet included
    assert all(len(e["passage"]) <= 600 for e in s)


def test_sample_skips_claims_without_evidence_or_subquestion():
    led = _ledger([
        _claim("", _src("https://x.dev", "no subq")),          # empty subq
        {"id": "c", "statement": "s", "subquestion": "q1",
         "evidence": [], "verdict": "supported",
         "confidence": "medium", "crosschecked": False,
         "conflicts": [], "note": ""},                         # no evidence
        _claim("q1", _src("https://y.dev", "Y")),
    ])
    s = extract_sample(led)
    assert [e["url"] for e in s] == ["https://y.dev"]
    assert s[0]["subquestion"] == "q1"


def test_sample_empty_and_deterministic():
    assert extract_sample(_ledger([])) == []
    a = extract_sample(_ledger([
        _claim("q", _src("https://1.dev", "1"), _src("https://2.dev", "2"))]))
    b = extract_sample(_ledger([
        _claim("q", _src("https://2.dev", "2"), _src("https://1.dev", "1"))]))
    # sub-question order is claim order; within a sub-question it is the
    # first-seen evidence order, so both runs must agree on url order
    assert [e["url"] for e in a] == [e["url"] for e in b]
