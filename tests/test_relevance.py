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


def test_latest_run_dir_picks_by_mtime_not_name(tmp_path):
    from bench.collect_relevance import _latest_run_dir
    # name order says nocc-pilot is "newer", but det-2's scorecard is newer
    for name in ("cc-20260905-100000", "nocc-pilot", "det-2"):
        d = tmp_path / name
        d.mkdir()
        (d / "scorecard.json").write_text("{}")
    import os
    old = tmp_path / "nocc-pilot" / "scorecard.json"
    new = tmp_path / "det-2" / "scorecard.json"
    os.utime(old, (1_700_000_000, 1_700_000_000))
    os.utime(new, (1_800_000_000, 1_800_000_000))
    assert _latest_run_dir(tmp_path).name == "det-2"


def test_judgements_action_requires_force_on_sample_change():
    from bench.collect_relevance import judgements_action
    s1 = [{"url": "https://a.dev"}, {"url": "https://b.dev"}]
    s2 = [{"url": "https://a.dev"}, {"url": "https://c.dev"}]
    assert judgements_action(None, s1, False, False) == "write"
    assert judgements_action(s1, s1, True, False) == "preserve"
    assert judgements_action(s1, s2, True, False) == "abort"
    assert judgements_action(s1, s2, True, True) == "write"


def test_relevance_binding_error(tmp_path):
    from bench.run_benchmark import relevance_binding_error, read_keyed_relevance
    import hashlib as _h
    sample = tmp_path / "relevance-sample.json"
    sample_text = json.dumps([{"url": "a"}, {"url": "b"}])
    sample.write_text(sample_text)
    sha = _h.sha1(sample_text.encode()).hexdigest()[:16]
    assert relevance_binding_error(sha, sample) is None
    assert "do not match" in relevance_binding_error("deadbeef", sample) \
        or "does not" in relevance_binding_error("deadbeef", sample)
    assert "not bound" in relevance_binding_error(None, sample)
    assert relevance_binding_error(sha, tmp_path / "missing.json") is not None
    # keyed reader: valid object, plain list, and garbage
    (tmp_path / "k.json").write_text(json.dumps(
        {"sample_sha": sha, "judgements": [0, 1]}))
    vals, got_sha = read_keyed_relevance(tmp_path / "k.json")
    assert vals == [0, 1] and got_sha == sha
    (tmp_path / "p.json").write_text(json.dumps([1, 1]))
    vals, got_sha = read_keyed_relevance(tmp_path / "p.json")
    assert vals == [1, 1] and got_sha is None
    (tmp_path / "bad.json").write_text(json.dumps([0, 2]))
    import pytest as _pytest
    with _pytest.raises(ValueError):
        read_keyed_relevance(tmp_path / "bad.json")
