"""Unit tests for the claim extractor's evidence-binding enforcement."""

from __future__ import annotations

import json

from veritas import Evidence, FakeLLM, Source
from veritas.pipeline.claims import extract_claims
from veritas.pipeline.prompts import CLAIMS_SYSTEM


def make_evidence(n: int = 3) -> list[Evidence]:
    return [Evidence(source=Source(url=f"https://example.com/{i}", title=f"s{i}"),
                     passage=f"passage number {i}") for i in range(1, n + 1)]


def llm_responding(raw: str) -> FakeLLM:
    return FakeLLM({CLAIMS_SYSTEM: raw})


def test_claim_keeps_bound_evidence_objects():
    evidence = make_evidence()
    raw = json.dumps({"claims": [
        {"statement": "First claim", "evidence_idx": [1, 3]},
    ], "noted_gaps": []})
    claims, gaps = extract_claims(llm_responding(raw), "q", evidence)
    assert len(claims) == 1
    c = claims[0]
    assert c.statement == "First claim"
    assert [e.source.locator() for e in c.evidence] == [
        "https://example.com/1", "https://example.com/3"]
    assert c.subquestion == "q"
    assert c.verdict.value == "unsupported"  # verification decides later
    assert gaps == []


def test_claim_without_evidence_is_dropped_to_gap():
    evidence = make_evidence()
    raw = json.dumps({"claims": [
        {"statement": "No evidence cited", "evidence_idx": []},
        {"statement": "Missing key", },
    ], "noted_gaps": ["a real question gap"]})
    claims, gaps = extract_claims(llm_responding(raw), "q", evidence)
    assert claims == []
    assert len(gaps) == 3  # both dropped claims + the noted gap


def test_out_of_range_index_dropped_but_valid_ones_kept():
    evidence = make_evidence(2)
    raw = json.dumps({"claims": [
        {"statement": "Partial evidence", "evidence_idx": [1, 99]},
    ], "noted_gaps": []})
    claims, gaps = extract_claims(llm_responding(raw), "q", evidence)
    assert len(claims) == 1
    assert len(claims[0].evidence) == 1
    assert claims[0].evidence[0].source.url == "https://example.com/1"
    assert len(gaps) == 0  # valid binding exists -> not dropped


def test_non_numeric_index_dropped():
    evidence = make_evidence()
    raw = json.dumps({"claims": [
        {"statement": "Bad index type", "evidence_idx": ["x"]},
    ], "noted_gaps": []})
    claims, gaps = extract_claims(llm_responding(raw), "q", evidence)
    assert claims == []
    assert any("dropped" in g for g in gaps)


def test_duplicate_statements_within_one_call_merged():
    evidence = make_evidence()
    raw = json.dumps({"claims": [
        {"statement": "Same thing", "evidence_idx": [1]},
        {"statement": "same thing", "evidence_idx": [2]},
    ], "noted_gaps": []})
    claims, _ = extract_claims(llm_responding(raw), "q", evidence)
    assert len(claims) == 1


def test_evidence_too_thin_for_model_reported_as_gap_when_nothing_else():
    # model returns garbage that is not a claim list at all -> resilient
    raw = json.dumps({"noted_gaps": ["sources too thin"], "claims": "n/a"})
    claims, gaps = extract_claims(llm_responding(raw), "q", make_evidence())
    assert claims == []
    assert gaps == ["sources too thin"]
