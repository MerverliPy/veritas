"""Cross-check: corroboration, candidate selection, contradiction pairs."""

from __future__ import annotations

import json

from veritas import Claim, Evidence, Source, Surface, Verdict
from veritas.llm import FakeLLM
from veritas.pipeline.crosscheck import detect_contradictions, reconcile
from veritas.pipeline.prompts import CONFLICT_DETECTOR_SYSTEM


def ev(url: str) -> Evidence:
    return Evidence(source=Source(url=url, title=url), passage="p")


def claim(statement: str, url: str, verdict: Verdict = Verdict.SUPPORTED,
          confidence: str = "medium", cid: str = "c1") -> Claim:
    return Claim(id=cid, statement=statement, evidence=[ev(url)],
                 verdict=verdict, confidence=confidence)


# ----------------------------------------------------------------- reconcile


def test_independent_corroboration_bumps_to_high():
    pc = claim("Veritas ships a nightly JSON batch process", "https://a.example/x")
    xc = claim("Veritas ships a nightly JSON batch process regularly",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert pc.confidence == "high"
    assert pc.crosschecked is True
    assert result["corroborated"] == 1
    assert result["candidates"] == []


def test_same_sources_do_not_bump_to_high():
    pc = claim("The release happened in March 2025", "https://a.example/x")
    xc = claim("The release happened in March 2025 per changelog",
               "https://a.example/x", cid="x1")  # same locator
    reconcile([pc], [xc])
    assert pc.confidence == "medium"
    assert pc.crosschecked is True


def test_unsupported_primary_is_never_corroborated():
    pc = claim("Speculative statement", "https://a.example/x")
    pc.verdict = Verdict.UNSUPPORTED
    pc.confidence = "unsupported"
    xc = claim("Speculative statement is indeed correct",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 0
    assert xc in result["candidates"]  # independent claim, not 'agreement'


def test_unmatched_cross_claim_becomes_candidate_not_auto_claim():
    pc = claim("Widgets process JSON", "https://a.example/x")
    xc = claim("Llamas are vegetarian herd animals", "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert pc.crosschecked is False
    assert pc.confidence == "medium"      # untouched
    assert result["candidates"] == [xc]   # must pass verification later
    # ...and a candidate is never already marked as an established claim
    assert xc.confidence != "high"


def test_antonym_contradiction_is_not_falsely_corroborated():
    # "unmaintained" vs "actively maintained" share no 4+ char tokens except
    # 'service' — must NOT be treated as agreement.
    pc = claim("The service is unmaintained", "https://a.example/x")
    xc = claim("The service is actively maintained by the team",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert pc.crosschecked is False
    assert xc in result["candidates"]


# -------------------------------------------------------- detect_contradictions


def test_detector_reports_genuine_opposites_and_validates_indices():
    claims = [
        claim("The service is unmaintained", "https://a.example/x"),
        claim("The service is actively maintained", "https://b.example/y"),
        claim("The sky is blue", "https://c.example/z"),
    ]
    llm = FakeLLM({CONFLICT_DETECTOR_SYSTEM: json.dumps(
        {"contradicting_pairs": [[1, 2], [99, 3], [1, 1], [2, 1], "junk", [3]]})})
    pairs = detect_contradictions(llm, claims)
    assert pairs == [(1, 2)]  # invalid/duplicate/self pairs dropped


def test_detector_empty_when_under_two_claims():
    llm = FakeLLM({})
    assert detect_contradictions(llm, [claim("only one", "https://a")]) == []
    assert detect_contradictions(llm, []) == []


def test_detector_pairs_sorted_and_deduped():
    claims = [
        claim("A is 10", "https://a"),
        claim("A is 20", "https://b"),
        claim("B is red", "https://c"),
        claim("B is blue", "https://d"),
    ]
    llm = FakeLLM({CONFLICT_DETECTOR_SYSTEM: json.dumps(
        {"contradicting_pairs": [[2, 1], [4, 3], [2, 1]]})})
    assert detect_contradictions(llm, claims) == [(1, 2), (3, 4)]
