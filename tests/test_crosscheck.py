"""Cross-check: corroboration, candidate selection, contradiction pairs."""

from __future__ import annotations

import json

from veritas import Claim, Evidence, Source, Surface, Verdict
from veritas.llm import FakeLLM
from veritas.pipeline.crosscheck import (
    corroborate_from_semantic,
    detect_contradictions,
    reconcile,
    semantic_corroborate,
)
from veritas.pipeline.prompts import CONFLICT_DETECTOR_SYSTEM, CORROBORATOR_SYSTEM


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


def test_reconcile_subset_sources_do_not_promote():
    """Deterministic mirror of the semantic rule: primary {A,B} agreed with
    by a cross claim citing only {A} is subset agreement, not independent
    evidence — must not promote (Codex P1)."""
    pc = Claim(
        id="c1", statement="The USSR orbited Sputnik in October 1957",
        evidence=[ev("https://wiki.example/s"), ev("https://nasa.example/x")],
        verdict=Verdict.SUPPORTED, confidence="medium")
    xc = Claim(
        id="x1", statement="The USSR orbited Sputnik in October 1957",
        evidence=[ev("https://wiki.example/s")],
        verdict=Verdict.SUPPORTED, confidence="medium")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 1
    assert pc.confidence == "medium"
    assert pc.crosschecked is True


def test_reconcile_promotion_keeps_corroborating_source():
    """A matched cross claim is consumed (never appended), so a promotion to
    high must keep the corroborating claim's new source on the primary
    (report/ledger traceability)."""
    pc = claim("Veritas ships a nightly JSON batch process", "https://a.example/x")
    xc = claim("Veritas ships a nightly JSON batch process regularly",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 1
    assert pc.confidence == "high"
    assert result["candidates"] == []
    assert {e.source.locator() for e in pc.evidence} == {
        "https://a.example/x", "https://b.example/y"}


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


# ---------------------------------------------------- semantic corroboration


def test_semantic_corroboration_matches_paraphrased_fact():
    # generative phrasing restates the same fact in different words — the
    # token matcher misses it (full-1 A2 finding), the semantic pass must not
    pc = claim("The Soviet Union launched Sputnik 1 on October 4, 1957",
               "https://nasa.example/x")
    xc = claim("Sputnik 1 was put into orbit by the USSR in October 1957",
               "https://esa.example/y", cid="x1")
    llm = FakeLLM({CORROBORATOR_SYSTEM: json.dumps(
        {"same_fact_pairs": [[1, 1]]})})
    assert semantic_corroborate(llm, [pc], [xc]) == [(pc, xc)]


def test_semantic_corroboration_validates_indices_and_dedupes():
    pc1 = claim("A is ten", "https://a")
    pc2 = claim("B is red", "https://b")
    xc1 = claim("A equals ten exactly", "https://c", cid="x1")
    xc2 = claim("B is colored red", "https://d", cid="x2")
    llm = FakeLLM({CORROBORATOR_SYSTEM: json.dumps({
        "same_fact_pairs": [[1, 1], [2, 2], [1, 1], [99, 1], [1, 99],
                            "junk", [1], [3, 3], [1, 2]]})})
    # duplicate/self/out-of-range/double-use pairs all dropped
    assert semantic_corroborate(llm, [pc1, pc2], [xc1, xc2]) == [
        (pc1, xc1), (pc2, xc2)]


def test_semantic_corroboration_empty_inputs_or_llm_outage():
    pc = claim("A is ten", "https://a")
    xc = claim("A equals ten", "https://b", cid="x1")
    assert semantic_corroborate(FakeLLM({}), [pc], [xc]) == []  # no key: soft []
    assert semantic_corroborate(FakeLLM({}), [], [xc]) == []
    assert semantic_corroborate(FakeLLM({}), [pc], []) == []


def test_corroborate_from_semantic_promotes_only_on_different_sources():
    llm = FakeLLM({CORROBORATOR_SYSTEM: json.dumps(
        {"same_fact_pairs": [[1, 1]]})})
    # paraphrase citing a DIFFERENT source -> crosschecked + high
    pc = claim("Sputnik was launched by the USSR on 4 Oct 1957",
               "https://nasa.example/x")
    xc = claim("The USSR orbited Sputnik in October 1957",
               "https://esa.example/y", cid="x1")
    flags, promos, pairs = corroborate_from_semantic(llm, [pc], [xc])
    assert (flags, promos) == (1, 1)
    assert pairs == [(pc, xc)]
    assert pc.crosschecked is True
    assert pc.confidence == "high"
    # paraphrase citing the SAME source -> crosschecked but stays medium
    pc2 = claim("Sputnik was launched by the USSR on 4 Oct 1957",
                "https://wiki.example/s")
    xc2 = claim("The USSR orbited Sputnik in October 1957",
                "https://wiki.example/s", cid="x1")
    flags2, promos2, _ = corroborate_from_semantic(llm, [pc2], [xc2])
    assert (flags2, promos2) == (1, 0)
    assert pc2.crosschecked is True
    assert pc2.confidence == "medium"
    # cross citing only a SUBSET of the primary's sources (primary {A,B},
    # cross {A}) adds no new locator — set inequality alone must not promote
    pc3 = Claim(
        id="c1", statement="The USSR orbited Sputnik in October 1957",
        evidence=[ev("https://wiki.example/s"), ev("https://nasa.example/x")],
        verdict=Verdict.SUPPORTED, confidence="medium")
    xc3 = Claim(
        id="x1", statement="The USSR orbited Sputnik in October 1957",
        evidence=[ev("https://wiki.example/s")],
        verdict=Verdict.SUPPORTED, confidence="medium")
    flags3, promos3, _ = corroborate_from_semantic(llm, [pc3], [xc3])
    assert (flags3, promos3) == (1, 0)
    assert pc3.crosschecked is True
    assert pc3.confidence == "medium"


def test_semantic_consumed_evidence_retained_on_primary():
    """Matched cross claims are consumed (never appended), so their evidence
    must survive on the claim that now rests on it — the report/ledger must
    show the independent source that justified the promotion (Codex P1)."""
    llm = FakeLLM({CORROBORATOR_SYSTEM: json.dumps(
        {"same_fact_pairs": [[1, 1]]})})
    pc = claim("Sputnik was launched on 4 Oct 1957", "https://nasa.example/x")
    xc = claim("The USSR orbited Sputnik in October 1957",
               "https://esa.example/y", cid="x1")
    flags, promos, _ = corroborate_from_semantic(llm, [pc], [xc])
    assert (flags, promos) == (1, 1)
    assert pc.confidence == "high"
    assert {e.source.locator() for e in pc.evidence} == {
        "https://nasa.example/x", "https://esa.example/y"}
    # same-source agreement adds nothing and must not duplicate evidence
    pc2 = claim("Sputnik was launched on 4 Oct 1957", "https://wiki.example/s")
    xc2 = claim("The USSR orbited Sputnik in October 1957",
                "https://wiki.example/s", cid="x1")
    flags2, promos2, _ = corroborate_from_semantic(llm, [pc2], [xc2])
    assert (flags2, promos2) == (1, 0)
    assert {e.source.locator() for e in pc2.evidence} == {"https://wiki.example/s"}


def test_several_cross_claims_corroborate_one_primary():
    """Several independent claims may restate the same primary fact; the old
    primary single-use made promotion order-dependent (a same-source match
    first left the claim medium and pushed the independent paraphrase into
    the appended list). Counts stay per primary claim."""
    llm = FakeLLM({CORROBORATOR_SYSTEM: json.dumps(
        {"same_fact_pairs": [[1, 1], [2, 1]]})})
    pc = claim("The USSR orbited Sputnik in October 1957", "https://wiki.example/s")
    xc1 = claim("The USSR orbited Sputnik in Oct 1957", "https://wiki.example/s",
                cid="x1")   # same source, returned first
    xc2 = claim("Sputnik 1 entered orbit in October 1957", "https://esa.example/y",
                cid="x2")   # genuinely new source
    flags, promos, pairs = corroborate_from_semantic(llm, [pc], [xc1, xc2])
    assert (flags, promos) == (1, 1)   # one claim corroborated, one promotion
    assert len(pairs) == 2             # both cross claims consumed, not appended
    assert pc.confidence == "high"
    assert {e.source.locator() for e in pc.evidence} == {
        "https://wiki.example/s", "https://esa.example/y"}


def test_semantic_corroborate_cross_index_is_single_use_only():
    """A cross claim matches at most one primary, but one primary may accept
    several cross claims — single-use is enforced on the cross index only."""
    pc1 = claim("A is ten", "https://a")
    pc2 = claim("B is red", "https://b")
    xc1 = claim("A equals ten exactly", "https://c", cid="x1")
    xc2 = claim("A is ten", "https://d", cid="x2")
    llm = FakeLLM({CORROBORATOR_SYSTEM: json.dumps({
        "same_fact_pairs": [[1, 1], [1, 2], [2, 1]]})})
    # [1,2] dropped (xc1 may not match two primaries); [2,1] allowed
    assert semantic_corroborate(llm, [pc1, pc2], [xc1, xc2]) == [
        (pc1, xc1), (pc1, xc2)]


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
