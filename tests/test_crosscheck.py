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
    xc = claim("Veritas ships a nightly JSON batch process",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert pc.confidence == "high"
    assert pc.crosschecked is True
    assert result["corroborated"] == 1
    assert result["candidates"] == []


def test_reconcile_defers_near_identical_echo_with_extra_word():
    """Lexical corroboration needs a TOKEN-IDENTICAL statement: an echo that
    adds a qualifier differs in a material token and may disagree with the
    primary on it — it is deferred to the semantic/conflict path, never
    consumed as agreement (Codex round-7 P1)."""
    pc = claim("Veritas ships a nightly JSON batch process", "https://a.example/x")
    xc = claim("Veritas ships a nightly JSON batch process regularly",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 0
    assert pc.crosschecked is False
    assert pc.confidence == "medium"
    assert xc in result["candidates"]


def test_reconcile_defers_value_conflicting_match():
    """Same-polarity claims that disagree on a material VALUE ('March 2025'
    vs 'April 2025') share most tokens but are not the same fact: deferring
    lets detect_contradictions see the pair (Codex round-7 P1)."""
    pc = claim("Version 2.0 was released in March 2025", "https://a.example/x")
    xc = claim("Version 2.0 was released in April 2025",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 0
    assert pc.crosschecked is False
    assert pc.confidence == "medium"
    assert xc in result["candidates"]


def test_reconcile_defers_numeric_conflict():
    """Numbers are material: '20 percent' vs '30 percent' must not corroborate
    even though the wording is otherwise identical."""
    pc = claim("The treatment reduced mortality by 20 percent", "https://a.example/x")
    xc = claim("The treatment reduced mortality by 30 percent",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 0
    assert pc.crosschecked is False
    assert xc in result["candidates"]


def test_same_sources_do_not_bump_to_high():
    pc = claim("The release happened in March 2025", "https://a.example/x")
    xc = claim("The release happened in March 2025",
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


def test_reconcile_promotion_adopts_verified_evidence():
    """A matched donor that passed verification as supported may drive the
    promotion; its new-source evidence is adopted onto the primary so the
    report/ledger keep the independent source (Codex P1)."""
    pc = claim("Veritas ships a nightly JSON batch process", "https://a.example/x")
    xc = claim("Veritas ships a nightly JSON batch process",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 1
    assert pc.confidence == "high"
    assert result["candidates"] == []
    assert {e.source.locator() for e in pc.evidence} == {
        "https://a.example/x", "https://b.example/y"}


def test_reconcile_unsupported_lexical_echo_never_corroborates():
    """Only verified-SUPPORTED donors corroborate: a token-identical echo
    that failed verification (still default UNSUPPORTED) is consumed without
    marking the primary crosschecked or promoting it (Codex round-4 P1)."""
    pc = claim("The release happened in March 2025", "https://a.example/x")
    xc = Claim(id="x1", statement="The release happened in March 2025",
               evidence=[ev("https://b.example/y")],
               verdict=Verdict.UNSUPPORTED, confidence="unsupported")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 0
    assert pc.crosschecked is False
    assert pc.confidence == "medium"
    assert xc not in result["candidates"]  # consumed, no corroboration


def test_reconcile_negated_echo_defers_to_conflict_detector():
    """Any negation — 'not', bare 'no', 'can't' — makes the statement differ
    as a string, so a verified negated echo is deferred to the conflict
    detector, never consumed as agreement or promoted (Codex round-6/8)."""
    for negated in ("Version 2.0 was not released in March 2025",
                    "The drug has no effect",
                    "The drug can't cure the disease"):
        pc_stmt = {"Version 2.0 was not released in March 2025":
                   "Version 2.0 was released in March 2025",
                   "The drug has no effect": "The drug has an effect",
                   "The drug can't cure the disease":
                   "The drug cures the disease"}[negated]
        pc = claim(pc_stmt, "https://a.example/x")
        xc = claim(negated, "https://b.example/y", cid="x1")
        result = reconcile([pc], [xc])
        assert result["corroborated"] == 0
        assert pc.crosschecked is False
        assert pc.confidence == "medium"
        assert xc in result["candidates"]  # reaches detect_contradictions


def test_reconcile_role_reversal_defers():
    """Token-identical role reversals ('A beats B' vs 'B beats A') differ as
    strings: they must reach the conflict detector, never corroborate
    (Codex round-8 P1)."""
    pc = claim("Drug Alpha is more effective than Drug Beta", "https://a.example/x")
    xc = claim("Drug Beta is more effective than Drug Alpha",
               "https://b.example/y", cid="x1")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 0
    assert pc.crosschecked is False
    assert pc.confidence == "medium"
    assert xc in result["candidates"]


def test_reconcile_comparison_signs_are_preserved():
    """Semantic operators carry meaning: '>20' vs '<20' (or '+20' vs '-20')
    must NOT normalize to the same string — a verified opposite comparison
    defers to the conflict detector (Codex round-9 P1)."""
    for pc_stmt, xc_stmt in (("Output was >20 units", "Output was <20 units"),
                             ("Temperature was +20 degrees",
                              "Temperature was -20 degrees")):
        pc = claim(pc_stmt, "https://a.example/x")
        xc = claim(xc_stmt, "https://b.example/y", cid="x1")
        result = reconcile([pc], [xc])
        assert result["corroborated"] == 0
        assert pc.crosschecked is False
        assert xc in result["candidates"]


def test_reconcile_no_promotion_on_bundled_unsupporting_locator():
    """A cross claim verified on the SHARED source alone (its bundled new
    locator B was NOT found supporting) must not promote the primary via B,
    and B's evidence is not adopted (Codex round-6 P1)."""
    pc = claim("Widgets process JSON nightly", "https://a.example/x")
    xc = Claim(
        id="x1", statement="Widgets process JSON nightly",
        evidence=[ev("https://a.example/x"),
                  Evidence(source=Source(url="https://b.example/y",
                                         title="b"), passage="p",
                           supports=False)],
        verdict=Verdict.SUPPORTED, confidence="medium")
    result = reconcile([pc], [xc])
    assert result["corroborated"] == 1   # agreement on the shared fact
    assert pc.crosschecked is True
    assert pc.confidence == "medium"       # B did not support -> stays medium
    assert {e.source.locator() for e in pc.evidence} == {"https://a.example/x"}


def test_semantic_no_promotion_on_unsupporting_new_locator():
    """Semantic mirror: a matched donor whose only new locator was not found
    supporting by the verifier must not promote the primary (Codex round-6
    P1)."""
    llm = FakeLLM({CORROBORATOR_SYSTEM: json.dumps(
        {"same_fact_pairs": [[1, 1]]})})
    pc = claim("Sputnik was launched on 4 Oct 1957", "https://nasa.example/x")
    xc = Claim(
        id="x1", statement="The USSR orbited Sputnik in October 1957",
        evidence=[Evidence(source=Source(url="https://esa.example/y",
                                         title="esa"), passage="p",
                           supports=False)],
        verdict=Verdict.SUPPORTED, confidence="medium")
    flags, promos, _ = corroborate_from_semantic(llm, [pc], [xc])
    assert (flags, promos) == (1, 0)
    assert pc.crosschecked is True
    assert pc.confidence == "medium"
    assert {e.source.locator() for e in pc.evidence} == {"https://nasa.example/x"}


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


def test_crosschecked_primary_still_promotable_by_semantic_new_source():
    """A primary first crosschecked by a SAME-source lexical match stays
    eligible for the semantic pass (Codex P2): a verified paraphrase from a
    genuinely new source must promote it, not be appended as a duplicate."""
    pc = claim("The USSR orbited Sputnik in October 1957", "https://wiki.example/s")
    dup = claim("The USSR orbited Sputnik in October 1957", "https://wiki.example/s",
                cid="x1")  # same source -> crosschecked, stays medium
    result = reconcile([pc], [dup])
    assert result["corroborated"] == 1
    assert pc.crosschecked is True
    assert pc.confidence == "medium"
    # a verified paraphrase from a genuinely new source (survives reconcile
    # as a candidate, then passes verify_claim) must still promote the claim
    xc = claim("Sputnik 1 entered orbit in October 1957", "https://esa.example/y",
               cid="x2")
    llm = FakeLLM({CORROBORATOR_SYSTEM: json.dumps(
        {"same_fact_pairs": [[1, 1]]})})
    flags, promos, pairs = corroborate_from_semantic(llm, [pc], [xc])
    assert (flags, promos) == (1, 1)
    assert pc.confidence == "high"
    assert len(pairs) == 1
    assert {e.source.locator() for e in pc.evidence} == {
        "https://wiki.example/s", "https://esa.example/y"}


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
