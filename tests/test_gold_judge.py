"""Hermetic tests for the LLM gold judge (bench/judge.py) — FakeLLM only."""

from __future__ import annotations

from bench.judge import GOLD_JUDGE_SYSTEM, _label, make_claim_judge
from bench.score import compute_query_metrics
from veritas.llm import FakeLLM, LLMError

GOLD = {"query_id": "q", "class": "F", "source_landscape": "t",
        "expected_claims": [
            {"statement": "EternalBlue was released publicly in April 2017 "
                           "by the Shadow Brokers.",
             "gold_label": "correct", "confidence_class": "high"}]}


def _scripted_judge(label_fn):
    """FakeLLM whose judge responses depend on the claim text (callable)."""
    def respond(user: str) -> str:
        claim = user.rsplit("CLAIM TO JUDGE:\n", 1)[-1].split("\n\n", 1)[0]
        return '{"label": "%s", "reason": "test"}' % label_fn(claim)
    return FakeLLM({GOLD_JUDGE_SYSTEM: respond})


def test_label_parse_tolerates_prose():
    assert _label('{"label": "correct", "reason": "x"}') == "correct"
    assert _label('here: {"label": "off-topic", "reason": "y"} trailing') \
        == "off-topic"
    assert _label('{"label": "maybe", "reason": "x"}') is None
    assert _label("no json at all") is None


def test_judge_credits_correct_generative_claim():
    # Lexically far from gold (would be unmatched), but substantively correct
    claim = "A group called the Shadow Brokers published EternalBlue in 2017."
    judge = make_claim_judge(_scripted_judge(
        lambda c: "correct" if "Shadow Brokers" in c else "off-topic"))
    cb, state = judge
    assert cb(claim, GOLD["expected_claims"], "q") == "correct"
    assert state["fallbacks"] == 0


def test_judge_off_topic_never_credited():
    judge = make_claim_judge(_scripted_judge(lambda c: "off-topic"))
    cb, _state = judge
    assert cb("The port database covers 500 ports.",
              GOLD["expected_claims"]) == "off-topic"


def test_judge_outage_falls_back_to_lexical_and_counts():
    # FakeLLM raises LLMError for unknown system prefixes -> judge outage
    broken = FakeLLM({})
    cb, state = make_claim_judge(broken)
    stmt = "EternalBlue was released publicly in April 2017 by the " \
           "Shadow Brokers."  # identical to gold -> lexical correct
    assert cb(stmt, GOLD["expected_claims"]) == "correct"
    assert state["fallbacks"] == 1
    # lexically unmatched claim falls back to the FALLBACK_UNMATCHED sentinel
    from bench.score import FALLBACK_UNMATCHED
    assert cb("Something completely different.", GOLD["expected_claims"]) \
        == FALLBACK_UNMATCHED
    assert state["fallbacks"] == 2


def test_judge_precision_integration():
    """A lexically-unmatched but judge-correct claim scores full precision."""
    gold = {"query_id": "q", "class": "F", "expected_claims": [
        {"statement": "EternalBlue was released publicly in April 2017 by "
                       "the Shadow Brokers.",
         "gold_label": "correct", "confidence_class": "high"}]}
    claims = [{"id": "c1",
               "statement": "A group called the Shadow Brokers published "
                            "EternalBlue in 2017.",
               "subquestion": "", "evidence": [], "verdict": "supported",
               "confidence": "medium", "crosschecked": False,
               "conflicts": [], "note": ""}]
    ledger = {"query": "q", "created_at": "", "surfaces": ["web"],
              "confidence_counts": {}, "claims": claims, "gaps": [],
              "crosscheck": {}, "conflicts": []}
    judge = make_claim_judge(_scripted_judge(lambda c: "correct"))
    cb, _ = judge
    m = compute_query_metrics(ledger, gold, claim_judge=cb)
    assert m["precision_supported"] == 1.0     # judge credits what lexical can't
    assert m["judge_counts"] == {"correct": 1}
    # lexical-only baseline for the same claim: unmatched supported = 0 credit
    assert compute_query_metrics(ledger, gold)["precision_supported"] == 0.0


def test_invalid_label_counts_as_fallback():
    """Valid JSON with an unknown label (refusal/schema drift) must be
    counted as a fallback, not silently swapped to the lexical verdict."""
    llm = FakeLLM({GOLD_JUDGE_SYSTEM: lambda _u: '{"label": "unsure", "reason": "refusal"}'})
    cb, state = make_claim_judge(llm)
    stmt = "EternalBlue was released publicly in April 2017 by the " \
           "Shadow Brokers."
    assert cb(stmt, GOLD["expected_claims"]) == "correct"   # lexical fallback
    assert state["fallbacks"] == 1


def test_outage_fallback_stays_in_precision_denominator():
    """One judged-correct claim + one judge outage on an unmatched claim
    must report 50% precision, never 100% (fallback != judge off-topic)."""
    from bench.score import FALLBACK_UNMATCHED, compute_query_metrics
    gold = {"query_id": "q", "class": "F", "expected_claims": [
        {"statement": "EternalBlue was released publicly in April 2017 by "
                       "the Shadow Brokers.",
         "gold_label": "correct", "confidence_class": "high"}]}
    good = "EternalBlue was released publicly in April 2017 by the Shadow Brokers."
    fab = "The Moon is made of cheese."      # lexically unmatched + outage
    claims = [
        {"id": "c1", "statement": good, "subquestion": "", "evidence": [],
         "verdict": "supported", "confidence": "medium", "crosschecked": False,
         "conflicts": [], "note": ""},
        {"id": "c2", "statement": fab, "subquestion": "", "evidence": [],
         "verdict": "supported", "confidence": "medium", "crosschecked": False,
         "conflicts": [], "note": ""}]
    ledger = {"query": "q", "created_at": "", "surfaces": ["web"],
              "confidence_counts": {}, "claims": claims, "gaps": [],
              "crosscheck": {}, "conflicts": []}

    def respond(u):
        claim = u.rsplit("CLAIM TO JUDGE:\n", 1)[-1].split("\n\n", 1)[0]
        return '{"label": "correct", "reason": "t"}' if "Shadow Brokers" in claim \
            else '{"label": "oops"}'   # invalid label -> JudgeError -> fallback
    llm = FakeLLM({GOLD_JUDGE_SYSTEM: respond})
    cb, state = make_claim_judge(llm)
    m = compute_query_metrics(ledger, gold, claim_judge=cb)
    assert m["precision_supported"] == 0.5     # 1 correct of 2 placed
    assert m["precision_unscored_n"] == 0      # nothing excluded
    assert state["fallbacks"] == 1
    # sanity: the sentinel is what claim_judge returns on outage
    assert cb(fab, gold["expected_claims"]) == FALLBACK_UNMATCHED


def test_outage_fallback_in_calibration_buckets():
    """A high-confidence fabrication whose judge call fails must depress high
    reliability, not vanish from the bucket."""
    from bench.score import compute_query_metrics
    gold = {"query_id": "q", "class": "F", "expected_claims": [
        {"statement": "EternalBlue was released publicly in April 2017 by "
                       "the Shadow Brokers.",
         "gold_label": "correct", "confidence_class": "high"}]}
    good = "EternalBlue was released publicly in April 2017 by the Shadow Brokers."
    fab = "The Moon is made of cheese."
    claims = [
        {"id": "c1", "statement": good, "subquestion": "", "evidence": [],
         "verdict": "supported", "confidence": "high", "crosschecked": True,
         "conflicts": [], "note": ""},
        {"id": "c2", "statement": fab, "subquestion": "", "evidence": [],
         "verdict": "supported", "confidence": "high", "crosschecked": True,
         "conflicts": [], "note": ""}]
    ledger = {"query": "q", "created_at": "", "surfaces": ["web"],
              "confidence_counts": {}, "claims": claims, "gaps": [],
              "crosscheck": {}, "conflicts": []}

    def respond(u):
        claim = u.rsplit("CLAIM TO JUDGE:\n", 1)[-1].split("\n\n", 1)[0]
        return '{"label": "correct", "reason": "t"}' if "Shadow Brokers" in claim \
            else '{"label": "oops"}'
    cb, _state = make_claim_judge(FakeLLM({GOLD_JUDGE_SYSTEM: respond}))
    m = compute_query_metrics(ledger, gold, claim_judge=cb)
    assert m["precision_supported"] == 0.5
    assert m["reliability"]["high"] == 0.5    # 1 correct of 2 in the bucket
    assert m["reliability"]["high_n"] == 2
