"""Hermetic tests for bench/score.py + driver guard helpers — no network, no LLM."""

from __future__ import annotations

import pytest

from bench.run_benchmark import parse_relevance, select_queries
from bench.score import (
    compute_query_metrics,
    est_cost_usd,
    flip_rate,
    gates,
    gold_verdict,
)

# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def claim(statement: str, *, verdict: str = "supported",
          confidence: str = "medium", conflicts: list | None = None,
          crosschecked: bool = False) -> dict:
    return {"id": f"c{hash(statement) & 0xffff}",
            "statement": statement,
            "subquestion": "sq",
            "evidence": [],
            "verdict": verdict,
            "confidence": confidence,
            "crosschecked": crosschecked,
            "conflicts": conflicts or [],
            "note": ""}


def ledger(claims: list[dict], *, conflicts: list | None = None) -> dict:
    return {"query": "q", "created_at": "2026-01-01T00:00:00Z",
            "surfaces": ["web"], "confidence_counts": {},
            "claims": claims, "gaps": [],
            "crosscheck": {},
            "conflicts": conflicts or []}


def gold(cls: str, expected: list[dict]) -> dict:
    return {"query_id": "q", "class": cls,
            "source_landscape": "test", "expected_claims": expected}


def exp(statement: str, label: str = "correct",
        confidence_class: str = "medium") -> dict:
    return {"statement": statement, "gold_label": label,
            "confidence_class": confidence_class, "note": ""}


# --------------------------------------------------------------------------
# cost + matching
# --------------------------------------------------------------------------

def test_est_cost_usd_formula():
    # 4000 chars / 4 chars-per-token = 1000 tokens * 0.55 / 1e6
    assert abs(est_cost_usd("x" * 4000) - 1000 * 0.55 / 1_000_000) < 1e-12
    assert est_cost_usd("") == 0.0


def test_gold_verdict_matching():
    g = gold("F", [exp("The WannaCry ransomware spread via the EternalBlue "
                       "exploit of SMBv1 in May 2017."),
                   exp("Penguins breed on the Antarctic continent.", "incorrect")])
    # identical statement matches correct
    assert gold_verdict("The WannaCry ransomware spread via the EternalBlue "
                        "exploit of SMBv1 in May 2017.", g["expected_claims"]) \
        == "correct"
    # near-duplicate (same significant tokens, different wording) matches
    assert gold_verdict("WannaCry spread quickly in May 2017 using the "
                        "EternalBlue exploit of SMBv1 ransomware.",
                        g["expected_claims"]) == "correct"
    # unrelated statement -> unmatched (never scored as correct)
    assert gold_verdict("Cats are solitary nocturnal hunters.",
                        g["expected_claims"]) == "unmatched"
    # falsehood asserted -> incorrect
    assert gold_verdict("Penguins breed on the Antarctic continent.",
                        g["expected_claims"]) == "incorrect"


def test_flip_rate_detects_verdict_reversal():
    stmt = "The WannaCry worm halted after a researcher registered the kill-switch."
    a = ledger([claim(stmt, verdict="supported", confidence="medium")])
    b = ledger([claim(stmt, verdict="contradicted", confidence="low")])
    assert flip_rate(a, b) == 1.0
    b2 = ledger([claim(stmt, verdict="supported", confidence="medium")])
    assert flip_rate(a, b2) == 0.0
    assert flip_rate(ledger([]), b) is None  # no matched pairs


# --------------------------------------------------------------------------
# per-query metrics
# --------------------------------------------------------------------------

def test_f_class_precision_and_recall():
    expected = [
        exp("EternalBlue exploited a Windows SMBv1 vulnerability."),
        exp("The WannaCry worm spread to roughly one hundred fifty countries."),
        exp("Registering the kill switch domain halted the ransomware outbreak."),
    ]
    g = gold("F", expected)
    claims = [claim("EternalBlue exploited a Windows SMBv1 vulnerability."),
              claim("The WannaCry worm spread to roughly one hundred fifty countries."),
              claim("Penguins breed on the Antarctic continent during winter.")]
    m = compute_query_metrics(ledger(claims), g)
    assert m["class"] == "F"
    assert m["precision_supported"] == 2 / 3   # unmatched supported = not correct
    assert m["recall_gold"] == 2 / 3           # 2 of 3 gold-correct covered


def test_u_class_honest_failure():
    g = gold("U", [])
    honest = ledger([claim("Maybe x.", verdict="unsupported", confidence="unsupported"),
                     claim("Thin y.", verdict="partial", confidence="low")])
    m = compute_query_metrics(honest, g)
    assert m["unsupported_share_U"] == 1.0
    assert m["fabrication_U"] == 0
    fabricating = ledger([claim("Z definitely happened.", verdict="supported",
                                confidence="high")])
    m2 = compute_query_metrics(fabricating, g)
    assert m2["unsupported_share_U"] == 0.0
    assert m2["fabrication_U"] == 1


def test_d_class_contradiction_fires():
    g = gold("D", [exp("Side A.", "contested"), exp("Side B.", "contested")])
    fired = ledger([claim("Side A.", verdict="contradicted", confidence="low",
                          conflicts=["Side B."])])
    assert compute_query_metrics(fired, g)["contradiction_fires_D"] == 1
    quiet = ledger([claim("Side A.", verdict="supported", confidence="medium")])
    assert compute_query_metrics(quiet, g)["contradiction_fires_D"] == 0


def test_structure_only_without_gold():
    m = compute_query_metrics(ledger([claim("Any statement.")]), None)
    assert m["class"] is None
    assert m["n_claims"] == 1
    assert m["precision_supported"] is None   # never scored without gold


# --------------------------------------------------------------------------
# gates — all-pass happy path mirrors a realistic full run
# --------------------------------------------------------------------------

def _all_pass_query_metrics():
    """One F, one U, one D ledger whose combined metrics satisfy A1–A4."""
    f_expected = ([exp(f"Solid fact {i} is well documented.") for i in range(11)]
                  + [exp(f"False claim {i} is not true.", "incorrect")
                     for i in range(4)]
                  + [exp("A twelfth correct fact no claim covers.")])
    f_claims = []
    # supported-correct: 3 high, 6 medium, 2 low
    for i in range(3):
        f_claims.append(claim(f"Solid fact {i} is well documented.",
                              confidence="high"))
    for i in range(3, 9):
        f_claims.append(claim(f"Solid fact {i} is well documented.",
                              confidence="medium"))
    for i in range(9, 11):
        f_claims.append(claim(f"Solid fact {i} is well documented.",
                              confidence="low"))
    # partial-incorrect: 2 medium, 2 low — depress calibration, not precision
    for i in range(2):
        f_claims.append(claim(f"False claim {i} is not true.",
                              verdict="partial", confidence="medium"))
    for i in range(2, 4):
        f_claims.append(claim(f"False claim {i} is not true.",
                              verdict="partial", confidence="low"))
    f = compute_query_metrics(ledger(f_claims), gold("F", f_expected))
    u = compute_query_metrics(
        ledger([claim("Scant trace of an answer.", verdict="unsupported",
                      confidence="unsupported"),
                claim("Weak hint only.", verdict="partial", confidence="low")]),
        gold("U", []))
    d = compute_query_metrics(
        ledger([claim("Side A is documented.", verdict="contradicted",
                      confidence="low", conflicts=["Side B is documented."])],
               conflicts=[{"a": "Side A", "b": "Side B"}]),
        gold("D", [exp("Side A is documented.", "contested"),
                   exp("Side B is documented.", "contested")]))
    return [f, u, d]


def test_recall_excludes_rejected_claims():
    """A gold-correct statement found but rejected by verification
    (contradicted/unsupported) is not coverage — the pipeline did not
    assert it."""
    stmt = "The worm stopped after the kill switch domain was registered."
    g = gold("F", [exp(stmt)])
    rejected = ledger([claim(stmt, verdict="contradicted", confidence="low")])
    m = compute_query_metrics(rejected, g)
    assert m["recall_gold"] == 0.0
    asserted = ledger([claim(stmt, verdict="supported", confidence="medium")])
    assert compute_query_metrics(asserted, g)["recall_gold"] == 1.0
    # partial (asserted-with-correction) also counts as coverage
    partial = ledger([claim(stmt, verdict="partial", confidence="low")])
    assert compute_query_metrics(partial, g)["recall_gold"] == 1.0


def test_gates_all_pass():
    qm = _all_pass_query_metrics()
    # paired arm: same precision, zero high claims -> cross-check benefit
    f_nocc_claims = [claim(f"Solid fact {i} is well documented.",
                           confidence="medium") for i in range(11)]
    f_nocc = compute_query_metrics(ledger(f_nocc_claims),
                                   gold("F", [exp(f"Solid fact {i} "
                                                   "is well documented.")
                                               for i in range(11)]))
    qm_nocc = [f_nocc, qm[1], qm[2]]
    g = gates(qm, q_metrics_nocc=qm_nocc,
              relevance_judgements=[1, 1, 1, 0],
              flip_pairs=[(ledger([claim("First stable claim statement here.",
                                          verdict="supported"),
                                   claim("Second stable claim statement here.",
                                          verdict="partial")]),
                           ledger([claim("First stable claim statement here.",
                                          verdict="supported"),
                                   claim("Second stable claim statement here.",
                                          verdict="partial")]))])
    assert g["A1_precision_fabrication"]["ok"] is True   # precision 1.0, fab 0
    assert g["A2_calibration"]["ok"] is True             # 1.0 / 0.75 / 0.5
    assert g["A3_honest_failure_U"]["ok"] is True        # share 1.0
    assert g["A4_crosscheck_benefit"]["ok"] is True      # fires + delta
    assert g["A5_relevance"]["ok"] is True               # median 1.0
    assert g["A6_determinism"]["ok"] is True             # flip rate 0
    assert g["A1_precision_fabrication"]["value"]["precision_supported"] == 1.0
    rel = g["A2_calibration"]["value"]["reliability"]
    assert rel["high"] == 1.0 and abs(rel["medium"] - 0.75) < 1e-9 \
        and rel["low"] == 0.5
    v4 = g["A4_crosscheck_benefit"]["value"]
    assert v4["all_D_fired"] is True
    assert v4["high_share_with"] > v4["high_share_without"]
    assert v4["precision_with"] >= v4["precision_without"]


def test_gates_detect_failures_and_na():
    # A1 fail: supported falsehood on F (with U data present so A1 assessable)
    f_bad = compute_query_metrics(
        ledger([claim("Supported fact zero is true."),
                claim("Supported falsehood one is true.")]),
        gold("F", [exp("Supported fact zero is true."),
                   exp("Supported falsehood one is true.", "incorrect")]))
    u = compute_query_metrics(
        ledger([claim("Nothing found.", verdict="unsupported",
                      confidence="unsupported")]), gold("U", []))
    g1 = gates([f_bad, u])
    assert g1["A1_precision_fabrication"]["ok"] is False  # precision 0.5
    # A5 fail: poor relevance judgements
    g2 = gates(_all_pass_query_metrics(), relevance_judgements=[0, 0])
    assert g2["A5_relevance"]["ok"] is False
    # A6 fail: verdict reversal across reruns
    g3 = gates(_all_pass_query_metrics(),
               flip_pairs=[(ledger([claim("Kill-switch stopped it.",
                                          verdict="supported")]),
                            ledger([claim("Kill-switch stopped it.",
                                          verdict="contradicted")]))])
    assert g3["A6_determinism"]["ok"] is False
    # Not-applicable: no data for a gate -> None, never FAIL
    g4 = gates([_all_pass_query_metrics()[0]])
    assert g4["A3_honest_failure_U"]["ok"] is None
    assert g4["A5_relevance"]["ok"] is None
    assert g4["A6_determinism"]["ok"] is None
    # A4 without the paired arm is n/a, never PASS on a mainline fire alone
    assert gates(_all_pass_query_metrics())["A4_crosscheck_benefit"]["ok"] \
        is None


# --------------------------------------------------------------------------
# driver guard helpers (run_benchmark.py)
# --------------------------------------------------------------------------

def _qs(ids):
    return [{"id": i, "class": "F", "query": "q"} for i in ids]


def test_select_queries_reports_unknown_ids():
    qs = _qs(["a", "b", "c"])
    chosen, unknown = select_queries(qs, "a, c")
    assert [q["id"] for q in chosen] == ["a", "c"] and unknown == []
    _, unknown = select_queries(qs, "a,nope")
    assert unknown == ["nope"]          # caller must reject, never silently drop
    chosen, unknown = select_queries(qs, "nope")
    assert chosen == [] and unknown == ["nope"]


def test_parse_relevance_accepts_binary_only(tmp_path):
    def write(v):
        f = tmp_path / "rel.json"
        f.write_text(__import__("json").dumps(v))
        return f

    assert parse_relevance(write([0, 1, 1])) == [0, 1, 1]
    assert parse_relevance(write([1])) == [1]
    for bad in ([0, 2], [0, -1], [0, "1"], [1.5, 0], [True, 0], "0,1"):
        with pytest.raises(ValueError):
            parse_relevance(write(bad))


def _gold_file(gold_dir, qid, cls="F", *, query_id=None, expected=None,
               malformed=False):
    g = {"query_id": query_id or qid, "class": cls,
         "source_landscape": "t",
         "expected_claims": expected or [{"statement": "Fact one is true.",
                                          "gold_label": "correct",
                                          "confidence_class": "medium"}]}
    text = "not json{{{" if malformed else __import__("json").dumps(g)
    (gold_dir / f"{qid}.json").write_text(text)


def test_preflight_errors_detects_bad_gold_before_running(tmp_path):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    queries = [{"id": "f1", "class": "F", "query": "q1"},
               {"id": "u1", "class": "U", "query": "q2"},
               {"id": "d1", "class": "D", "query": "q3"}]
    from bench.run_benchmark import preflight_errors
    assert preflight_errors(queries, gold_dir) == []   # missing sheets allowed
    _gold_file(gold_dir, "f1")                          # good sheet -> ok
    assert preflight_errors(queries, gold_dir) == []
    _gold_file(gold_dir, "u1", malformed=True)          # unreadable JSON
    errs = preflight_errors(queries, gold_dir)
    assert any("unreadable JSON" in e for e in errs)
    _gold_file(gold_dir, "u1", query_id="other")        # id mismatch
    errs = preflight_errors(queries, gold_dir)
    assert any("query_id" in e for e in errs)
    (gold_dir / "d1.json").write_text(__import__("json").dumps(
        {"query_id": "d1", "class": "D", "expected_claims": [
            {"statement": "", "gold_label": "correct",
             "confidence_class": "medium"}]}))
    errs = preflight_errors(queries, gold_dir)
    assert any("statement" in e for e in errs)
    # duplicate ids + bad class caught too
    dup = [{"id": "f1", "class": "F", "query": "q"}] * 2
    assert any("duplicate" in e for e in preflight_errors(dup, gold_dir))
    badcls = [{"id": "x1", "class": "Z", "query": "q"}]
    assert any("class" in e for e in preflight_errors(badcls, gold_dir))
