"""Hermetic tests for bench/score.py + driver guard helpers — no network, no LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.run_benchmark import parse_relevance, preflight_errors, select_queries
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
          crosschecked: bool = False, subquestion: str = "sq") -> dict:
    return {"id": f"c{hash(statement) & 0xffff}",
            "statement": statement,
            "subquestion": subquestion,
            "evidence": [],
            "verdict": verdict,
            "confidence": confidence,
            "crosschecked": crosschecked,
            "conflicts": conflicts or [],
            "note": ""}


def ledger(claims: list[dict], *, conflicts: list | None = None,
           gaps: list | None = None) -> dict:
    return {"query": "q", "created_at": "2026-01-01T00:00:00Z",
            "surfaces": ["web"], "confidence_counts": {},
            "claims": claims, "gaps": gaps or [],
            "crosscheck": {},
            "conflicts": conflicts or []}


def gold(cls: str, expected: list[dict], *, query_id: str = "q") -> dict:
    return {"query_id": query_id, "class": cls,
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
    assert m["precision_supported"] == 2 / 3   # lexical: unmatched = not correct
    assert m["precision_unscored_n"] == 0      # (judge path only)
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
    """One F, one U, one D ledger (query ids f1/u1/d1) whose combined metrics
    satisfy A1–A4 under the re-spec gates (A5/A6 take their own inputs)."""
    f_expected = ([exp(f"Solid fact {i} is well documented.") for i in range(11)]
                  + [exp(f"False claim {i} is not true.", "incorrect")
                     for i in range(4)]
                  + [exp("A twelfth correct fact no claim covers.")])
    f_claims = []
    # supported-correct: 3 high (cross-checked), 6 medium, 2 low
    for i in range(3):
        f_claims.append(claim(f"Solid fact {i} is well documented.",
                              confidence="high", crosschecked=True))
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
    f = compute_query_metrics(ledger(f_claims),
                              gold("F", f_expected, query_id="f1"))
    u = compute_query_metrics(
        ledger([claim("Scant trace of an answer.", verdict="unsupported",
                      confidence="unsupported"),
                claim("Weak hint only.", verdict="partial",
                      confidence="low")]),
        gold("U", [], query_id="u1"))
    d = compute_query_metrics(
        ledger([claim("Side A is documented.", verdict="contradicted",
                      confidence="low", conflicts=["Side B is documented."])],
               conflicts=[{"a": "Side A", "b": "Side B"}]),
        gold("D", [exp("Side A is documented.", "contested"),
                   exp("Side B is documented.", "contested")],
             query_id="d1"))
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


def _nocc_f_metric():
    """Paired-arm F metric for the SAME query (f1) with zero high claims
    (no cross-check) — the A4 benefit is the with-arm high_share."""
    f_nocc_claims = [claim(f"Solid fact {i} is well documented.",
                           confidence="medium") for i in range(11)]
    return compute_query_metrics(
        ledger(f_nocc_claims),
        gold("F", [exp(f"Solid fact {i} is well documented.")
                    for i in range(11)], query_id="f1"))


def _stable_reruns(n: int = 3) -> list[list[dict]]:
    """n reruns of the SAME query with identical confidence distributions and
    identical sub-question sets — a perfectly deterministic arm."""
    def _one():
        return ledger([claim("First stable claim statement here.",
                             verdict="supported", confidence="medium"),
                       claim("Second stable claim statement here.",
                             verdict="supported", confidence="high",
                             crosschecked=True),
                       claim("Third weak trace here.", verdict="unsupported",
                             confidence="unsupported")])
    return [[_one() for _ in range(n)]]


def test_gates_all_pass():
    qm = _all_pass_query_metrics()
    # paired arm: same query ids (f1/u1/d1), no high claims -> benefit visible
    qm_nocc = [_nocc_f_metric(), qm[1], qm[2]]
    g = gates(qm, q_metrics_nocc=qm_nocc,
              relevance_judgements=[1, 1, 1, 0],
              rerun_groups=_stable_reruns())
    assert g["A1_precision_fabrication"]["ok"] is True   # precision 1.0, fab 0
    assert g["A2_calibration"]["ok"] is True             # medium 0.75, corr ok
    assert g["A3_honest_failure_U"]["ok"] is True        # sub-q share 1.0
    assert g["A4_crosscheck_benefit"]["ok"] is True      # fires + delta
    assert g["A5_relevance"]["ok"] is True               # median 1.0
    assert g["A6_determinism"]["ok"] is True             # dist L1 0 <= 0.30
    assert g["A1_precision_fabrication"]["value"]["precision_supported"] == 1.0
    rel = g["A2_calibration"]["value"]["reliability"]
    assert rel["high"] == 1.0 and abs(rel["medium"] - 0.75) < 1e-9 \
        and rel["low"] == 0.5
    v4 = g["A4_crosscheck_benefit"]["value"]
    assert v4["n_paired"] == 3
    assert v4["D_fired_with"] == 1
    assert v4["high_share_with"] > v4["high_share_without"]
    assert v4["precision_with"] >= v4["precision_without"] - 0.05
    assert g["A6_determinism"]["value"]["median_pairwise_l1"] == 0.0
    assert g["A6_determinism"]["value"]["median_pairwise_subq_jaccard"] == 1.0


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
    # A6 fail: divergent confidence distributions across reruns
    divergent = [[
        ledger([claim("Same topic claim.", confidence="low")]),
        ledger([claim("Same topic claim.", confidence="medium")]),
        ledger([claim("Same topic claim.", confidence="high")]),
    ]]
    g3 = gates(_all_pass_query_metrics(), rerun_groups=divergent)
    assert g3["A6_determinism"]["ok"] is False   # dist L1 > 0.30
    # A6 statement-level flip_rate is informational: a reversal with NO
    # rerun groups is not-applicable, not a FAIL (flip is reported, not gated)
    g3b = gates(_all_pass_query_metrics(),
                flip_pairs=[(ledger([claim("Kill-switch stopped it.",
                                          verdict="supported")]),
                             ledger([claim("Kill-switch stopped it.",
                                          verdict="contradicted")]))])
    assert g3b["A6_determinism"]["ok"] is None
    assert g3b["A6_determinism"]["value"]["flip_rate"] == 1.0
    # A2 fail: corroboration below the floor (cross-check never fired)
    f_nocorr = compute_query_metrics(
        ledger([claim("Supported fact zero is well documented.",
                      confidence="medium", crosschecked=False),
                claim("Supported fact one is well documented.",
                      confidence="medium", crosschecked=False)]),
        gold("F", [exp("Supported fact zero is well documented."),
                   exp("Supported fact one is well documented.")]))
    g5 = gates([f_nocorr,
                compute_query_metrics(
                    ledger([claim("Nothing found.", verdict="unsupported",
                                  confidence="unsupported")]),
                    gold("U", []))])
    assert g5["A2_calibration"]["ok"] is False       # corr 0.0 < 0.05
    # Not-applicable: no data for a gate -> None, never FAIL
    g4 = gates([_all_pass_query_metrics()[0]])
    assert g4["A3_honest_failure_U"]["ok"] is None
    assert g4["A5_relevance"]["ok"] is None
    assert g4["A6_determinism"]["ok"] is None
    # A4 without the paired arm is n/a, never PASS on a mainline fire alone
    assert gates(_all_pass_query_metrics())["A4_crosscheck_benefit"]["ok"] \
        is None
    # A4 with <2 paired same-query pairs is n/a (evaluated at full-run scale)
    only_one = gates([_all_pass_query_metrics()[0]],
                     q_metrics_nocc=[_nocc_f_metric()])
    assert only_one["A4_crosscheck_benefit"]["ok"] is None
    # A2 with NO populated medium bucket is n/a, not a silent pass
    u_only = compute_query_metrics(
        ledger([claim("Trace only.", verdict="unsupported",
                      confidence="unsupported")]),
        gold("U", []))
    assert gates([u_only])["A2_calibration"]["ok"] is None


# --------------------------------------------------------------------------
# re-spec A2/A3/A6 metric behavior
# --------------------------------------------------------------------------

def test_corroboration_rate_computed_per_query():
    """Corroboration rate = asserted claims the cross-check saw (crosschecked
    flag) or that reached high, over asserted claims. Ledger-only, no gold."""
    l = ledger([claim("A.", confidence="high", crosschecked=True),
                claim("B.", confidence="medium", crosschecked=True),
                claim("C.", confidence="medium"),
                claim("D.", confidence="low"),
                claim("E.", verdict="unsupported", confidence="unsupported")])
    m = compute_query_metrics(l, None)
    assert m["corroborated_n"] == 2          # A + B asserted & crosschecked
    assert m["asserted_n"] == 4
    assert abs(m["corroboration_rate"] - 0.5) < 1e-9


def test_a3_subquestion_unresolved_semantics():
    """A U sub-question is honestly unresolved when every claim is
    (unsupported|low) OR it produced no claims and appears in a gap. A
    confident medium claim makes its sub-question resolved."""
    g = gold("U", [])
    # two sub-questions: sq1 unresolved (all low/un), sq2 resolved (medium)
    l = ledger([
        claim("Nothing on sq1.", subquestion="sq1", verdict="unsupported",
              confidence="unsupported"),
        claim("Weak trace on sq1.", subquestion="sq1", verdict="partial",
              confidence="low"),
        claim("Confident tangent on sq2.", subquestion="sq2",
              confidence="medium"),
    ])
    m = compute_query_metrics(l, g)
    assert m["subquestion_total_n"] == 2
    assert m["subquestion_unresolved_n"] == 1
    assert m["subquestion_unresolved_U"] == 0.5


def test_a3_gap_named_subquestion_is_unresolved():
    """A sub-question that produced no claims and appears in a gap
    ('no evidence found for: X') counts as honestly unresolved."""
    g = gold("U", [])
    l = ledger([claim("Confident tangent.", confidence="medium")],
               gaps=["no evidence found for: sq-without-claims"])
    m = compute_query_metrics(l, g)
    assert m["subquestion_total_n"] == 2       # claim's sq + gap-named sq
    assert m["subquestion_unresolved_n"] == 1  # the gap-named one
    assert m["subquestion_unresolved_U"] == 0.5


def test_a3_claims_less_u_ledger_registers_gap_subquestions():
    """A U run that produced NO claims at all still registers gap-named
    sub-questions as honestly unresolved (the pipeline admitted failure)."""
    g = gold("U", [])
    l = ledger([], gaps=["no evidence found for: trieste-1928-tonnage"])
    m = compute_query_metrics(l, g)
    assert m["subquestion_total_n"] == 1
    assert m["subquestion_unresolved_n"] == 1
    assert m["subquestion_unresolved_U"] == 1.0
    assert m["unsupported_share_U_n"] == 0  # claim-level has nothing to say


def test_normalized_conf_l1_and_subquestion_jaccard():
    from bench.score import normalized_conf_l1, subquestion_jaccard
    same_a = ledger([claim("X.", confidence="medium") for _ in range(4)])
    same_b = ledger([claim("X.", confidence="medium") for _ in range(3)])
    assert normalized_conf_l1(same_a, same_b) == 0.0
    diff = ledger([claim("X.", confidence="high") for _ in range(4)])
    assert normalized_conf_l1(same_a, diff) == 1.0
    assert normalized_conf_l1(ledger([]), same_b) is None
    # sub-question set overlap
    a = ledger([claim("A.", subquestion="q1"),
                claim("B.", subquestion="q2")])
    b = ledger([claim("C.", subquestion="q2"),
                claim("D.", subquestion="q3")])
    assert subquestion_jaccard(a, b) == 1 / 3
    assert subquestion_jaccard(ledger([]), b) is None


def test_a6_rerun_groups_need_three_and_report_jaccard():
    """A6 requires >=3 usable reruns of a query; two reruns stay n/a; the
    median pairwise sub-question Jaccard is reported alongside the gating L1."""
    # only 2 reruns -> not applicable, never a pass
    one = _stable_reruns(3)[0][0]
    two = gates(_all_pass_query_metrics(), rerun_groups=[[one, one]])
    assert two["A6_determinism"]["ok"] is None
    assert two["A6_determinism"]["value"]["n_rerun_groups_ge3"] == 0


def test_a6_claims_less_rerun_is_not_a_phantom_third():
    """Codex P1 regression: a claims-less rerun has no confidence
    distribution, so a 3-ledger group with one empty rerun is only TWO usable
    runs — A6 must stay n/a, never pass on a phantom third rerun."""
    one = _stable_reruns(3)[0][0]
    phantom = gates(_all_pass_query_metrics(),
                    rerun_groups=[[one, one, ledger([])]])
    assert phantom["A6_determinism"]["ok"] is None
    assert phantom["A6_determinism"]["value"]["n_rerun_groups_ge3"] == 0
    # with three genuinely usable reruns it does gate (and passes)
    real = gates(_all_pass_query_metrics(), rerun_groups=[[one, one, one]])
    assert real["A6_determinism"]["ok"] is True
    assert real["A6_determinism"]["value"]["n_rerun_groups_ge3"] == 1


def test_subquestion_jaccard_includes_gap_only_subquestions():
    """Codex P2 regression: a planned sub-question that found no evidence
    exists in the plan as a gap, so it must count toward plan overlap —
    otherwise diverging gap-only plans look identical (Jaccard 1.0)."""
    from bench.score import subquestion_jaccard
    a = ledger([claim("A.", subquestion="q1")])
    b = ledger([claim("B.", subquestion="q1")],
               gaps=["no evidence found for: q2", "no evidence found for: q3"])
    assert subquestion_jaccard(a, b) == 1 / 3   # {q1} vs {q1, q2, q3}
    # a gap-only run has a plan too: overlap with an unrelated claim run is 0
    gap_only = ledger([], gaps=["no evidence found for: q9"])
    assert subquestion_jaccard(a, gap_only) == 0.0
    assert subquestion_jaccard(ledger([]), gap_only) is None  # no plan at all


def test_a4_precision_tolerance_and_populations():
    """A4 (c) allows a 0.05 sample-noise tolerance: with-arm precision
    0.08 below the without-arm still FAILS; populations are reported."""
    qm = _all_pass_query_metrics()
    # with-arm F identical to the all-pass F but with one extra supported
    # FALSEHOOD: 11 correct of 12 supported -> 0.917 < 1.0 - 0.05, while high
    # share still beats the without-arm (so only condition (c) fails)
    f_bad_expected = ([exp(f"Solid fact {i} is well documented.")
                       for i in range(11)]
                      + [exp(f"False claim {i} is not true.", "incorrect")
                         for i in range(4)]
                      + [exp("A twelfth correct fact no claim covers.")])
    f_bad_claims = []
    for i in range(3):
        f_bad_claims.append(claim(f"Solid fact {i} is well documented.",
                                  confidence="high", crosschecked=True))
    for i in range(3, 11):
        f_bad_claims.append(claim(f"Solid fact {i} is well documented.",
                                  confidence="medium"))
    f_bad_claims.append(claim("False claim 0 is not true."))  # supported lie
    f_bad = compute_query_metrics(
        ledger(f_bad_claims),
        gold("F", f_bad_expected, query_id="f1"))
    qm_bad = [f_bad, qm[1], qm[2]]
    g = gates(qm_bad, q_metrics_nocc=[_nocc_f_metric(), qm[1], qm[2]])
    v4 = g["A4_crosscheck_benefit"]["value"]
    assert v4["placed_with"] == 12 and v4["placed_without"] == 11
    assert abs(v4["precision_with"] - 11 / 12) < 1e-9
    assert v4["high_share_with"] > v4["high_share_without"]  # (b) holds
    # precision 11/12 ~= 0.917 < 1.0 - 0.05 -> A4 fails on (c)
    assert g["A4_crosscheck_benefit"]["ok"] is False


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


def _write_scorecard(run_dir, entries, *, crosscheck="off",
                     gold_judge="on"):
    (run_dir / "scorecard.json").write_text(json.dumps(
        {"provenance": {"crosscheck": crosscheck,
                         "gold_judge": gold_judge},
         "queries": entries}))


def test_load_paired_metrics_injects_query_id(tmp_path):
    """A4 paired-arm loader returns ok scored queries' metrics and injects
    query_id from the scorecard entry when the scorer predates the field."""
    from bench.run_benchmark import load_paired_metrics
    arm = tmp_path / "nocc"
    arm.mkdir()
    _write_scorecard(arm, [
        {"id": "f1", "ok": True, "metrics": {"class": "F",
                                                "precision_supported": 1.0},
         "query": "q f1", "class": "F"},
        {"id": "u1", "ok": False, "metrics": {}},   # failed -> excluded
        {"id": "d1", "ok": True, "metrics": {"class": "D"},
         "query": "q d1", "class": "D"},
    ])
    expected = [{"id": "f1", "query": "q f1", "class": "F"},
                {"id": "d1", "query": "q d1", "class": "D"}]
    out = load_paired_metrics(arm, require_crosscheck="off",
                              require_gold_judge_on=True, expected=expected)
    assert [m.get("query_id") for m in out] == ["f1", "d1"]
    assert [m.get("class") for m in out] == ["F", "D"]
    with pytest.raises(ValueError):
        load_paired_metrics(tmp_path / "missing")


def test_load_paired_metrics_rejects_wrong_arm_or_drift(tmp_path):
    """Codex: paired metrics must come from the OPPOSITE cross-check arm,
    the SAME gold-judge mode, and match the main arm's query text/class —
    unrelated missions, mixed scoring modes, or two enabled arms must never
    pair into an A4 gate."""
    from bench.run_benchmark import load_paired_metrics
    arm = tmp_path / "paired"
    arm.mkdir()
    base = [{"id": "f1", "ok": True, "metrics": {"class": "F"},
             "query": "q f1", "class": "F"}]
    expected = [{"id": "f1", "query": "q f1", "class": "F"}]
    # same-arm (crosscheck on) must be rejected when an off-arm is required
    _write_scorecard(arm, base, crosscheck="on")
    with pytest.raises(ValueError, match="must be the OTHER arm"):
        load_paired_metrics(arm, require_crosscheck="off",
                            require_gold_judge_on=True, expected=expected)
    # scoring-mode mismatch must be rejected (judge vs lexical precision)
    _write_scorecard(arm, base, crosscheck="off", gold_judge="off(--no-judge)")
    with pytest.raises(ValueError, match="gold_judge"):
        load_paired_metrics(arm, require_crosscheck="off",
                            require_gold_judge_on=True, expected=expected)
    # query text drift must be rejected
    _write_scorecard(arm, base, crosscheck="off")
    drifted = [{"id": "f1", "query": "a DIFFERENT question", "class": "F"}]
    with pytest.raises(ValueError, match="text drifted"):
        load_paired_metrics(arm, require_crosscheck="off",
                            require_gold_judge_on=True, expected=drifted)
    # class drift must be rejected
    cls_drifted = [{"id": "f1", "query": "q f1", "class": "U"}]
    with pytest.raises(ValueError, match="class drifted"):
        load_paired_metrics(arm, require_crosscheck="off",
                            require_gold_judge_on=True, expected=cls_drifted)
    # a query absent from the main arm must be rejected
    extra = [{"id": "zz", "ok": True, "metrics": {"class": "F"},
              "query": "q zz", "class": "F"},
             {"id": "f1", "ok": True, "metrics": {"class": "F"},
              "query": "q f1", "class": "F"}]
    _write_scorecard(arm, extra, crosscheck="off")
    with pytest.raises(ValueError, match="not in the main"):
        load_paired_metrics(arm, require_crosscheck="off",
                            require_gold_judge_on=True, expected=expected)


def test_load_paired_metrics_binds_gold_and_scorer_revision(tmp_path):
    """Codex P1: A4 must never compare a main arm scored under the current
    gold/scorer against a paired arm scored under older gold/scorer
    semantics. When gold_dir is supplied the loader requires the paired
    run's recorded gold_rev == current sheets' revision AND scorer_rev ==
    the current scorer; stale or unverifiable pairs are rejected loudly."""
    import bench.run_benchmark as rb
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    _gold_file(gold_dir, "f1")
    qs = [{"id": "f1", "ok": True, "query": "q f1", "class": "F",
           "metrics": {"class": "F"}}]
    expected = [{"id": "f1", "query": "q f1", "class": "F"}]

    def _write(provenance_extra):
        arm = tmp_path / "nocc"
        (arm / "scorecard.json").parent.mkdir(parents=True, exist_ok=True)
        prov = {"crosscheck": "off", "gold_judge": "on"}
        prov.update(provenance_extra)
        (arm / "scorecard.json").write_text(json.dumps(
            {"provenance": prov, "queries": qs}))
        return arm

    current = rb.gold_revision(gold_dir, ["f1"])
    # matching gold + scorer -> accepted
    arm = _write({"scorer_rev": rb.SCORER_REVISION, "gold_rev": current})
    out = rb.load_paired_metrics(arm, require_crosscheck="off",
                                 require_gold_judge_on=True,
                                 expected=expected, gold_dir=gold_dir)
    assert [m["query_id"] for m in out] == ["f1"]
    # stale gold (sheet edited since the paired run) -> rejected
    arm2 = _write({"scorer_rev": rb.SCORER_REVISION,
                   "gold_rev": current + "ff"})
    with pytest.raises(ValueError, match="gold revision"):
        rb.load_paired_metrics(arm2, require_crosscheck="off",
                               require_gold_judge_on=True,
                               expected=expected, gold_dir=gold_dir)
    # older scorer revision -> rejected
    arm3 = _write({"scorer_rev": "pre-gate-respec", "gold_rev": current})
    with pytest.raises(ValueError, match="scorer revision"):
        rb.load_paired_metrics(arm3, require_crosscheck="off",
                               require_gold_judge_on=True,
                               expected=expected, gold_dir=gold_dir)
    # no recorded revisions (pre-binding run) -> rejected, not guessed
    arm4 = _write({})
    with pytest.raises(ValueError, match="scorer revision"):
        rb.load_paired_metrics(arm4, require_crosscheck="off",
                               require_gold_judge_on=True,
                               expected=expected, gold_dir=gold_dir)


def test_gold_revision_content_sensitive_and_deterministic(tmp_path):
    """gold_revision hashes sheet content + presence: unchanged sheets give
    the same revision; an edit or added sheet changes it."""
    import bench.run_benchmark as rb
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    _gold_file(gold_dir, "f1")
    _gold_file(gold_dir, "d1", "D")
    rev1 = rb.gold_revision(gold_dir, ["f1", "d1"])
    assert rev1 == rb.gold_revision(gold_dir, ["d1", "f1"])  # order-free
    (gold_dir / "f1.json").write_text(json.dumps({"query_id": "f1",
                                                   "class": "F",
                                                   "changed": True}))
    assert rb.gold_revision(gold_dir, ["f1", "d1"]) != rev1
    rev2 = rb.gold_revision(gold_dir, ["f1", "d1"])
    assert rev2 == rb.gold_revision(gold_dir, ["f1", "d1"])  # deterministic


def test_load_paired_metrics_prefers_rescore_scorecard(tmp_path):
    """A paired run that was re-scored (scorecard-rescore.json present) is
    used instead of the original scorecard — the owner can refresh a stale
    paired arm under current gold without new paid missions."""
    import bench.run_benchmark as rb
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    _gold_file(gold_dir, "f1")
    current = rb.gold_revision(gold_dir, ["f1"])
    expected = [{"id": "f1", "query": "q f1", "class": "F"}]
    arm = tmp_path / "paired"
    arm.mkdir()
    # stale original scorecard (old gold revision, old scorer)
    (arm / "scorecard.json").write_text(json.dumps({
        "provenance": {"crosscheck": "off", "gold_judge": "on",
                         "scorer_rev": "pre-gate-respec",
                         "gold_rev": "deadbeef"},
        "queries": [{"id": "f1", "ok": True, "query": "q f1",
                      "class": "F", "metrics": {"class": "F"}}]}))
    # fresh re-score under the current gold/scorer
    (arm / "scorecard-rescore.json").write_text(json.dumps({
        "provenance": {"crosscheck": "off", "gold_judge": "on",
                         "scorer_rev": rb.SCORER_REVISION,
                         "gold_rev": current},
        "queries": [{"id": "f1", "ok": True, "query": "q f1",
                      "class": "F", "metrics": {"class": "F",
                                                     "precision_supported": 1.0}}]}))
    out = rb.load_paired_metrics(arm, require_crosscheck="off",
                                 require_gold_judge_on=True,
                                 expected=expected, gold_dir=gold_dir)
    assert out and out[0].get("precision_supported") == 1.0


def test_rescore_provenance_records_crosscheck_and_revisions(tmp_path):
    """A scorecard-rescore.json must be self-describing: it records the
    ORIGINAL run's crosscheck arm plus the gold/scorer revisions it was
    rescored under, so it can later serve as a bound paired arm."""
    import bench.run_benchmark as rb
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    _gold_file(gold_dir, "f1")
    run_dir = tmp_path / "run"
    (run_dir / "f1").mkdir(parents=True)
    (run_dir / "f1" / "ledger.json").write_text(json.dumps(ledger([
        claim("Fact one is true.")])))
    queries = [{"id": "f1", "class": "F", "query": "q f1"}]
    rb._rescore_main(run_dir, queries, gold_dir, judge_enabled=False,
                     relevance=None, no_crosscheck=False, cap_usd=10.0,
                     crosscheck="on")
    sc = json.loads((run_dir / "scorecard-rescore.json").read_text())
    prov = sc["provenance"]
    assert prov["crosscheck"] == "on"
    assert prov["scorer_rev"] == rb.SCORER_REVISION
    assert prov["gold_rev"] == rb.gold_revision(gold_dir, ["f1"])


def _rerun_dir(tmp_path, name, *query_ids, crosscheck="on", ok_ids=None):
    """Build a realistic determinism run dir: scorecard.json (provenance
    crosscheck + per-query ok status) plus empty per-query subdirs, matching
    the layout run_benchmark.py produces. Query ids default to all ok."""
    d = tmp_path / name
    for qid in query_ids:
        (d / qid).mkdir(parents=True, exist_ok=True)
    ok_ids = query_ids if ok_ids is None else ok_ids
    (d / "scorecard.json").write_text(json.dumps({
        "provenance": {"crosscheck": crosscheck},
        "queries": [{"id": qid, "ok": qid in ok_ids,
                      "metrics": {}} for qid in query_ids]}))
    return d


def test_collect_rerun_groups_requires_three_same_query_ledgers(tmp_path):
    """A6 determinism helper groups ledgers per query id; a query needs >=3
    rerun dirs to form a group; corrupt reruns are skipped, not fatal; a
    ledger whose scorecard entry is not ok is excluded (Codex)."""
    from bench.run_benchmark import collect_rerun_groups
    dirs = []
    for name in ("det-1", "det-2", "det-3"):
        d = _rerun_dir(tmp_path, name, "f1", "u1")
        dirs.append(d)
    # f1 present in all three; u1 only in two
    for i, d in enumerate(dirs):
        (d / "f1" / "ledger.json").write_text(json.dumps(
            {"claims": [{"statement": f"run {i}", "verdict": "supported",
                          "confidence": "medium", "subquestion": "q1"}],
             "query": "q f1",
             "confidence_counts": {"medium": 1}}))
        if i < 2:
            (d / "u1" / "ledger.json").write_text(json.dumps(
                {"query": "q u1", "claims": [], "confidence_counts": {}}))
    queries = [{"id": "f1", "query": "q f1"},
               {"id": "u1", "query": "q u1"}]
    groups = collect_rerun_groups(dirs, queries)
    assert len(groups) == 1 and len(groups[0]) == 3      # only f1 qualifies
    # a corrupt rerun is skipped: f1 drops to 2 usable -> no group forms
    (dirs[1] / "f1" / "ledger.json").write_text("not json{{")
    groups2 = collect_rerun_groups(dirs, queries)
    assert len(groups2) == 0                            # <3 usable reruns
    # a stale ledger whose scorecard marks the query ok:false is excluded
    stale = _rerun_dir(tmp_path, "det-4", "f1", ok_ids=())
    (stale / "f1" / "ledger.json").write_text(json.dumps(
        {"query": "q f1", "claims": [], "confidence_counts": {"high": 1}}))
    groups3 = collect_rerun_groups([dirs[0], dirs[1], stale], queries)
    assert len(groups3) == 0                            # stale never counts


def test_collect_rerun_groups_skips_mismatched_query_ledgers(tmp_path):
    """Codex: a rerun ledger whose embedded query text differs from the
    expected query is a different mission and must not count toward A6."""
    from bench.run_benchmark import collect_rerun_groups
    dirs = []
    for name in ("det-1", "det-2", "det-3", "det-4"):
        d = _rerun_dir(tmp_path, name, "f1")
        dirs.append(d)
    for i, d in enumerate(dirs):
        text = "q f1" if i < 3 else "a DIFFERENT question"
        (d / "f1" / "ledger.json").write_text(json.dumps(
            {"query": text, "claims": [], "confidence_counts": {"high": 1}}))
    queries = [{"id": "f1", "query": "q f1"}]
    groups = collect_rerun_groups(dirs, queries)
    assert len(groups) == 1 and len(groups[0]) == 3     # mismatched one skipped


def test_rerun_dirs_deduplicated_before_three_required(tmp_path):
    """Codex: passing the same rerun dir three times must NOT satisfy the
    >=3-distinct-runs requirement (three identical pairwise distances are all
    zero and would fabricate a determinism pass from one actual run)."""
    import bench.run_benchmark as rb
    d = _rerun_dir(tmp_path, "det-1", "f1")
    (d / "f1" / "ledger.json").write_text(json.dumps(
        {"query": "q f1", "claims": [], "confidence_counts": {"high": 1}}))
    from bench.run_benchmark import _load_optional_reruns
    with pytest.raises(SystemExit):
        _load_optional_reruns(f"{d},{d},{d}", parser=__import__(
            "argparse").ArgumentParser(), queries=[{"id": "f1",
                                                     "query": "q f1"}],
            require_crosscheck="on")
    # three genuinely distinct dirs pass the dedupe gate
    for name in ("det-2", "det-3"):
        d_other = _rerun_dir(tmp_path, name, "f1")
        (d_other / "f1" / "ledger.json").write_text(json.dumps(
            {"query": "q f1", "claims": [], "confidence_counts": {"high": 1}}))
    groups = _load_optional_reruns(
        f"{d},{tmp_path / 'det-2'},{tmp_path / 'det-3'}",
        queries=[{"id": "f1", "query": "q f1"}], require_crosscheck="on")
    assert groups and len(groups[0]) == 3


def test_a4_precisionless_paired_set_is_na_not_fail():
    """Codex P2: a paired set with a D query but no usable F/C precision
    population on either arm never measured condition (c) — A4 is n/a, never
    a FAIL on an unevaluated precision condition."""
    qm = _all_pass_query_metrics()
    # paired set = U + D only (no F/C): both precision values are None
    paired = gates([qm[1], qm[2]], q_metrics_nocc=[qm[1], qm[2]])
    v4 = paired["A4_crosscheck_benefit"]
    assert v4["ok"] is None
    assert v4["value"]["n_paired"] == 2
    assert v4["value"]["precision_with"] is None
    assert v4["value"]["precision_without"] is None


def test_a4_off_arm_main_run_is_rejected(tmp_path):
    """Codex: A4's main position must be the cross-check-ON arm. Rescoring a
    crosscheck=off run with a paired arm would evaluate the deltas in
    reverse — reject it before any gate is produced."""
    from bench.run_benchmark import main as _main  # noqa: F401 - CLI guarded
    # exercise the same guard logic directly via the loader's crosscheck rule:
    # a nocc main run requires a cc paired arm (off main -> 'on' required),
    # which the CLI now refuses up front; here we assert the CLI-level
    # behavior through the validator used by main() by checking the
    # provenance rejection when the arms do not differ.
    from bench.run_benchmark import load_paired_metrics
    arm = tmp_path / "paired"
    arm.mkdir()
    _write_scorecard(arm, [{"id": "f1", "ok": True,
                            "metrics": {"class": "F"},
                            "query": "q f1", "class": "F"}],
                     crosscheck="on")
    expected = [{"id": "f1", "query": "q f1", "class": "F"}]
    with pytest.raises(ValueError, match="must be the OTHER arm"):
        # a cc main (require off) paired with this cc arm is rejected
        load_paired_metrics(arm, require_crosscheck="off",
                            require_gold_judge_on=True, expected=expected)


def test_execute_reuse_drops_stale_rescore_artifact(tmp_path, monkeypatch):
    """Codex P1: reusing a named run dir must drop a prior
    scorecard-rescore.json before fresh missions overwrite scorecard.json —
    the paired-arm loader prefers the rescore artifact, so a stale one would
    feed A4 metrics from the wrong (older) run."""
    import sys
    from bench.run_benchmark import main as _main  # noqa: F401 - CLI guarded
    run_dir = tmp_path / "full-1"
    run_dir.mkdir()
    stale = run_dir / "scorecard-rescore.json"
    stale.write_text(json.dumps({"provenance": {"crosscheck": "on"}}))
    assert stale.exists()
    # Execute path: fresh --no-crosscheck mission with a paired arm is
    # rejected AFTER the run dir is prepared, so the stale rescore must
    # already be gone when that error fires (and gone before any new
    # scorecard.json could be written).
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark.py", "--run-id", "full-1",
        "--out", str(tmp_path),
        "--no-crosscheck", "--paired-arm", str(tmp_path / "nocc")])
    with pytest.raises(SystemExit):
        _main()
    assert not stale.exists()


def test_rescore_rejects_stale_ledger_from_failed_source(tmp_path, monkeypatch):
    """Codex P1: rescoring a run whose source scorecard marks a query failed
    (ok:false) must not score the stale ledger.json left behind — a
    fabricated rescored success would later be treated as authoritative by a
    paired-arm load."""
    import sys
    from bench.run_benchmark import REPO, main as _main  # noqa: F401
    spec = json.loads((REPO / "bench" / "queries.json").read_text())
    f1 = next(q for q in spec["queries"] if q["id"] == "f1-wannacry")
    run_dir = tmp_path / "run"
    (run_dir / "f1-wannacry").mkdir(parents=True)
    # stale ledger from an EARLIER successful mission; the latest mission
    # failed and scorecard.json records ok:false
    (run_dir / "f1-wannacry" / "ledger.json").write_text(json.dumps({
        "query": f1["query"], "claims": []}))
    (run_dir / "scorecard.json").write_text(json.dumps({
        "provenance": {"crosscheck": "on", "gold_judge": "off"},
        "queries": [{"id": "f1-wannacry", "ok": False,
                      "query": f1["query"], "class": f1["class"],
                      "metrics": {"class": "F"}}]}))
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py",
                                      "--rescore", str(run_dir),
                                      "--ids", "f1-wannacry",
                                      "--no-judge"])
    _main()
    out = json.loads((run_dir / "scorecard-rescore.json").read_text())
    q = out["queries"][0]
    assert q["ok"] is False
    assert "stale ledger" in q["error"]
    assert q["metrics"] == {}


def test_rescore_rejects_missing_crosscheck_provenance(tmp_path, monkeypatch):
    """Codex P1: rescoring a source scorecard without provenance.crosscheck
    must abort — defaulting a missing arm to 'on' could rescore an off-arm
    run as if it were the main arm and fabricate an arm identity in the
    rescore artifact."""
    import sys
    from bench.run_benchmark import main as _main
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    # legacy/malformed scorecard: queries scored but no crosscheck provenance
    (run_dir / "scorecard.json").write_text(json.dumps({
        "provenance": {"gold_judge": "on"},
        "queries": [{"id": "f1", "ok": True, "query": "q f1",
                      "class": "F", "metrics": {"class": "F"}}]}))
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py",
                                      "--rescore", str(run_dir),
                                      "--no-judge"])
    with pytest.raises(SystemExit):
        _main()


def test_rerun_dirs_without_usable_groups_fail_preflight(tmp_path):
    """Codex: three distinct but unusable rerun dirs (missing ledgers) must be
    rejected up front, not silently produce an empty A6 after missions ran."""
    from bench.run_benchmark import _load_optional_reruns
    dirs = [tmp_path / f"det-{i}" for i in (1, 2, 3)]  # dirs exist but empty
    for d in dirs:
        d.mkdir(exist_ok=True)
    groups = _load_optional_reruns(
        ",".join(str(d) for d in dirs),
        queries=[{"id": "f1", "query": "q f1"}])
    assert groups == []        # no usable group
    # the CLI-level guard (not shown here) turns this into parser.error; the
    # collect function returning [] is what main() rejects on



def test_rerun_dirs_require_crosscheck_arm_provenance(tmp_path):
    """Codex P1: rerun dirs must carry a scorecard whose crosscheck arm
    matches the evaluated run — cross-check promotes medium->high, so mixing
    arms measures configuration drift, not determinism. Missing or
    mismatched provenance aborts loudly."""
    from bench.run_benchmark import _load_optional_reruns
    import bench.run_benchmark as rb

    d1, d2, d3 = (_rerun_dir(tmp_path, n, "f1", crosscheck="on")
                  for n in ("det-1", "det-2", "det-3"))
    for d in (d1, d2, d3):
        (d / "f1" / "ledger.json").write_text(json.dumps(
            {"query": "q f1", "claims": [],
             "confidence_counts": {"high": 1}}))
    qs = [{"id": "f1", "query": "q f1"}]
    # all on-arm dirs match an on-arm evaluation
    groups = _load_optional_reruns(f"{d1},{d2},{d3}", queries=qs,
                                   require_crosscheck="on")
    assert groups and len(groups[0]) == 3
    # a single off-arm dir mixed in aborts loudly (not silently skipped)
    d4 = _rerun_dir(tmp_path, "det-4", "f1", crosscheck="off")
    (d4 / "f1" / "ledger.json").write_text(json.dumps(
        {"query": "q f1", "claims": [], "confidence_counts": {"high": 1}}))
    parser = __import__("argparse").ArgumentParser()
    with pytest.raises(SystemExit):
        _load_optional_reruns(f"{d1},{d2},{d4}", parser=parser, queries=qs,
                              require_crosscheck="on")
    # a dir with no scorecard at all is unverifiable -> abort
    d5 = tmp_path / "det-5"
    (d5 / "f1").mkdir(parents=True)
    (d5 / "f1" / "ledger.json").write_text(json.dumps(
        {"query": "q f1", "claims": [], "confidence_counts": {"high": 1}}))
    with pytest.raises(SystemExit):
        _load_optional_reruns(f"{d1},{d2},{d5}", parser=parser, queries=qs,
                              require_crosscheck="on")


def test_rerun_groups_reject_distribution_less_ledgers(tmp_path):
    """Codex P2: readable ledgers with NO claims and NO confidence counts are
    not usable reruns — a group of three such ledgers must not pass
    preflight (an empty distribution would make A6 n/a after paid missions).
    Ledgers that carry a plan via gap-names still qualify (plan overlap is
    measured separately from the distribution gate)."""
    from bench.run_benchmark import collect_rerun_groups
    dirs = []
    for name in ("det-1", "det-2", "det-3"):
        d = _rerun_dir(tmp_path, name, "f1")
        dirs.append(d)
    # three distribution-less, plan-less ledgers -> no group
    for d in dirs:
        (d / "f1" / "ledger.json").write_text(json.dumps(
            {"query": "q f1", "claims": [], "confidence_counts": {}}))
    qs = [{"id": "f1", "query": "q f1"}]
    assert collect_rerun_groups(dirs, qs) == []
    # same but each records a gap-named plan -> plan-bearing, so a group forms
    for d in dirs:
        (d / "f1" / "ledger.json").write_text(json.dumps(
            {"query": "q f1", "claims": [], "confidence_counts": {},
             "gaps": ["no evidence found for: q1"]}))
    groups = collect_rerun_groups(dirs, qs)
    assert len(groups) == 1 and len(groups[0]) == 3


def test_a6_gap_only_rerun_counts_toward_plan_overlap():
    """Codex P2: a claims-less rerun that recorded its planned questions as
    gaps still has a plan — it must contribute to the reported plan-overlap
    Jaccard (computed independently of the confidence-distribution filter),
    not be silently dropped from the group."""
    one = _stable_reruns(1)[0][0]          # two claim-bearing reruns
    gap_only = ledger([], gaps=["no evidence found for: q1"])
    g = gates(_all_pass_query_metrics(), rerun_groups=[[one, one, gap_only]])
    # plan overlap is measured over all three plan-bearing ledgers...
    assert g["A6_determinism"]["value"]["median_pairwise_subq_jaccard"] \
        is not None
    # ...while the distribution gate stays n/a (only two usable distributions)
    assert g["A6_determinism"]["ok"] is None


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


def test_gold_match_rejects_truth_critical_disagreement():
    """Token overlap must never certify a claim that contradicts gold on
    quantities or polarity — the whole point of A1/A2 credit."""
    from bench.score import best_gold_match
    g1957 = [exp("Sputnik 1 was launched by the Soviet Union in 1957.")]
    assert best_gold_match(
        "Sputnik 1 was launched by the Soviet Union in 1958.", g1957) is None
    assert gold_verdict("Sputnik 1 was launched by the Soviet Union in 1958.",
                        g1957) != "correct"
    # exact year still matches
    assert gold_verdict("Sputnik 1 was launched by the Soviet Union in 1957.",
                        g1957) == "correct"
    g_launch = [exp("The Soviet Union launched Sputnik 1.")]
    # polarity flip -> different claim, never credited
    assert gold_verdict("The Soviet Union did not launch Sputnik 1.",
                        g_launch) != "correct"
    assert best_gold_match("The Soviet Union did not launch Sputnik 1.",
                           g_launch) is None
    # same polarity near-duplicate still matches (wording only)
    assert best_gold_match("Sputnik 1 was put into orbit by the Soviet Union.",
                           g_launch) is not None
    # a claim citing no year still matches year-citing gold (subset is not
    # a contradiction — only BOTH citing differing years is blocked)
    vague = [exp("The launch happened in 1957.")]
    assert best_gold_match("The launch happened.", vague) is not None
    assert best_gold_match("The launch occurred.", vague) is None  # weak sim
    # non-year counts omitted by the claim do not block a match
    scale = [exp("WannaCry affected around 150 countries in May 2017.")]
    assert best_gold_match("WannaCry affected around 150 countries.",
                           scale) is not None
    # ...but explicitly DISAGREEING quantities never match (neither set
    # contains the other), even though digits drop out of token overlap
    port = [exp("EternalBlue sends crafted packets to a vulnerable machine "
                "over port 445.")]
    assert best_gold_match("EternalBlue sends crafted packets to a "
                           "vulnerable machine over port 444.", port) is None
    assert gold_verdict("EternalBlue sends crafted packets to a vulnerable "
                        "machine over port 444.", port) != "correct"
    big = [exp("WannaCry infected roughly 200,000 computers across more "
               "than 150 countries in May 2017.")]
    assert best_gold_match("WannaCry infected 100,000 computers across 15 "
                           "countries.", big) is None


def test_real_seed_gold_sheets_pass_preflight():
    """Every bench/gold/<id>.json in the repo must satisfy the driver's own
    pre-flight validation against bench/queries.json — schema drift breaks the
    benchmark silently otherwise."""
    from bench.run_benchmark import preflight_errors
    repo = Path(__file__).resolve().parents[1]
    spec = json.loads((repo / "bench" / "queries.json").read_text())
    queries = spec["queries"]
    assert len(queries) == 7
    for q in queries:
        assert (repo / "bench" / "gold" / f"{q['id']}.json").exists(), \
            f"seed query {q['id']} has no gold sheet"
    errs = preflight_errors(queries, repo / "bench" / "gold")
    assert errs == [], f"gold pre-flight errors: {errs}"


def test_d2_radio_priority_claims_never_certified():
    """d2 regression: flat one-sided 'X invented the radio' claims must
    resolve to contested entries, never gain correct credit by matching the
    Tesla-patent or 1943-Supreme-Court anchors (mirror of the d1 test)."""
    repo = Path(__file__).resolve().parents[1]
    g = json.loads((repo / "bench" / "gold" / "d2-radio.json").read_text())
    expected = g["expected_claims"]
    assert gold_verdict("Guglielmo Marconi invented the radio.",
                        expected) == "contested"
    assert gold_verdict("Nikola Tesla invented the radio, not Marconi.",
                        expected) == "contested"
    # positive phrasing (no negation) must classify as contested too (P2)
    assert gold_verdict("Nikola Tesla invented the radio.",
                        expected) == "contested"
    # the neutral patent anchor still matches when asserted verbatim-ish
    assert gold_verdict("In 1900 the United States Patent Office granted "
                        "Nikola Tesla patent number 645,576 for a wireless "
                        "transmission system, on an application filed on "
                        "2 September 1897.", expected) == "correct"
    # the 320 U.S. 1 Stone-anticipation holding is the accurate reading;
    # 'patent number 763,772' avoids the lexical negation parser (P2)
    assert gold_verdict("In Marconi Wireless Telegraph Co. of America v. "
                        "United States (1943), the United States Supreme "
                        "Court held that the principal tuning claims of "
                        "Marconi's wireless patent number 763,772 were "
                        "invalid because they were anticipated by an earlier "
                        "patent of the American inventor John Stone Stone.",
                        expected) == "correct"
    # concise non-verbatim formulations must still credit the anchors (P2)
    assert gold_verdict("Tesla's patent 645,576 was filed in 1897 and "
                        "granted in 1900.", expected) == "correct"
    assert gold_verdict("The United States Supreme Court held claims 10 and "
                        "11 of Marconi's patent 763,772 invalid as "
                        "anticipated by John Stone Stone's earlier patent.",
                        expected) == "correct"
    # Marconi-side factual anchor + concise phrasing (round-4 P2)
    assert gold_verdict("Guglielmo Marconi shared the 1909 Nobel Prize in "
                        "Physics with Karl Ferdinand Braun in recognition of "
                        "their contributions to the development of wireless "
                        "telegraphy.", expected) == "correct"
    assert gold_verdict("Marconi won the 1909 Nobel Prize in Physics for "
                        "his work developing wireless telegraphy.",
                        expected) == "correct"
    # false 'rejected' disposition of the Tesla patent is never credited
    assert gold_verdict("Tesla's patent 645,576 for wireless transmission "
                        "was filed in 1897 and rejected in 1900.",
                        expected) == "incorrect"
    # Nobel-fused one-sided invention claims resolve contested, not correct
    assert gold_verdict("Guglielmo Marconi received the 1909 Nobel Prize in "
                        "Physics in recognition that he invented the radio.",
                        expected) == "contested"
    # entity/category substitutions must never ride the correct anchors
    assert gold_verdict("In Marconi Wireless Telegraph Co. of America v. "
                        "United States (1943), the United States Supreme "
                        "Court held that Marconi's tuning claims were "
                        "invalid because they were anticipated by Nikola "
                        "Tesla.", expected) == "incorrect"
    assert gold_verdict("Guglielmo Marconi received the 1909 Nobel Prize "
                        "in Chemistry in recognition of his contributions to "
                        "wireless telegraphy.", expected) == "incorrect"
    assert gold_verdict("In 1900 the United States Patent Office granted "
                        "Marconi patent number 645,576 for a wireless "
                        "transmission system.", expected) == "incorrect"
    assert gold_verdict("In 1943 the United States Supreme Court held "
                        "claims 10 and 11 of Marconi's patent number 763,772 "
                        "valid despite John Stone Stone's earlier patent.",
                        expected) == "incorrect"
    assert gold_verdict("Guglielmo Marconi shared the 1909 Nobel Prize in "
                        "Physics with Nikola Tesla in recognition of their "
                        "contributions to the development of wireless "
                        "telegraphy.", expected) == "incorrect"
    # concise wrong-holder / wrong-co-recipient forms (round-8 P2s)
    assert gold_verdict("Marconi's patent 645,576 for wireless transmission "
                        "was filed in 1897 and granted in 1900.",
                        expected) == "incorrect"
    assert gold_verdict("Marconi and Tesla won the 1909 Nobel Prize in "
                        "Physics for wireless telegraphy.",
                        expected) == "incorrect"
    assert gold_verdict("Marconi and Braun shared the 1909 Nobel Prize in "
                        "Physics.", expected) == "correct"
    # grant synonyms must not tie-break into the rejection guard (round 9)
    assert gold_verdict("Tesla's patent 645,576 for wireless transmission "
                        "was filed in 1897 and issued in 1900.",
                        expected) == "correct"
    assert gold_verdict("Tesla's patent 645,576 for wireless transmission "
                        "was filed in 1897 and awarded in 1900.",
                        expected) == "correct"
    assert gold_verdict("Tesla's patent 645,576 for wireless transmission "
                        "was filed in 1897 and allowed in 1900.",
                        expected) == "correct"
    assert gold_verdict("Tesla's patent 645,576 for wireless transmission "
                        "was filed in 1897 and approved in 1900.",
                        expected) == "correct"
    assert gold_verdict("Tesla's patent 645,576 for wireless transmission "
                        "was patented in 1900.", expected) == "correct"
    # the claim-16 holding (valid and infringed) is a scored anchor too
    assert gold_verdict("The Supreme Court held claim 16 of Marconi patent "
                        "763,772 valid and infringed.", expected) == "correct"
    # concise invalid-holding phrasing credits; 'valid despite' never does
    assert gold_verdict("In 1943 the Supreme Court held Marconi's claims 10 "
                        "and 11 invalid due to Stone's earlier patent.",
                        expected) == "correct"
    assert gold_verdict("In 1943 the United States Supreme Court held "
                        "claims 10 and 11 of Marconi's patent number 763,772 "
                        "valid despite John Stone Stone's earlier patent.",
                        expected) == "incorrect"
    # Marconi-only Nobel claims are correct; inflected 'invalidated' reversal
    # of claim 16 never credits (round-11 P2s)
    assert gold_verdict("Marconi won the 1909 Nobel Prize in Physics.",
                        expected) == "correct"
    assert gold_verdict("In 1943, the United States Supreme Court invalidated "
                        "claim 16 of Marconi's patent 763,772.",
                        expected) != "correct"
    assert gold_verdict("In 1943, the United States Supreme Court held claim "
                        "16 of Marconi's patent 763,772 unenforceable.",
                        expected) != "correct"
    assert gold_verdict("In 1943, the United States Supreme Court voided "
                        "claim 16 of Marconi's patent 763,772.",
                        expected) != "correct"
    # claim-16 'invalid'-by-Stone must not ride the claims-10/11 anchor; and
    # unenforceable is a distinct disposition, never credited as invalid
    assert gold_verdict("In Marconi Wireless Telegraph Co. of America v. "
                        "United States (1943), the United States Supreme "
                        "Court held claim 16 of Marconi's patent number "
                        "763,772 invalid because it was anticipated by the "
                        "earlier patent of the American inventor John Stone "
                        "Stone.", expected) != "correct"
    assert gold_verdict("In 1943 the United States Supreme Court held "
                        "Marconi's claims 10 and 11 unenforceable as "
                        "anticipated by Stone's earlier patent.",
                        expected) != "correct"
    assert gold_verdict("Guglielmo Marconi and Nikola Tesla won the 1909 "
                        "Nobel Prize in Physics.", expected) == "incorrect"
    # 'No.' patent abbreviation is not a negation (round-8 P2 matcher fix)
    assert gold_verdict("Tesla's patent No. 645,576 for wireless "
                        "transmission was filed in 1897 and granted in "
                        "1900.", expected) == "correct"
    assert gold_verdict("Tesla's patent No. 645,576 for wireless "
                        "transmission was filed in 1897 and rejected in "
                        "1900.", expected) == "incorrect"
    # the 'Court declared Tesla the inventor' overclaim must never win credit
    assert gold_verdict("In 1943 the United States Supreme Court declared "
                        "Nikola Tesla the inventor of radio.",
                        expected) != "correct"


def test_priority_claim_never_certified_by_patent_fact():
    """d1 regression: 'Bell invented the telephone' must resolve to a
    contested entry, never gain correct credit by matching the patent fact."""
    repo = Path(__file__).resolve().parents[1]
    g = json.loads((repo / "bench" / "gold" / "d1-telephone.json").read_text())
    expected = g["expected_claims"]
    assert gold_verdict("Alexander Graham Bell invented the telephone.",
                        expected) == "contested"
    assert gold_verdict("Antonio Meucci invented the telephone, not Bell.",
                        expected) == "contested"
    # the neutral patent fact still matches when asserted verbatim-ish
    assert gold_verdict("In 1876 the United States Patent Office granted "
                        "Alexander Graham Bell a patent for the telephone.",
                        expected) == "correct"


def test_atomic_gold_matches_verifier_shaped_claims():
    """c1 regression: atomic gold entries match ordinary single-fact claims."""
    repo = Path(__file__).resolve().parents[1]
    g = json.loads((repo / "bench" / "gold" / "c1-comet-asteroid.json").read_text())
    expected = g["expected_claims"]
    assert gold_verdict("Comets consist largely of ice and dust.",
                        expected) == "correct"
    assert gold_verdict("Asteroids are rocky or metallic bodies.",
                        expected) == "correct"
    assert gold_verdict("Comets grow tails as they approach the Sun.",
                        expected) == "correct"
    # synonym paraphrase without the gold's 'largely' still matches
    assert gold_verdict("Comets consist of ice and dust.",
                        expected) == "correct"
    # phrasing-variant entry covers natural wording like Codex's example
    assert gold_verdict("Comets are icy bodies made of dust.",
                        expected) == "correct"


def test_quantity_roles_never_swapped():
    """Role-aware check: '150 computers' must not match gold's '200,000
    computers' even though {150} is a subset of gold's quantities."""
    scale = [exp("WannaCry infected roughly 200,000 computers in May 2017.")]
    exact = "WannaCry infected roughly 200,000 computers in May 2017."
    assert gold_verdict(exact, scale) == "correct"
    swapped = "WannaCry infected roughly 150 computers in May 2017."
    assert gold_verdict(swapped, scale) != "correct"            # role swap
    machines = "WannaCry infected roughly 200,000 machines in May 2017."
    assert gold_verdict(machines, scale) == "correct"           # synonym role
    # f1 sheet: the 150-countries fact stays correct; the infection-count
    # fact is contested (estimates 200k-300k+), so count claims are excluded
    repo = Path(__file__).resolve().parents[1]
    f1 = json.loads((repo / "bench" / "gold" / "f1-wannacry.json").read_text())
    fe = f1["expected_claims"]
    assert gold_verdict("WannaCry affected computers in more than 150 "
                        "countries in May 2017.", fe) == "correct"
    assert gold_verdict("WannaCry infected roughly 200,000 computers "
                        "in May 2017.", fe) == "contested"

def test_variant_does_not_inflate_recall():
    repo = Path(__file__).resolve().parents[1]
    c1 = json.loads((repo / "bench" / "gold" / "c1-comet-asteroid.json").read_text())
    g = gold("C", c1["expected_claims"])
    # one claim per distinct fact, composition stated in the VARIANT wording
    stmts = ["Comets are icy bodies made of dust.",   # variant phrasing
             "Asteroids are rocky or metallic bodies.",
             "Comets develop a coma and tails as they approach the Sun.",
             "Both comets and asteroids are leftovers from the formation "
             "of the Solar System.",
             "Most asteroids orbit the Sun in the main belt between Mars "
             "and Jupiter.",
             "Comets typically travel on more eccentric orbits than "
             "main-belt asteroids."]
    claims = [claim(s, verdict="supported", confidence="medium")
              for s in stmts]
    m = compute_query_metrics(ledger(claims), g)
    assert m["recall_gold"] == 1.0, m   # 6 base facts, all covered
    assert m["recall_gold_n"] == 6      # variant not a 7th denominator entry


def test_f1_positive_patch_claim_matches_immune_paraphrase():
    repo = Path(__file__).resolve().parents[1]
    f1 = json.loads((repo / "bench" / "gold" / "f1-wannacry.json").read_text())
    expected = f1["expected_claims"]
    assert gold_verdict("Systems that had installed the MS17-010 patch, "
                        "released in March 2017, were immune to "
                        "EternalBlue-based propagation.", expected) == "correct"


def test_preflight_rejects_dangling_variant(tmp_path):
    from bench.run_benchmark import preflight_errors
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    queries = [{"id": "f1", "class": "F", "query": "q"}]
    (gold_dir / "f1.json").write_text(json.dumps({
        "query_id": "f1", "class": "F",
        "expected_claims": [
            {"statement": "Base fact one is true.", "gold_label": "correct",
             "confidence_class": "high"},
            {"statement": "Base fact one holds.", "variant_of": "No such base.",
             "gold_label": "correct", "confidence_class": "high"}]}))
    errs = preflight_errors(queries, gold_dir)
    assert any("variant_of does not match" in e for e in errs)


def test_spelled_out_quantity_disagreement_rejected():
    """'three months' vs gold 'two months' must reject even though the
    numbers are spelled out and digits drop out of token overlap."""
    repo = Path(__file__).resolve().parents[1]
    f1 = json.loads((repo / "bench" / "gold" / "f1-wannacry.json").read_text())
    expected = f1["expected_claims"]
    exact = "Microsoft released the MS17-010 security update in March 2017, " \
            "about two months before the WannaCry outbreak."
    assert gold_verdict(exact, expected) == "correct"
    wrong = "Microsoft released the MS17-010 security update in March 2017, " \
            "about three months before the WannaCry outbreak."
    assert gold_verdict(wrong, expected) != "correct"


def test_f1_split_origin_and_release_bases():
    """NSA-origin and Shadow-Brokers-release are distinct base facts: a
    release-only claim must not credit the NSA-origin fact, and each base
    is covered by its own claim."""
    repo = Path(__file__).resolve().parents[1]
    f1 = json.loads((repo / "bench" / "gold" / "f1-wannacry.json").read_text())
    expected = f1["expected_claims"]
    assert gold_verdict("The Shadow Brokers released EternalBlue in April 2017.",
                        expected) == "correct"
    assert gold_verdict("EternalBlue was developed by the United States "
                        "National Security Agency.", expected) == "correct"
    g = gold("F", expected)
    m = compute_query_metrics(ledger([
        claim("The Shadow Brokers released EternalBlue in April 2017.",
              verdict="supported")]), g)
    assert m["recall_gold"] is not None and m["recall_gold"] < 1.0  # NSA fact uncovered
    assert m["recall_gold_n"] >= 20  # every distinct canonical fact is a base


def test_f1_positive_patch_delay_claim_matches():
    repo = Path(__file__).resolve().parents[1]
    f1 = json.loads((repo / "bench" / "gold" / "f1-wannacry.json").read_text())
    expected = f1["expected_claims"]
    assert gold_verdict("many organizations left the available MS17-010 patch "
                        "unapplied", expected) == "correct"


def test_synonym_quantity_roles_never_swapped():
    """'150 machines' must not match gold '200,000 computers': anchor roles
    are normalized (machines -> computers) before comparison."""
    scale = [exp("WannaCry infected roughly 200,000 computers in May 2017.")]
    swapped = "WannaCry infected roughly 150 machines in May 2017."
    assert gold_verdict(swapped, scale) != "correct"
    exact = "WannaCry infected roughly 200,000 machines in May 2017."
    assert gold_verdict(exact, scale) == "correct"


def test_double_negation_never_certifies_opposite():
    repo = Path(__file__).resolve().parents[1]
    f1 = json.loads((repo / "bench" / "gold" / "f1-wannacry.json").read_text())
    expected = f1["expected_claims"]
    # 'did not spread without requiring' states the opposite of the no-click
    # fact; two negation markers must not collapse to gold's single 'without'
    assert gold_verdict("WannaCry did not spread without requiring any user "
                        "interaction.", expected) != "correct"
    assert gold_verdict("WannaCry spread without requiring any user "
                        "interaction.", expected) == "correct"


def test_conflicting_month_names_rejected():
    repo = Path(__file__).resolve().parents[1]
    f1 = json.loads((repo / "bench" / "gold" / "f1-wannacry.json").read_text())
    expected = f1["expected_claims"]
    gold_stmt = "The Shadow Brokers released EternalBlue in April 2017."
    assert gold_verdict(gold_stmt, expected) == "correct"
    assert gold_verdict("The Shadow Brokers released EternalBlue in May 2017.",
                        expected) != "correct"
    # omitting the month entirely still matches
    assert gold_verdict("The Shadow Brokers released EternalBlue in 2017.",
                        expected) == "correct"


def test_negation_count_treats_patent_number_abbreviation_as_non_negation():
    """'patent No. 763,772' spells a number, not a negation: polarity
    matching must not reject otherwise-verbatim claims that use the
    conventional abbreviation, while real negations keep counting."""
    from bench.score import _negation_count
    assert _negation_count("patent No. 763,772 was granted in 1900") == 0
    assert _negation_count("patent no. 763,772") == 0
    assert _negation_count("no evidence was found") == 1
    assert _negation_count("did not spread without user interaction") == 2
    assert _negation_count("no copy of the issue survived") == 1
    # a quantified negation is not an abbreviation: no period follows 'no'
    assert _negation_count("affected no 150 countries in May 2017") == 1


def test_status_antonym_and_synonym_matching():
    """granted/rejected and valid/invalid are truth-critical opposites the
    word-level matcher cannot see; grant synonyms normalize onto 'granted'
    so a true claim never tie-breaks into a mirror rejection guard."""
    from bench.score import _antonym_conflict, gold_verdict
    assert _antonym_conflict("the patent was granted in 1900",
                             "the patent was rejected in 1900")
    assert _antonym_conflict("claims 10 and 11 are invalid",
                             "claims 10 and 11 are valid")
    assert _antonym_conflict("the Court invalidated claim 16",
                             "claim 16 was valid and infringed")
    assert _antonym_conflict("held claim 16 unenforceable",
                             "claim 16 was valid and infringed")
    assert _antonym_conflict("the Court voided claim 16",
                             "claim 16 was valid and infringed")
    # unenforceable is a distinct disposition from invalid: it must never
    # match an 'invalid' anchor for credit, but it does conflict with it
    assert _antonym_conflict("claims 10 and 11 unenforceable",
                             "claims 10 and 11 invalid")
    assert _antonym_conflict("claims 10 and 11 unenforceable",
                             "claims 10 and 11 are valid")
    assert not _antonym_conflict("the patent was granted in 1900",
                                 "the patent was granted in 1901")
    gold = [{"statement": "Tesla's patent was granted in 1900.",
             "gold_label": "correct"},
            {"statement": "Tesla's patent was rejected in 1900.",
             "gold_label": "incorrect"}]
    assert gold_verdict("Tesla's patent was issued in 1900.",
                        gold) == "correct"
    assert gold_verdict("Tesla's patent was awarded in 1900.",
                        gold) == "correct"
    assert gold_verdict("Tesla's patent was allowed in 1900.",
                        gold) == "correct"
    assert gold_verdict("Tesla's patent was rejected in 1900.",
                        gold) == "incorrect"
