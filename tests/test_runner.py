"""Offline end-to-end runner tests (FakeLLM + real local connectors)."""

from __future__ import annotations

import json
from pathlib import Path

from veritas import Query, Surface, Verdict
from veritas.connectors import build_providers
from veritas.pipeline.runner import Runner

from tests.fake_llm_util import scripted_llm


def make_notes(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "alpha.md").write_text(
        "AlphaNote processes JSON files nightly. AlphaNote version 2.4 was "
        "released March 2025. It supports batching and retries.\n" * 3)
    (root / "beta.md").write_text(
        "BetaNote is a scheduling daemon. BetaNote supports cron syntax. "
        "BetaNote has been unmaintained since 2023.\n" * 3)
    return root


def test_semantic_corroboration_consumes_paraphrased_candidate(tmp_path: Path):
    """A cross-pass claim that restates a primary fact (different wording,
    same fact) is recognised after verification: it corroborates the primary
    claim instead of being appended as a duplicate candidate (full-1 A2
    finding: token matching alone corroborates ~4% of generative claims)."""
    from veritas.llm import FakeLLM
    from veritas.pipeline.prompts import CLAIMS_SYSTEM

    notes = make_notes(tmp_path)
    plan = json.dumps({"overview": "compare the two tools",
        "subquestions": [
            {"text": "What does AlphaNote do and what is its version?", "rationale": "r"},
            {"text": "Is BetaNote maintained?", "rationale": "r"}],
        "crosscheck_seed_note": "maintenance angle"})

    def claims_fn(user: str) -> str:
        # the cross-check pass extracts the SAME fact with different wording
        if "cross sub-question" in user:
            return json.dumps({"claims": [{
                "statement": "AlphaNote carries out nightly JSON batch processing.",
                "evidence_idx": [1]}], "noted_gaps": []})
        return json.dumps({"claims": [
            {"statement": "AlphaNote processes JSON files nightly.",
             "evidence_idx": [1]},
            {"statement": "BetaNote is unmaintained since 2023.",
             "evidence_idx": [1]},
        ], "noted_gaps": []})

    def corroborator(user: str) -> str:
        # match the independent-pass claim to the primary AlphaNote fact by
        # CONTENT, not position: primaries are verified on a thread pool, so
        # their order in the prompt is not deterministic
        import re
        try:
            prim_block = user.split("Primary claims (verified):", 1)[1]
            prim_block = prim_block.split(
                "Independent-pass claims (verified):", 1)[0]
        except IndexError:
            return json.dumps({"same_fact_pairs": []})
        for i, stmt in re.findall(r"\[(\d+)\] (.+)", prim_block):
            low = stmt.lower()
            if "alphanote" in low and "nightly" in low:
                return json.dumps({"same_fact_pairs": [[1, int(i)]]})
        return json.dumps({"same_fact_pairs": []})

    responses = {}
    responses["You are Veritas Planner."] = plan
    # the cross sub-question must be answerable from the local notes (its
    # search tokens hit alpha.md), or the cross pass gathers no evidence and
    # the paraphrased claim below is never extracted
    responses["You are Veritas CrossCheck Planner."] = json.dumps({
        "overview": "independent view",
        "subquestions": [{
            "text": "cross sub-question: does any source confirm AlphaNote "
                     "really processes JSON files nightly?",
            "rationale": "r"}],
    })
    responses["You are Veritas Researcher."] = json.dumps({
        "key_points": [], "conflicts": [], "uncertainties": []})
    responses[CLAIMS_SYSTEM] = claims_fn
    responses["You are Veritas Verifier."] = json.dumps({
        "verdict": "supported", "reason": "scripted", "better_statement": ""})
    responses["You are Veritas Synthesizer."] = "Scripted answer."
    responses["You are Veritas Conflict Detector."] = json.dumps({"contradicting_pairs": []})
    responses["You are Veritas Corroborator."] = corroborator
    llm = FakeLLM(responses)

    outdir = tmp_path / "out-sem"
    runner = Runner(llm=llm,
                    providers=build_providers([Surface.LOCAL], local_root=notes),
                    outdir=outdir)
    report = runner.run(Query("Compare AlphaNote and BetaNote",
                              surfaces=[Surface.LOCAL]))

    # the paraphrase is corroboration, NOT a third appended candidate
    statements = [c.statement for c in report.claims]
    assert len(report.claims) == 2
    assert not any("batch processing" in s for s in statements)
    corroborated = [c for c in report.claims if c.crosschecked]
    assert len(corroborated) == 1
    assert "nightly" in corroborated[0].statement
    assert corroborated[0].confidence in ("medium", "high")
    ledger = json.loads((outdir / "ledger.json").read_text())
    assert ledger["crosscheck"]["corroborated_semantic"] == 1
    assert ledger["crosscheck"]["corroborated"] >= 1


def test_d_query_contradiction_fires_through_crosspass(tmp_path: Path):
    """full-1 A4 (d1): a genuinely disputed topic fires the contradiction
    only when BOTH sides reach the detector. The grounded independent pass
    is the coverage half — its counter-evidence sub-question surfaces the
    other side, the claim is verified and appended, and the detector fires
    on the pair. The pair then lands on report-level conflicts AND the
    claims' own conflicts (visibility half, mirrored), so the benchmark
    D-class read fires. The append path must also keep ledger claim ids
    unique (cross-pass extraction restarts at c1)."""
    from veritas.llm import FakeLLM
    from veritas.pipeline.prompts import CLAIMS_SYSTEM, CONFLICT_DETECTOR_SYSTEM

    root = tmp_path / "notes"
    root.mkdir()
    (root / "bell.md").write_text(
        "Alexander Graham Bell alone invented the telephone in 1876. Bell "
        "filed US patent 174465 in February 1876 and was granted it in "
        "March 1876.\n" * 3)
    (root / "meucci.md").write_text(
        "Antonio Meucci invented the telephone, not Bell. A 2002 resolution "
        "of the United States House of Representatives honors Antonio "
        "Meucci's work on the telephone.\n" * 3)

    plan = json.dumps({"overview": "who invented the telephone",
        "subquestions": [
            {"text": "What did Alexander Graham Bell contribute to the "
                      "telephone?", "rationale": "r"}],
        "crosscheck_seed_note": "rival claimant angle"})

    def claims_fn(user: str) -> str:
        # the independent pass extracts the OTHER side of the dispute
        if "cross sub-question" in user:
            return json.dumps({"claims": [{
                "statement": "Antonio Meucci invented the telephone, not Bell.",
                "evidence_idx": [1]}], "noted_gaps": []})
        return json.dumps({"claims": [{
            "statement": "Alexander Graham Bell alone invented the telephone.",
            "evidence_idx": [1]}], "noted_gaps": []})

    responses = {}
    responses["You are Veritas Planner."] = plan
    # cross sub-question must be answerable from the local notes (its search
    # tokens hit meucci.md), or the cross pass gathers no evidence and the
    # counter-claim below is never extracted
    responses["You are Veritas CrossCheck Planner."] = json.dumps({
        "overview": "independent view",
        "subquestions": [{
            "text": "cross sub-question: do sources say Antonio Meucci — not "
                     "Bell alone — invented the telephone?",
            "rationale": "r"}],
    })
    responses[CLAIMS_SYSTEM] = claims_fn
    responses["You are Veritas Researcher."] = json.dumps({
        "key_points": [], "conflicts": [], "uncertainties": []})
    responses["You are Veritas Verifier."] = json.dumps({
        "verdict": "supported", "reason": "scripted", "better_statement": ""})
    responses["You are Veritas Synthesizer."] = "Scripted answer."
    responses[CONFLICT_DETECTOR_SYSTEM] = '{"contradicting_pairs": [[1, 2]]}'
    responses["You are Veritas Corroborator."] = json.dumps({"same_fact_pairs": []})
    llm = FakeLLM(responses)

    outdir = tmp_path / "out-d"
    runner = Runner(llm=llm,
                    providers=build_providers([Surface.LOCAL], local_root=root),
                    outdir=outdir)
    report = runner.run(Query("Who invented the telephone?",
                              surfaces=[Surface.LOCAL]))

    bell = "Alexander Graham Bell alone invented the telephone."
    meucci = "Antonio Meucci invented the telephone, not Bell."
    stmts = [c.statement for c in report.claims]
    # both sides asserted: one primary claim + one verified cross-pass claim
    assert len(report.claims) == 2
    assert set(stmts) == {bell, meucci}
    xc = next(c for c in report.claims if c.statement == meucci)
    assert "from the independent cross-check pass" in xc.note
    # detector pair on the report AND mirrored on both claims
    assert report.conflicts == [{"a": bell, "b": meucci,
                                "basis": "model-pairing"}]
    by_stmt = {c.statement: c for c in report.claims}
    assert by_stmt[bell].conflicts == [meucci]
    assert by_stmt[meucci].conflicts == [bell]
    # cross-pass extraction restarts ids at c1: the merged list must be
    # renumbered so every ledger claim id is unique
    assert {c.id for c in report.claims} == {"c1", "c2"}

    ledger = json.loads((outdir / "ledger.json").read_text())
    assert len({c_["id"] for c_ in ledger["claims"]}) == len(ledger["claims"])
    assert ledger["conflicts"] == [{"a": bell, "b": meucci,
                                    "basis": "model-pairing"}]
    lby = {c_["statement"]: c_ for c_ in ledger["claims"]}
    assert lby[bell]["conflicts"] == [meucci]
    assert lby[meucci]["conflicts"] == [bell]
    # the benchmark D-class read (bench/score.py) fires on this ledger
    from bench.score import compute_query_metrics
    gold = {"query_id": "d1-hermetic", "class": "D",
            "source_landscape": "test",
            "expected_claims": [
                {"statement": bell, "gold_label": "contested",
                 "confidence_class": "low", "note": ""},
                {"statement": meucci, "gold_label": "contested",
                 "confidence_class": "low", "note": ""}]}
    m = compute_query_metrics(ledger, gold)
    assert m["contradiction_fires_D"] == 1


def test_conflict_pairs_are_mirrored_on_the_claims(tmp_path: Path):
    """Detector pairs land on the claims themselves (Claim.conflicts), so the
    ledger shows per-claim conflicts and the benchmark D-class metric sees the
    fired contradiction wherever it reads (full-1 A4: D-fire flakiness is
    visibility + coverage; this is the visibility half)."""
    from veritas.llm import FakeLLM
    from veritas.pipeline.prompts import CONFLICT_DETECTOR_SYSTEM
    from veritas.schema import Claim, Evidence, Plan, Source, SubQuestion, Verdict

    def c(ident: str, stmt: str, url: str) -> Claim:
        return Claim(id=ident, statement=stmt,
                     evidence=[Evidence(source=Source(url=url, title=url),
                                        passage="p")],
                     verdict=Verdict.SUPPORTED, confidence="medium")

    a = c("c1", "Bell alone invented the telephone.", "https://a/x")
    b = c("c2", "Meucci invented the telephone, not Bell.", "https://b/y")
    llm = FakeLLM({CONFLICT_DETECTOR_SYSTEM: '{"contradicting_pairs": [[1, 2]]}'})
    plan = Plan(overview="x", subquestions=[SubQuestion(text="sq")])
    outdir = tmp_path / "o-con"
    runner = Runner(llm=llm, outdir=outdir)
    runner._finalize(Query("q"), plan, [a, b], [], {}, [a, b])
    assert a.conflicts == [b.statement]
    assert b.conflicts == [a.statement]
    ledger = json.loads((outdir / "ledger.json").read_text())
    assert ledger["conflicts"] == [{"a": a.statement, "b": b.statement,
                                    "basis": "model-pairing"}]
    by_stmt = {c_["statement"]: c_ for c_ in ledger["claims"]}
    assert by_stmt[a.statement]["conflicts"] == [b.statement]


def test_ledger_persists_conflict_pairs(tmp_path: Path):
    """Regression: contradiction pairs live on Report.conflicts and must be
    persisted in ledger.json (the benchmark's D-class metric reads them)."""
    from veritas.llm import FakeLLM
    from veritas.schema import Report

    runner = Runner(llm=FakeLLM({}), outdir=tmp_path / "out")
    pairs = [{"a": "Bell invented the telephone.",
              "b": "Meucci invented the telephone.",
              "resolved": False}]
    report = Report(query="q", answer="a", claims=[], gaps=[],
                    conflicts=pairs)
    runner._write_artifacts(report)
    ledger = json.loads((tmp_path / "out" / "ledger.json").read_text())
    assert ledger["conflicts"] == pairs


def test_full_mission_offline(tmp_path: Path):
    notes = make_notes(tmp_path)
    plan = json.dumps({"overview": "compare the two tools",
        "subquestions": [
            {"text": "What does AlphaNote do and what is its version?", "rationale": "r"},
            {"text": "Is BetaNote maintained?", "rationale": "r"}],
        "crosscheck_seed_note": "maintenance angle"})
    claims = json.dumps({"claims": [
        {"statement": "AlphaNote processes JSON files nightly.", "evidence_idx": [1]},
        {"statement": "BetaNote is unmaintained since 2023.", "evidence_idx": [1]},
    ], "noted_gaps": []})
    llm = scripted_llm(plan=plan, claims=claims)
    outdir = tmp_path / "out"
    runner = Runner(llm=llm, providers=build_providers([Surface.LOCAL], local_root=notes),
                    outdir=outdir)
    report = runner.run(Query("Compare AlphaNote and BetaNote", surfaces=[Surface.LOCAL]))

    # every claim verified supported, no duplicates
    assert len(report.claims) >= 2
    assert all(c.verdict is Verdict.SUPPORTED for c in report.claims)
    statements = [c.statement for c in report.claims]
    assert len(set(statements)) == len(statements)

    # artifacts written and parseable
    assert (outdir / "report.md").exists()
    ledger = json.loads((outdir / "ledger.json").read_text())
    assert ledger["query"] == "Compare AlphaNote and BetaNote"
    assert len(ledger["claims"]) == len(report.claims)
    assert "evidence" in ledger["claims"][0]
    # every ledger claim carries its source locator
    loc = (ledger["claims"][0]["evidence"][0]["source"].get("url")
           or ledger["claims"][0]["evidence"][0]["source"].get("path"))
    assert loc


def test_mission_with_no_evidence_reports_gaps(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    plan = json.dumps({"overview": "x", "subquestions": [
        {"text": "What color is the sky in this void?", "rationale": "r"}],
        "crosscheck_seed_note": "n/a"})
    llm = scripted_llm(plan=plan)
    runner = Runner(llm=llm, providers=build_providers([Surface.LOCAL], local_root=empty),
                    outdir=tmp_path / "o")
    report = runner.run(Query("color of sky", surfaces=[Surface.LOCAL]))
    assert report.claims == []
    assert any("no evidence found" in g for g in report.gaps)
    assert (tmp_path / "o" / "report.md").exists()


def test_evidence_with_empty_extraction_records_named_gap(tmp_path, monkeypatch):
    """A planned sub-question that fetched evidence but extracted nothing
    assertable must stay observable as a named gap — otherwise it vanishes
    from the ledger (no claim, no gap) and biases the A3 honest-failure
    denominator (Codex P2)."""
    import veritas.pipeline.runner as runner_mod
    notes = make_notes(tmp_path)
    plan = json.dumps({"overview": "x", "subquestions": [
        {"text": "What is AlphaNote's version?", "rationale": "r"}],
        "crosscheck_seed_note": "n/a"})
    llm = scripted_llm(plan=plan)
    monkeypatch.setattr(runner_mod, "extract_claims",
                        lambda llm_, sub, evidence, researcher=None:
                        ([], []))  # evidence found, nothing assertable
    runner = Runner(llm=llm,
                    providers=build_providers([Surface.LOCAL],
                                              local_root=notes),
                    outdir=tmp_path / "o-gap")
    report = runner.run(Query("AlphaNote version", surfaces=[Surface.LOCAL]))
    assert report.claims == []
    assert "no evidence found for: What is AlphaNote's version?" in report.gaps
    ledger = json.loads(
        (tmp_path / "o-gap" / "ledger.json").read_text())
    assert any("no evidence found for: What is AlphaNote's version?" in g
               for g in ledger["gaps"])


def test_crosscheck_can_be_disabled(tmp_path: Path):
    notes = make_notes(tmp_path)
    llm = scripted_llm()
    runner = Runner(llm=llm,
                    providers=build_providers([Surface.LOCAL], local_root=notes),
                    enable_crosscheck=False, outdir=tmp_path / "o2")
    report = runner.run(Query("t", surfaces=[Surface.LOCAL]))
    assert report.crosscheck == {}


def test_claims_are_deduped_across_subquestions(tmp_path: Path):
    """The same statement surfacing under two sub-questions verifies once."""
    notes = make_notes(tmp_path)
    plan = json.dumps({"overview": "x", "subquestions": [
        {"text": "AlphaNote facts", "rationale": "r"},
        {"text": "AlphaNote facts again", "rationale": "r"}],
        "crosscheck_seed_note": "n/a"})
    # the (scripted) model emits the same claim both times
    claims = json.dumps({"claims": [
        {"statement": "AlphaNote processes JSON files nightly.", "evidence_idx": [1]},
    ], "noted_gaps": []})
    llm = scripted_llm(plan=plan, claims=claims)
    runner = Runner(llm=llm, providers=build_providers([Surface.LOCAL], local_root=notes),
                    enable_crosscheck=False, outdir=tmp_path / "o3")
    report = runner.run(Query("t", surfaces=[Surface.LOCAL]))
    assert len(report.claims) == 1
    assert report.claims[0].id == "c1"
