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
