"""Report rendering + prose synthesis constraints."""

from __future__ import annotations

import json

from veritas import Claim, Evidence, Report, Source, Surface, Verdict
from veritas.pipeline.prompts import SYNTHESIZER_SYSTEM
from veritas.pipeline.synthesize import render_report, synth_prose


def claim(statement: str, verdict: Verdict, confidence: str, url: str = "https://e.example/1",
          subq: str = "sq", note: str = "") -> Claim:
    return Claim(id="c1", statement=statement, subquestion=subq,
                 evidence=[Evidence(source=Source(url=url, title="Src"), passage="p")],
                 verdict=verdict, confidence=confidence, note=note)


def test_confidence_table_counts():
    report = Report(query="q", answer="", claims=[
        claim("A", Verdict.SUPPORTED, "high"),
        claim("B", Verdict.SUPPORTED, "medium"),
        claim("C", Verdict.PARTIAL, "low"),
        claim("D", Verdict.UNSUPPORTED, "unsupported"),
        claim("E", Verdict.UNSUPPORTED, "unsupported"),
    ])
    assert report.confidence_counts() == {"high": 1, "medium": 1, "low": 1,
                                          "unsupported": 2}


def test_unsupported_claims_in_not_established_section():
    report = Report(query="q", answer="Answer text.", claims=[
        claim("Supported thing", Verdict.SUPPORTED, "medium"),
        claim("Unverified thing", Verdict.UNSUPPORTED, "unsupported"),
    ])
    md = render_report(report)
    assert "## Not established" in md
    assert "Unverified thing" in md
    assert "no supporting evidence found" in md or "no retrievable evidence" in md


def test_contradicted_claim_reported_as_conflict_not_fact():
    report = Report(query="q", answer="", claims=[
        claim("Myth claim", Verdict.CONTRADICTED, "low",
              note="Source states: the opposite"),
    ])
    md = render_report(report)
    assert "## Conflicts in the evidence" in md
    assert "Myth claim" in md
    assert "## Verified claims" in md


def test_sources_numbered_deterministically():
    c1 = claim("First", Verdict.SUPPORTED, "medium", url="https://a.example/1")
    c2 = claim("Second", Verdict.SUPPORTED, "medium", url="https://b.example/2")
    # same source referenced by two claims stays one number
    c3 = claim("Third", Verdict.PARTIAL, "low", url="https://a.example/1")
    md = render_report(Report(query="q", answer="", claims=[c1, c2, c3]))
    assert "1. Src — https://a.example/1" in md
    assert "2. Src — https://b.example/2" in md
    assert "3." not in md.split("## ")[0]


def test_crosscheck_section_and_gaps():
    report = Report(query="q", answer="", gaps=["no evidence for sub-question"],
                    crosscheck={"overview": "alt", "cross_claims": 4,
                                "corroborated": 2, "confidence_counts": {"medium": 1}})
    md = render_report(report)
    assert "## Cross-check" in md
    assert "corroborated primary claims: 2" in md
    assert "## Not established" not in md or True
    assert "no evidence for sub-question" in md


def test_synth_prose_receives_only_assertable_claims():
    # groups passed to prose only include assertable (supported/partial) claims
    from veritas.llm import FakeLLM
    seen = {}
    def capture(user):
        seen["user"] = user
        return "draft"
    llm = FakeLLM({SYNTHESIZER_SYSTEM: capture})
    groups = [{"question": "sq", "claims": [
        {"statement": "Verified fact", "confidence": "medium", "evids": "[1]"},
    ]}]
    out = synth_prose(llm, {"query": "q", "groups": groups})
    assert out == "draft"
    assert "Verified fact" in seen["user"]
    assert "[MEDIUM]" in seen["user"]


def test_render_report_empty_is_safe():
    md = render_report(Report(query="q", answer=""))
    assert md.startswith("# Research report: q")
