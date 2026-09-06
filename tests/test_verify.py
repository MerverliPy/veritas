"""Verification: verdict -> confidence mapping + refetch behaviour."""

from __future__ import annotations

import json
from pathlib import Path

from veritas import Claim, Evidence, FakeLLM, Source, Surface, Verdict
from veritas.connectors import build_providers
from veritas.pipeline.prompts import VERIFY_SYSTEM
from veritas.pipeline.verify import verify_claim


def claim_with_file_source(tmp_path: Path, text: str) -> Claim:
    f = tmp_path / "doc.md"
    f.write_text(text)
    source = Source(path="doc.md", title="doc.md", surface=Surface.LOCAL, anchor="L1-L2")
    ev = Evidence(source=source, passage=text[:200], kind="file")
    return Claim(id="c1", statement="the file supports X", evidence=[ev])


def test_supported_maps_to_medium(tmp_path: Path):
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps(
        {"verdict": "supported", "reason": "text agrees", "better_statement": ""})})
    provs = build_providers([Surface.LOCAL], local_root=tmp_path)
    c = claim_with_file_source(tmp_path, "X is true and well documented.\n")
    verify_claim(llm, c, {Surface.LOCAL: provs[0]})
    assert c.verdict is Verdict.SUPPORTED
    assert c.confidence == "medium"
    assert c.note == "text agrees"


def _two_file_claim(tmp_path: Path) -> Claim:
    (tmp_path / "a.md").write_text("Widgets process JSON nightly.\n")
    (tmp_path / "b.md").write_text("Widgets process JSON nightly.\n")
    evs = [
        Evidence(source=Source(path="a.md", title="a.md", surface=Surface.LOCAL,
                               anchor="L1"), passage="Widgets process JSON nightly.",
                 kind="file"),
        Evidence(source=Source(path="b.md", title="b.md", surface=Surface.LOCAL,
                               anchor="L1"), passage="Widgets process JSON nightly.",
                 kind="file"),
    ]
    return Claim(id="c1", statement="Widgets process JSON nightly", evidence=evs)


def test_supporting_sources_annotate_evidence(tmp_path: Path):
    """verify_claim marks exactly the model-named sources as supporting; a
    bundled-but-irrelevant locator can then never drive a promotion."""
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps({
        "verdict": "supported", "reason": "text agrees",
        "better_statement": "", "supporting_sources": [2]})})
    provs = build_providers([Surface.LOCAL], local_root=tmp_path)
    c = _two_file_claim(tmp_path)
    verify_claim(llm, c, {Surface.LOCAL: provs[0]})
    assert c.evidence[0].supports is False
    assert c.evidence[1].supports is True


def test_supporting_sources_absent_fails_closed(tmp_path: Path):
    """A supported verdict with the field absent/malformed confers NO
    per-locator support (the response schema is not enforced on the model, so
    a missing field must not enable promotion via a bundled locator)."""
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps({
        "verdict": "supported", "reason": "text agrees",
        "better_statement": ""})})
    provs = build_providers([Surface.LOCAL], local_root=tmp_path)
    c = _two_file_claim(tmp_path)
    verify_claim(llm, c, {Surface.LOCAL: provs[0]})
    assert c.evidence[0].supports is False
    assert c.evidence[1].supports is False


def test_supporting_sources_empty_marks_none_supporting(tmp_path: Path):
    """supported verdict naming NO supporting source: no locator may drive a
    later promotion."""
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps({
        "verdict": "supported", "reason": "text agrees",
        "better_statement": "", "supporting_sources": []})})
    provs = build_providers([Surface.LOCAL], local_root=tmp_path)
    c = _two_file_claim(tmp_path)
    verify_claim(llm, c, {Surface.LOCAL: provs[0]})
    assert c.evidence[0].supports is False
    assert c.evidence[1].supports is False


def test_nonsupported_verdict_clears_support_flags(tmp_path: Path):
    """partial/contradicted/unsupported claims must not leave their evidence
    reading as supporting in the ledger when the field is absent (Codex
    round-9 P2)."""
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps({
        "verdict": "partial", "reason": "says most not all",
        "better_statement": "most widgets work"})})
    provs = build_providers([Surface.LOCAL], local_root=tmp_path)
    c = _two_file_claim(tmp_path)
    verify_claim(llm, c, {Surface.LOCAL: provs[0]})
    assert c.verdict is Verdict.PARTIAL
    assert c.evidence[0].supports is False
    assert c.evidence[1].supports is False


def test_partial_low_with_correction(tmp_path: Path):
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps(
        {"verdict": "partial", "reason": "says most not all",
         "better_statement": "most widgets work"})})
    provs = build_providers([Surface.LOCAL], local_root=tmp_path)
    c = claim_with_file_source(tmp_path, "Most widgets work.\n")
    verify_claim(llm, c, {Surface.LOCAL: provs[0]})
    assert c.verdict is Verdict.PARTIAL
    assert c.confidence == "low"
    assert "Corrected statement: most widgets work" in c.note


def test_contradicted_low_never_assertable(tmp_path: Path):
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps(
        {"verdict": "contradicted", "reason": "source says otherwise",
         "better_statement": "the opposite holds"})})
    provs = build_providers([Surface.LOCAL], local_root=tmp_path)
    c = claim_with_file_source(tmp_path, "The opposite holds.\n")
    verify_claim(llm, c, {Surface.LOCAL: provs[0]})
    assert c.verdict is Verdict.CONTRADICTED
    assert c.confidence == "low"
    assert "Source states: the opposite holds" in c.note


def test_unsupported_defaults_when_no_reason(tmp_path: Path):
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps(
        {"verdict": "unsupported", "reason": "", "better_statement": ""})})
    provs = build_providers([Surface.LOCAL], local_root=tmp_path)
    c = claim_with_file_source(tmp_path, "unrelated content\n")
    verify_claim(llm, c, {Surface.LOCAL: provs[0]})
    assert c.verdict is Verdict.UNSUPPORTED
    assert c.confidence == "unsupported"
    assert "no retrievable evidence" in c.note


def test_bad_verdict_string_falls_back_to_unsupported(tmp_path: Path):
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps(
        {"verdict": "definitely", "reason": "weird output", "better_statement": ""})})
    provs = build_providers([Surface.LOCAL], local_root=tmp_path)
    c = claim_with_file_source(tmp_path, "anything\n")
    verify_claim(llm, c, {Surface.LOCAL: provs[0]})
    assert c.verdict is Verdict.UNSUPPORTED


def test_refetch_failure_still_judges_on_quoted_passage(tmp_path: Path):
    # provider raises/fails fetch -> judge must still get the quoted passage
    class Boom:
        surface = Surface.LOCAL
        def fetch(self, source):  # simulates network failure
            return None
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps(
        {"verdict": "supported", "reason": "passage suffices", "better_statement": ""})})
    f = tmp_path / "gone.md"
    f.write_text("X is true.\n")
    source = Source(path="gone.md", title="gone", surface=Surface.LOCAL, anchor="L1")
    ev = Evidence(source=source, passage="X is true.", kind="file")
    c = Claim(id="c1", statement="X is true", evidence=[ev])
    verify_claim(llm, c, {Surface.LOCAL: Boom()})
    assert c.verdict is Verdict.SUPPORTED
    user_prompt = llm.calls[-1][1]
    assert "refetch failed" in user_prompt
    assert "X is true." in user_prompt


def test_web_source_gets_refetched_and_capped(tmp_path: Path):
    class FakeWeb:
        surface = Surface.WEB
        def fetch(self, source):
            return "T" * 30_000  # far over cap
    llm = FakeLLM({VERIFY_SYSTEM: json.dumps(
        {"verdict": "supported", "reason": "ok", "better_statement": ""})})
    ev = Evidence(source=Source(url="https://example.com/a", title="a"), passage="short")
    c = Claim(id="c1", statement="big page claim", evidence=[ev])
    verify_claim(llm, c, {Surface.WEB: FakeWeb()})
    prompt = llm.calls[-1][1]
    assert len(prompt) < 12_000  # cap applied before prompting
