"""Schema round-trips + LLM plumbing edge cases."""

from __future__ import annotations

import json

import pytest

from veritas import Evidence, Source, Surface
from veritas.llm import FakeLLM, LLMError, extract_json


def test_source_round_trip_with_anchor():
    s = Source(path="a.py", title="t", surface=Surface.CODE, anchor="L10-L12")
    d = s.to_dict()
    back = Source.from_dict(d)
    assert back == s
    assert back.locator() == "a.py#L10-L12"


def test_web_source_round_trip():
    s = Source(url="https://x.dev/p", title="page")
    assert Source.from_dict(s.to_dict()) == s


def test_evidence_and_claim_round_trip():
    ev = Evidence(source=Source(url="https://x.dev"), passage="quote")
    c = {"id": "c7", "statement": "st", "subquestion": "sq",
         "evidence": [ev.to_dict()], "verdict": "supported",
         "confidence": "medium", "crosschecked": True,
         "conflicts": [], "note": "n"}
    from veritas import Claim
    claim = Claim.from_dict(c)
    assert claim.id == "c7"
    assert claim.verdict.value == "supported"
    assert claim.evidence[0].passage == "quote"


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('prefix text {"b": [1, 2]} suffix', {"b": [1, 2]}),
    ('{"outer": {"inner": true}} trailing', {"outer": {"inner": True}}),
])
def test_extract_json_variants(raw, expected):
    assert extract_json(raw) == expected


def test_extract_json_rejects_non_object():
    with pytest.raises(LLMError):
        extract_json("[1, 2, 3]")
    with pytest.raises(LLMError):
        extract_json("nothing to see")


def test_fake_llm_prefix_selection_and_fallback():
    llm = FakeLLM({"AAA": "1", "*": "fallback"})
    assert llm.complete("AAA: anything", "u") == "1"
    assert llm.complete("ZZZ no match", "u") == "fallback"
    with pytest.raises(LLMError):
        FakeLLM({}).complete("no keys", "u")


def test_deepseek_client_requires_key(monkeypatch):
    from veritas import DeepSeekClient
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    c = DeepSeekClient(api_key="")
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        c.complete("s", "u")


def test_surface_enum_values():
    assert Surface("web") is Surface.WEB
    assert [s.value for s in [Surface.WEB, Surface.LOCAL, Surface.CODE]] == \
        ["web", "local", "code"]
