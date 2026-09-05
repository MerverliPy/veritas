"""Research-stage behaviour: term-windowed fetches + near-empty page skip.

These guard the regression where a fetch's first N chars were navigation
boilerplate, so claim extraction saw titles and menus instead of content.
"""

from __future__ import annotations

from veritas import Evidence, FakeLLM, Source, Surface
from veritas.connectors.base import Provider
from veritas.pipeline.prompts import RESEARCHER_SYSTEM
from veritas.pipeline.research import research_subquestion, _fetched_passage


class StubWeb(Provider):
    surface = Surface.WEB

    def __init__(self, page_text: str):
        super().__init__()
        self.page_text = page_text

    def search(self, query, limit=8):
        return [Evidence(source=Source(url="https://stub.example/page",
                                       title="Stub page"), passage="search snippet")]

    def fetch(self, source):
        return self.page_text


NAVY = "Menu | Home | About | Products | Contact | Privacy | Terms" + " nav " * 30


def test_fetch_windows_around_subquestion_terms():
    # answer buried far below a nav header — truncation would miss it
    page = NAVY + "\n\n" + ("boilerplate paragraph. " * 8) + "\n\n" + \
        "EternalBlue spread via SMBv1 on port 445 allowing remote code execution."
    providers = [StubWeb(page)]
    evidence, _ = research_subquestion(FakeLLM({}), providers,
                                       "how did EternalBlue spread via SMBv1?",
                                       limit=8, fetch_top=3)
    assert len(evidence) == 1
    assert evidence[0].kind == "fetch"
    assert "SMBv1" in evidence[0].passage
    assert "EternalBlue" in evidence[0].passage
    assert len(evidence[0].passage) <= 2000


def test_near_empty_page_is_skipped_with_warning():
    providers = [StubWeb("<h1>Just a title</h1><nav>Home About</nav>")]
    evidence, warnings = research_subquestion(
        FakeLLM({}), providers, "anything about widgets", limit=8, fetch_top=3)
    # only the original search hit survives; the fetch was skipped
    assert len(evidence) == 1
    assert evidence[0].kind == "search"
    assert any("near-empty page skipped" in w for w in warnings)


def test_fetched_passage_returns_head_when_no_terms_match():
    text = "unrelated content without the question's words " * 40
    passage = _fetched_passage(text, ["widgets", "nightly"])
    assert len(passage) <= 2000
    assert passage == text[:2000]


def test_researcher_notes_prompt_numbers_evidence_in_order():
    evidence = [
        Evidence(source=Source(url="https://a/x", title="A"), passage="aa"),
        Evidence(source=Source(url="https://b/y", title="B"), passage="bb"),
    ]
    llm = FakeLLM({RESEARCHER_SYSTEM: '{"key_points": ["ok"], "conflicts": [], "uncertainties": []}'})
    llm.complete_json(RESEARCHER_SYSTEM, "unused")  # priming not needed
    from veritas.pipeline.research import researcher_notes
    out = researcher_notes(llm, "the question", evidence)
    assert out["key_points"] == ["ok"]
    prompt = llm.calls[-1][1]
    assert "[1] A" in prompt and "[2] B" in prompt
    assert "https://a/x" in prompt
