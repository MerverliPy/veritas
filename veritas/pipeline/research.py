"""Planner + Researcher stages: decompose the query and gather raw evidence.

Research is deliberately *evidence-first*: each sub-question is searched
across the enabled surfaces, top sources are re-fetched for full text, and the
researcher LLM only summarises what the passages establish. The claim stage
(below) is what turns that into assertable statements, and it must cite
evidence by index — free-text support is not accepted anywhere.
"""

from __future__ import annotations

from ..llm import BaseLLM
from ..schema import Evidence, Plan, Query
from .prompts import (CROSSCHECK_PLANNER_SYSTEM, PLANNER_SYSTEM,
                      RESEARCHER_SYSTEM, subquestions_from_plan_json)


def make_plan(llm: BaseLLM, query: Query) -> Plan:
    """Decompose a query into sub-questions (with cross-check seed note)."""
    surfaces = ", ".join(query.surface_names())
    user = (
        f"Research request: {query.text}\n"
        f"Research surfaces enabled: {surfaces}\n"
        + (f"User-provided starting sources: {', '.join(query.sources)}\n" if query.sources else "")
        + "\nDecompose into sub-questions now."
    )
    data = llm.complete_json(PLANNER_SYSTEM, user)
    plan = subquestions_from_plan_json(data, query.text)
    if not plan.crosscheck_seed_note:
        plan.crosscheck_seed_note = ("Same request, but prioritise the sources the first "
                                     "plan under-used, and look for counter-evidence.")
    return plan


def make_crosscheck_plan(llm: BaseLLM, query: Query, seed_note: str) -> Plan:
    """Second, independent decomposition for the cross-check pass."""
    user = (
        f"Original research request: {query.text}\n"
        f"Alternative framing to explore: {seed_note}\n"
        f"Surfaces enabled: {', '.join(query.surface_names())}\n"
        "\nRe-plan from this independent angle."
    )
    data = llm.complete_json(CROSSCHECK_PLANNER_SYSTEM, user)
    return subquestions_from_plan_json(data, query.text)


def evidence_bundle_label(evidence: list[Evidence], start: int = 1) -> str:
    """Render numbered evidence for an LLM prompt, e.g.
    [1] Title — https://...\n    "quoted passage" """
    lines = []
    for i, ev in enumerate(evidence, start=start):
        src = ev.source
        lines.append(f"[{i}] {src.title or src.locator()}")
        lines.append(f"    source: {src.locator()}")
        lines.append(f'    passage: "{ev.passage}"')
    return "\n".join(lines)


def research_subquestion(
    llm: BaseLLM,
    providers: list,
    subquestion_text: str,
    *,
    limit: int = 8,
    fetch_top: int = 3,
) -> tuple[list[Evidence], list[str]]:
    """Gather evidence for one sub-question across all providers.

    Returns (evidence, warnings). Evidence order is the prompt order, so
    indices stay stable for claim citation.
    """
    evidence: list[Evidence] = []
    warnings: list[str] = []
    seen_locators: set[str] = set()

    for provider in providers:
        try:
            hits = provider.search(subquestion_text, limit=max(3, limit // max(1, len(providers))))
        except Exception as e:  # connectors promise not to raise, but be safe
            warnings.append(f"{provider.surface.value}: {type(e).__name__}: {e}")
            continue
        warnings.extend(provider.warnings)
        provider.warnings.clear()
        for ev in hits:
            key = ev.source.locator()
            if key in seen_locators:
                continue
            seen_locators.add(key)
            evidence.append(ev)

    evidence = evidence[:limit]

    # Re-fetch full text of the strongest sources so claim extraction and
    # verification read the actual document, not just snippets.
    fetched: list[Evidence] = []
    for ev in evidence:
        if len(fetched) >= fetch_top:
            break
        provider = None
        for p in providers:
            if p.surface == ev.source.surface:
                provider = p
                break
        if provider is None:
            continue
        try:
            text = provider.fetch(ev.source)
        except Exception as e:
            warnings.append(f"fetch {ev.source.locator()}: {e}")
            text = None
        if text:
            fetched.append(Evidence(
                source=ev.source,
                passage=text[:1600] if ev.source.surface.value == "web" else text[:2400],
                kind="fetch",
            ))
    if fetched:
        evidence = fetched + [e for e in evidence if e not in fetched]
    # dedupe by locator, preferring the longer (fetch) passage
    by_loc: dict[str, Evidence] = {}
    for ev in evidence:
        loc = ev.source.locator()
        if loc not in by_loc or len(ev.passage) > len(by_loc[loc].passage):
            by_loc[loc] = ev
    return list(by_loc.values())[: limit + fetch_top], warnings


def researcher_notes(llm: BaseLLM, subquestion_text: str, evidence: list[Evidence]) -> dict:
    """LLM summarises what the gathered passages establish (no new facts)."""
    user = (
        f"Sub-question: {subquestion_text}\n\n"
        f"Evidence passages:\n{evidence_bundle_label(evidence)}\n"
        "\nExtract key_points, conflicts and uncertainties."
    )
    return llm.complete_json(RESEARCHER_SYSTEM, user)
