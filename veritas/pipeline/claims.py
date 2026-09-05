"""Claim extraction: turn evidence into atomic, evidence-bound claims.

The accuracy guarantee lives here: the LLM proposes claims that *cite* evidence
by index; this module then enforces the binding deterministically —

* any claim whose evidence_idx is empty/out-of-range is rejected and logged as
  a gap (the model may not invent support);
* evidence objects are attached from the real bundle, not from model text;
* duplicate statements are merged;
* the claim's sub-question and quoted passages travel with it all the way to
  the report, so nothing is asserted without a retrievable origin.
"""

from __future__ import annotations

from ..llm import BaseLLM
from ..schema import Claim, Evidence, Verdict
from .research import evidence_bundle_label
from .prompts import CLAIMS_SYSTEM


def extract_claims(
    llm: BaseLLM,
    subquestion_text: str,
    evidence: list[Evidence],
    researcher: dict | None = None,
) -> tuple[list[Claim], list[str]]:
    """Returns (claims, noted_gaps). Claims carry the evidence they cite."""
    parts = [f"Sub-question: {subquestion_text}",
             f"Evidence passages:\n{evidence_bundle_label(evidence)}"]
    if researcher:
        if researcher.get("key_points"):
            parts.append("Researcher key points:\n- " + "\n- ".join(researcher["key_points"]))
        if researcher.get("uncertainties"):
            parts.append("Researcher noted uncertainties:\n- "
                         + "\n- ".join(researcher["uncertainties"]))
    parts.append("\nExtract claims now, citing evidence_idx exactly as numbered above.")
    data = llm.complete_json(CLAIMS_SYSTEM, "\n".join(parts))

    claims: list[Claim] = []
    gaps: list[str] = [g for g in data.get("noted_gaps", []) if isinstance(g, str) and g.strip()]
    seen: set[str] = set()
    for raw in data.get("claims", []):
        if not isinstance(raw, dict):
            continue
        statement = (raw.get("statement") or "").strip()
        if not statement:
            continue
        idxs = raw.get("evidence_idx")
        if not isinstance(idxs, list) or not idxs:
            gaps.append(f"claim without evidence was dropped: {statement[:160]}")
            continue
        bound: list[Evidence] = []
        for i in idxs:
            try:
                bound.append(evidence[int(i) - 1])
            except (ValueError, IndexError, TypeError):
                continue
        if not bound:
            gaps.append(f"claim citing out-of-range evidence was dropped: {statement[:160]}")
            continue
        norm = statement.lower()
        if norm in seen:
            continue
        seen.add(norm)
        claims.append(Claim(
            id=f"c{len(claims) + 1}",
            statement=statement,
            subquestion=subquestion_text,
            evidence=bound,
            verdict=Verdict.UNSUPPORTED,  # verification decides
        ))
    return claims, gaps
