"""Shared test helpers: scripted FakeLLM builds + tiny fixture trees."""

from __future__ import annotations

import json

from veritas import FakeLLM


def scripted_llm(
    *,
    plan: str | None = None,
    cross_plan: str | None = None,
    researcher: str | None = None,
    claims: str | None = None,
    verdict: str | None = None,
    synth: str | None = None,
) -> FakeLLM:
    """FakeLLM with sensible JSON defaults; override any stage by passing raw
    JSON strings (verdict may be a user-text callable)."""
    def _json(obj: dict) -> str:
        return json.dumps(obj)

    responses = {}
    responses["You are Veritas Planner."] = plan or _json({
        "overview": "test plan",
        "subquestions": [{"text": "test sub-question", "rationale": "r"}],
        "crosscheck_seed_note": "look from the other side",
    })
    responses["You are Veritas CrossCheck Planner."] = cross_plan or _json({
        "overview": "independent view",
        "subquestions": [{"text": "cross sub-question", "rationale": "r"}],
    })
    responses["You are Veritas Researcher."] = researcher or _json({
        "key_points": [], "conflicts": [], "uncertainties": []})
    responses["You are Veritas Claim Extractor."] = claims or _json({
        "claims": [{"statement": "sample claim", "evidence_idx": [1]}],
        "noted_gaps": []})
    responses["You are Veritas Verifier."] = verdict or _json({
        "verdict": "supported", "reason": "scripted", "better_statement": ""})
    responses["You are Veritas Synthesizer."] = synth or "Scripted answer."
    responses["You are Veritas Conflict Detector."] = json.dumps({"contradicting_pairs": []})
    return FakeLLM(responses)
