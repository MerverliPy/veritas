"""Verification: re-check every claim against the retrieved source text.

Two-stage per claim:

1. *re-fetch* — pull the full text behind each cited source again (web pages
   re-downloaded, file/code sources re-read at their exact ``#L#-#`` anchor),
   so the judge reads what the source *actually says*, not a snippet.
2. *judge* — an LLM verdict: supported / partial / contradicted / unsupported,
   with a concrete reason and (for partial/contradicted) a corrected
   ``better_statement`` grounded in the source text.

Confidence mapping (deterministic, no model judgement):
  supported            -> medium   (bumped to high only by cross-check later)
  partial              -> low      (claim statement is corrected if better_statement)
  contradicted         -> low      (kept as a conflict, never asserted as fact)
  unsupported          -> unsupported (reported under "not established")
"""

from __future__ import annotations

from ..llm import BaseLLM
from ..schema import Claim, Evidence, Source, Surface, Verdict
from .prompts import VERIFY_SYSTEM


def _refetch(provider_by_surface: dict, source: Source) -> tuple[str, bool]:
    provider = provider_by_surface.get(source.surface)
    if provider is None:
        return "", False
    try:
        text = provider.fetch(source)
    except Exception:
        return "", False
    if not text:
        return "", False
    cap = 6000 if source.surface is Surface.WEB else 8000
    return text[:cap], True


def verify_claim(
    llm: BaseLLM,
    claim: Claim,
    provider_by_surface: dict,
) -> Claim:
    """Return a copy of the claim with verdict/confidence set."""
    judge_input: list[str] = [f"CLAIM: {claim.statement}"]
    for i, ev in enumerate(claim.evidence, start=1):
        full, ok = _refetch(provider_by_surface, ev.source)
        if ok:
            judge_input.append(
                f"\nSOURCE [{i}] {ev.source.title} — {ev.source.locator()} (refetched):\n{full}"
            )
        else:
            judge_input.append(
                f"\nSOURCE [{i}] {ev.source.title} — {ev.source.locator()} (refetch failed, "
                f"only quoted passage available):\n{ev.passage[:1200]}"
            )
    judge_input.append(
        "\nGive your verdict on whether the cited sources support this claim as stated."
    )
    data = llm.complete_json(VERIFY_SYSTEM, "\n".join(judge_input))

    raw_verdict = (data.get("verdict") or "").strip().lower()
    try:
        verdict = Verdict(raw_verdict)
    except ValueError:
        verdict = Verdict.UNSUPPORTED

    reason = (data.get("reason") or "").strip()
    better = (data.get("better_statement") or "").strip()
    _apply_support_flags(claim, data, verdict)

    claim.verdict = verdict
    claim.note = reason
    if verdict is Verdict.SUPPORTED:
        claim.confidence = "medium"
    elif verdict is Verdict.PARTIAL:
        claim.confidence = "low"
        if better:
            claim.note = f"{reason} Corrected statement: {better}"
    elif verdict is Verdict.CONTRADICTED:
        claim.confidence = "low"
        if better:
            claim.note = f"{reason} Source states: {better}"
    else:  # unsupported
        claim.confidence = "unsupported"
        if not reason:
            claim.note = "no retrievable evidence supports or refutes this claim"
    return claim


def _apply_support_flags(claim: Claim, data: dict, verdict: Verdict) -> None:
    """Annotate which evidence entries actually support the claim.

    The verifier names ``supporting_sources`` (1-based indices into the cited
    sources); every other entry is marked non-supporting so a bundled but
    irrelevant locator can never be used as independent corroboration.
    FAIL-CLOSED: when the field is absent, empty or malformed, NO entry is
    treated as supporting — the response schema is not enforced on the model,
    so a missing field must never confer verified per-locator support
    (Codex round-8 P1)."""
    raw = data.get("supporting_sources")
    if not isinstance(raw, list) or not raw:
        if verdict is Verdict.SUPPORTED and claim.evidence:
            for ev in claim.evidence:
                ev.supports = False
        return
    idx: set[int] = set()
    for i in raw:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(claim.evidence):
            idx.add(i)
    for k, ev in enumerate(claim.evidence, start=1):
        ev.supports = k in idx
