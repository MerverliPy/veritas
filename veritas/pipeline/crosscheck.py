"""Independent cross-check + reconciliation + contradiction detection.

A second, independently-planned research pass runs over the same request
(different decomposition / angle). Its claims are reconciled with the primary
run's claims *deterministically*:

* corroboration — a cross-run claim sharing most significant tokens with a
  primary claim is independent agreement. If the primary is ``supported`` and
  its cited sources differ from the cross claim's, confidence rises to
  ``high``. Token matching is conservative by design: a wrong 'agreement' is
  worse than none.
* candidates — cross-run claims with no primary counterpart are *candidates*:
  they are put through the same LLM verifier as primary claims (never trusted
  unchecked) and appended with a note marking their origin.
* conflicts — semantic contradictions are detected by ONE dedicated LLM pass
  over the final assertable claims (``detect_contradictions``). The model
  proposes index pairs only; the module validates indices against the real
  claim list, so it cannot invent a claim to argue with.

The second plan is shown the first pass's sub-questions (factual ground) so it
re-derives the same key facts from different sources instead of drifting onto a
wholly disjoint or counterfactual angle (see ``make_crosscheck_plan``). And
because generative phrasing restates the same fact in different words, one
post-verification LLM pass (``corroborate_from_semantic``) recognises
same-fact agreements the token matcher missed — a verified-supported
independent claim may corroborate a primary claim, never an unchecked one.
"""

from __future__ import annotations

import re
from collections import Counter

from ..llm import BaseLLM
from ..schema import Claim, Evidence, Query, Verdict
from .claims import extract_claims
from .prompts import (CONFLICT_DETECTOR_SYSTEM, CORROBORATOR_SYSTEM)
from .research import make_crosscheck_plan, research_subquestion, researcher_notes
from .verify import verify_claim


def _sig_tokens(statement: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{4,}", statement.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_crosscheck(
    llm: BaseLLM,
    query: Query,
    providers,
    primary: list[Claim],
    seed_note: str,
    *,
    subquestion_limit: int = 3,
    evidence_limit: int = 6,
    primary_subquestions: list[str] | None = None,
) -> dict:
    """Run the independent pass; verified candidates are appended to
    ``primary`` (the runner's own claim list, mutated in place).

    ``primary_subquestions`` (the first pass's own sub-question texts) guide
    the independent plan so it re-derives the same key facts from different
    sources (corroboration) and actively probes for counter-evidence
    (contradiction)."""
    original_primary = list(primary)  # snapshot before candidates are appended
    plan = make_crosscheck_plan(llm, query, seed_note,
                                primary_subquestions=primary_subquestions)
    cross_claims: list[Claim] = []
    gaps: list[str] = []
    for sub in plan.subquestions[:subquestion_limit]:
        evidence, _warnings = research_subquestion(
            llm, providers, sub.text, limit=evidence_limit, fetch_top=2)
        if not evidence:
            continue
        notes = researcher_notes(llm, sub.text, evidence)
        claims, g = extract_claims(llm, sub.text, evidence, researcher=notes)
        cross_claims.extend(claims)
        gaps.extend(g)

    summary = reconcile(primary, cross_claims, plan.overview, gaps)
    candidates: list[Claim] = summary.pop("candidates", [])

    # candidates get the same verification treatment as primary claims
    provider_by_surface = {p.surface: p for p in providers}
    for cand in candidates:
        try:
            verify_claim(llm, cand, provider_by_surface)
        except Exception:
            cand.verdict = Verdict.UNSUPPORTED
            cand.confidence = "unsupported"
        origin = cand.note
        cand.note = (origin + " " if origin else "") + \
            "(from the independent cross-check pass)"

    # Semantic corroboration: a VERIFIED-supported independent claim may
    # corroborate a primary claim whose paraphrase the token matcher missed.
    # Promotion to high still needs a different source locator (schema
    # 'high' = independently corroborated, consistent evidence). Matched
    # cross claims corroborate an existing claim, so they are not appended
    # as new candidates.
    eligible_cross = [c for c in candidates
                      if c.verdict is Verdict.SUPPORTED]
    eligible_primary = [c for c in original_primary
                        if c.verdict is Verdict.SUPPORTED
                        and c.confidence == "medium"
                        and not c.crosschecked]
    sem_flags, sem_promos, sem_pairs = corroborate_from_semantic(
        llm, eligible_primary, eligible_cross)
    consumed = {id(xc) for _pc, xc in sem_pairs}

    appended = 0
    for cand in candidates:
        if id(cand) in consumed:
            continue
        primary.append(cand)
        appended += 1
    summary["appended"] = appended
    summary["corroborated_semantic"] = sem_flags
    summary["promoted_semantic"] = sem_promos
    if sem_flags:
        summary["corroborated"] = (summary.get("corroborated") or 0) + sem_flags
    summary["cross_gaps"] = gaps
    summary["confidence_counts"] = dict(Counter(c.confidence for c in primary))
    # Appended candidates carry ids from the cross-pass extraction, which
    # restarts at c1 per pass (and per sub-question) — renumber the whole
    # list so every ledger claim id is unique across the mission (schema
    # contract: ids identify claims; duplicate ids would silently merge for
    # any consumer keying by id).
    for i, c in enumerate(primary, start=1):
        c.id = f"c{i}"
    return summary


def reconcile(
    primary: list[Claim],
    cross: list[Claim],
    cross_overview: str = "",
    cross_gaps: list[str] | None = None,
) -> dict:
    """Deterministic corroboration + candidate selection. See module doc."""
    matched_primary: set[str] = set()
    candidates: list[Claim] = []
    for xc in cross:
        xt = _sig_tokens(xc.statement)
        if not xt:
            candidates.append(xc)
            continue
        best: tuple[float, Claim | None] = (0.0, None)
        for pc in primary:
            if pc.confidence == "unsupported":
                continue  # do not let a claim 'agree' with a non-claim
            sim = _jaccard(xt, _sig_tokens(pc.statement))
            if sim > best[0]:
                best = (sim, pc)
        sim, pc = best
        if pc is not None and sim >= 0.5:
            matched_primary.add(pc.id)
            pc.crosschecked = True
            if pc.verdict is Verdict.SUPPORTED and pc.confidence == "medium":
                pc_locs = {e.source.locator() for e in pc.evidence}
                xc_locs = {e.source.locator() for e in xc.evidence}
                if pc_locs and pc_locs != xc_locs:
                    pc.confidence = "high"
        else:
            candidates.append(xc)

    return {
        "overview": cross_overview,
        "cross_claims": len(cross),
        "corroborated": len(matched_primary),
        "candidates": candidates,
        "cross_gaps": cross_gaps or [],
        "confidence_counts": dict(Counter(c.confidence for c in primary)),
    }


def semantic_corroborate(
    llm: BaseLLM,
    primary: list[Claim],
    cross: list[Claim],
) -> list[tuple[Claim, Claim]]:
    """One LLM pass proposes same-fact (cross -> primary) pairs the token
    matcher missed because generative phrasing restates a fact in different
    words. The model proposes index pairs only; indices are validated against
    the real lists (deduped, in-range), so it cannot invent a claim to agree
    with. Returns [] when the model cannot be asked (hermetic FakeLLM,
    outage) — corroboration then rests on the deterministic token matcher
    alone (conservative fallback, never a mission failure)."""
    if not primary or not cross:
        return []
    user = (
        "Primary claims (verified):\n"
        + "\n".join(f"[{i}] {c.statement}"
                     for i, c in enumerate(primary, start=1))
        + "\n\nIndependent-pass claims (verified):\n"
        + "\n".join(f"[{i}] {c.statement}"
                     for i, c in enumerate(cross, start=1))
        + "\n\nReturn same_fact_pairs [cross_index, primary_index] for "
          "claims that state the same fact."
    )
    try:
        data = llm.complete_json(CORROBORATOR_SYSTEM, user)
    except Exception:
        return []
    out: list[tuple[Claim, Claim]] = []
    used_primary: set[int] = set()
    used_cross: set[int] = set()
    for raw in data.get("same_fact_pairs", []):
        if not isinstance(raw, list) or len(raw) != 2:
            continue
        try:
            x_i, p_i = int(raw[0]), int(raw[1])
        except (ValueError, TypeError):
            continue
        if not (1 <= x_i <= len(cross) and 1 <= p_i <= len(primary)):
            continue
        if x_i in used_cross or p_i in used_primary:
            continue
        used_cross.add(x_i)
        used_primary.add(p_i)
        out.append((primary[p_i - 1], cross[x_i - 1]))
    return out


def corroborate_from_semantic(
    llm: BaseLLM,
    primary: list[Claim],
    cross: list[Claim],
) -> tuple[int, int, list[tuple[Claim, Claim]]]:
    """Run the semantic same-fact pass and apply it in place.

    Every matched primary claim is marked ``crosschecked``; a supported
    ``medium`` primary is promoted to ``high`` only when the corroborating
    claim cites at least one source the primary does not (schema 'high'
    semantics — independence is about the sources). Returns
    (n_corroborated, n_promoted, pairs) so the caller can exclude matched
    cross claims from the appended-candidates list."""
    pairs = semantic_corroborate(llm, primary, cross)
    flags = promotions = 0
    for pc, xc in pairs:
        pc.crosschecked = True
        flags += 1
        pc_locs = {e.source.locator() for e in pc.evidence}
        xc_locs = {e.source.locator() for e in xc.evidence}
        if (pc.verdict is Verdict.SUPPORTED
                and pc.confidence == "medium"
                and pc_locs and pc_locs != xc_locs):
            pc.confidence = "high"
            promotions += 1
    return flags, promotions, pairs


def detect_contradictions(llm: BaseLLM, claims: list[Claim]) -> list[tuple[int, int]]:
    """One LLM pass finds contradicting pairs; indices are validated against
    the real list so the model cannot invent claims to argue with."""
    if len(claims) < 2:
        return []
    numbered = "\n".join(f"[{i}] {c.statement}" for i, c in enumerate(claims, start=1))
    data = llm.complete_json(CONFLICT_DETECTOR_SYSTEM, numbered)
    pairs: list[tuple[int, int]] = []
    for raw in data.get("contradicting_pairs", []):
        if not isinstance(raw, list) or len(raw) != 2:
            continue
        try:
            i, j = int(raw[0]), int(raw[1])
        except (ValueError, TypeError):
            continue
        if i == j or not (1 <= i <= len(claims) and 1 <= j <= len(claims)):
            continue
        a, b = sorted((i, j))
        pair = (a, b)
        if pair not in pairs:
            pairs.append(pair)
    return pairs
