"""Independent cross-check + reconciliation + contradiction detection.

A second, independently-planned research pass runs over the same request
(different decomposition / angle). Its claims are reconciled with the primary
run's claims *deterministically*:

* corroboration — every cross-run claim is put through the same LLM
  verifier as primary claims FIRST (never trusted unchecked). A verified
  cross-run claim sharing most significant tokens with a primary claim is
  independent agreement: if the primary is ``supported`` and the agreeing
  claim cites a genuinely new source, confidence rises to ``high`` and that
  source's evidence is adopted onto the primary. Unverified echoes never
  corroborate. Token matching is conservative by design: a wrong
  'agreement' is worse than none.
* candidates — cross-run claims with no primary counterpart are
  *candidates*: verified like every other cross claim, then appended with a
  note marking their origin (unless a later semantic pass consumes them as
  corroboration).
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


def _adopt_evidence(target: Claim, donor: Claim) -> None:
    """Merge the donor claim's evidence onto the target, keeping only the
    first entry per source locator. Call only with VERIFIED-supported donors
    (reconcile and the semantic pass run after verify_claim): a consumed
    corroborating claim's evidence must stay on the claim it promoted rather
    than vanish with the dropped claim (Codex P1). Fresh evidence is
    PREPENDED so the independent corroborating source lands inside
    render_report's first-N evidence window instead of after it (Codex P2)."""
    have = {e.source.locator() for e in target.evidence}
    fresh = [e for e in donor.evidence
             if e.source.locator() not in have]
    if fresh:
        target.evidence[0:0] = fresh


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
    primary_claims: list[str] | None = None,
) -> dict:
    """Run the independent pass; verified candidates are appended to
    ``primary`` (the runner's own claim list, mutated in place).

    ``primary_subquestions`` (the first pass's own sub-question texts) guide
    the independent plan so it re-derives the same key facts from different
    sources (corroboration) and actively probes for counter-evidence
    (contradiction). ``primary_claims`` (the first pass's asserted claim
    statements) give the planner the concrete facts to challenge — for an
    open-ended request the sub-questions alone cannot say which date, number
    or attribution the counter-evidence pass should probe."""
    original_primary = list(primary)  # snapshot before candidates are appended
    plan = make_crosscheck_plan(llm, query, seed_note,
                                primary_subquestions=primary_subquestions,
                                primary_claims=primary_claims)
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

    # EVERY cross claim is verified BEFORE reconciliation: reconcile's
    # lexical corroboration (crosschecked flag + a possible promotion to
    # high) may only rest on a donor that passed verify_claim — an unverified
    # echo must never drive confidence (Codex round-4 P1).
    provider_by_surface = {p.surface: p for p in providers}
    for xc in cross_claims:
        try:
            verify_claim(llm, xc, provider_by_surface)
        except Exception:
            xc.verdict = Verdict.UNSUPPORTED
            xc.confidence = "unsupported"
        origin = xc.note
        xc.note = (origin + " " if origin else "") + \
            "(from the independent cross-check pass)"

    summary = reconcile(primary, cross_claims, plan.overview, gaps)
    candidates: list[Claim] = summary.pop("candidates", [])

    # Semantic corroboration: a VERIFIED-supported independent claim may
    # corroborate a primary claim whose paraphrase the token matcher missed.
    # Promotion to high still needs a different source locator (schema
    # 'high' = independently corroborated, consistent evidence). Matched
    # cross claims corroborate an existing claim, so they are not appended
    # as new candidates.
    eligible_cross = [c for c in candidates
                      if c.verdict is Verdict.SUPPORTED]
    # ALL supported primaries are eligible (any confidence): a claim already
    # promoted high by a lexical match must still consume a later verified
    # paraphrase of the same fact (adopt its evidence) instead of letting it
    # be appended as a duplicate; promotion itself stays medium-only inside
    # corroborate_from_semantic (Codex round-4 P2).
    eligible_primary = [c for c in original_primary
                        if c.verdict is Verdict.SUPPORTED]
    sem_flags, sem_promos, sem_pairs = corroborate_from_semantic(
        llm, eligible_primary, eligible_cross)
    consumed = {id(xc) for _pc, xc in sem_pairs}
    # consumed cross claims are NOT lost: corroborate_from_semantic already
    # adopted each donor's evidence onto its matched primary above

    appended = 0
    for cand in candidates:
        if id(cand) in consumed:
            continue
        primary.append(cand)
        appended += 1
    summary["appended"] = appended
    summary["corroborated_semantic"] = sem_flags
    summary["promoted_semantic"] = sem_promos
    # recount distinct claims, not passes: a primary the lexical pass
    # crosschecked may also be promoted semantically, so adding the two
    # passes' counts would double-count. crosschecked is the complete signal
    # here — reconcile sets it before its own promotion check, and 'high' is
    # only ever assigned inside cross-check promotion, never by verify.
    summary["corroborated"] = len(
        {c.id for c in original_primary if c.crosschecked})
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
    """Deterministic corroboration + candidate selection. See module doc.

    ``cross`` claims must already have passed ``verify_claim``: only a
    verified-SUPPORTED lexical match may set ``crosschecked`` or promote to
    ``high`` (its new-source evidence is then adopted onto the primary).
    Non-matching claims become ``candidates`` for the semantic pass and the
    appended-claims list."""
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
            # only a VERIFIED-SUPPORTED donor may corroborate: a lexical echo
            # that failed verification (unsupported/contradicted) is consumed
            # without marking the primary — it must never set crosschecked or
            # drive a high promotion (Codex round-4 P1)
            if xc.verdict is not Verdict.SUPPORTED:
                continue
            matched_primary.add(pc.id)
            pc.crosschecked = True
            if pc.verdict is Verdict.SUPPORTED and pc.confidence == "medium":
                pc_locs = {e.source.locator() for e in pc.evidence}
                xc_locs = {e.source.locator() for e in xc.evidence}
                # promotion needs a genuinely NEW locator: primary {A,B} vs
                # cross {A} is subset agreement, not independent evidence
                if pc_locs and (xc_locs - pc_locs):
                    pc.confidence = "high"
            # donor passed verify_claim as supported, so its new-source
            # evidence is adopted — the promoted claim keeps the independent
            # source that justified it in the report/ledger
            _adopt_evidence(pc, xc)
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
    with. Several cross claims may match the same primary (each restates the
    fact from its own source); a cross claim still matches at most one
    primary. Returns [] when the model cannot be asked (hermetic FakeLLM,
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
        # only the CROSS index is single-use: the prompt constrains one cross
        # claim to one primary, but several independent claims may restate the
        # same primary fact — enforcing primary single-use made the outcome
        # order-dependent (a same-source match first kept the primary medium
        # and pushed the later independent paraphrase into the appended list)
        if x_i in used_cross:
            continue
        used_cross.add(x_i)
        out.append((primary[p_i - 1], cross[x_i - 1]))
    return out


def corroborate_from_semantic(
    llm: BaseLLM,
    primary: list[Claim],
    cross: list[Claim],
) -> tuple[int, int, list[tuple[Claim, Claim]]]:
    """Run the semantic same-fact pass and apply it in place.

    Every matched primary claim is marked ``crosschecked``; a supported
    ``medium`` primary is promoted to ``high`` when any corroborating claim
    cites at least one source the primary does not (schema 'high' semantics —
    independence is about the sources). The corroborating claim's evidence is
    adopted onto the primary (deduped by locator) because matched cross
    claims are consumed rather than appended, and the promotion must stay
    traceable in the report/ledger. Counts are per primary claim, so several
    cross claims agreeing with one primary count as one corroboration.
    Returns (n_corroborated, n_promoted, pairs) so the caller can exclude
    matched cross claims from the appended-candidates list."""
    pairs = semantic_corroborate(llm, primary, cross)
    corroborated: set[int] = set()
    promoted: set[int] = set()
    for pc, xc in pairs:
        corroborated.add(id(pc))
        pc.crosschecked = True
        if (pc.verdict is Verdict.SUPPORTED
                and pc.confidence == "medium"
                and {e.source.locator() for e in pc.evidence}
                and ({e.source.locator() for e in xc.evidence}
                     - {e.source.locator() for e in pc.evidence})):
            pc.confidence = "high"
            promoted.add(id(pc))
        _adopt_evidence(pc, xc)
    return len(corroborated), len(promoted), pairs


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
