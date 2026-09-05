"""Synthesis: assemble the final report.

The LLM writes the readable prose answer from verified claims only. The rest
of the report (confidence table, citations, gaps, conflicts, cross-check
summary) is assembled deterministically from the claim ledger, so the report's
*structure* can never drift from the verified data even if prose is imperfect.

Rendering contract
------------------
The synthesizer may not add facts. Citations ``[n]`` inside the prose refer to
the deterministic reference list appended by :func:`render_report`.
"""

from __future__ import annotations

from ..llm import BaseLLM
from ..schema import Claim, Report, Verdict
from .prompts import SYNTHESIZER_SYSTEM

ASSERTABLE = {Verdict.SUPPORTED, Verdict.PARTIAL}


def synth_prose(llm: BaseLLM, report_pre: dict) -> str:
    """LLM writes prose constrained to given claims (see system prompt)."""
    sections = []
    for group in report_pre["groups"]:
        lines = [f"Sub-question: {group['question']}"]
        for c in group["claims"]:
            tag = f"[{c['confidence'].upper()}]"
            lines.append(f"- {tag} {c['statement']} (evidence: {c['evids']})")
        sections.append("\n".join(lines))
    user = (
        f"Original request: {report_pre['query']}\n\n"
        "Verified claims, grouped by sub-question:\n\n" + "\n\n".join(sections)
    )
    return llm.complete(SYNTHESIZER_SYSTEM, user, temperature=0.2, max_tokens=2500)


def render_report(report: Report, sources_per_claim: int = 3) -> str:
    """Deterministic markdown report from the ledger. Pure function."""
    out: list[str] = []
    out.append(f"# Research report: {report.query}")
    out.append("")
    counts = report.confidence_counts()
    out.append(f"*Claims: {sum(counts.values())} — "
               f"high {counts.get('high',0)} · medium {counts.get('medium',0)} · "
               f"low {counts.get('low',0)} · unsupported {counts.get('unsupported',0)}*")
    out.append(f"*Surfaces: {', '.join(report.surfaces_used) or 'none'}*")
    out.append("")
    if report.answer.strip():
        out.append("## Answer")
        out.append("")
        out.append(report.answer.strip())
        out.append("")

    out.append("## Verified claims")
    out.append("")
    by_q: dict[str, list[Claim]] = {}
    for c in report.claims:
        by_q.setdefault(c.subquestion or "General", []).append(c)
    ref_index = 0
    refs: dict[str, int] = {}
    for question, claims in by_q.items():
        out.append(f"### {question}")
        out.append("")
        for c in claims:
            tag = f"**{c.confidence.upper()}**"
            evids = []
            for ev in c.evidence[:sources_per_claim]:
                loc = ev.source.locator()
                if loc not in refs:
                    ref_index += 1
                    refs[loc] = ref_index
                evids.append(f"[{refs[loc]}]")
            note = f" — {c.note}" if c.note else ""
            out.append(f"- {tag} {c.statement}{note} {(''.join(evids)) if evids else ''}")
        out.append("")

    assertable = [c for c in report.claims if c.verdict in ASSERTABLE and c.confidence != "unsupported"]
    if assertable:
        out.append("## Sources")
        out.append("")
        numbered = {}
        for c in assertable:
            for ev in c.evidence[:sources_per_claim]:
                loc = ev.source.locator()
                numbered[refs[loc]] = (ev.source.title, loc)
        for n in sorted(numbered):
            title, loc = numbered[n]
            out.append(f"{n}. {title} — {loc}")
        out.append("")

    unsupported = [c for c in report.claims if c.confidence == "unsupported"]
    if unsupported:
        out.append("## Not established (uncertainty is stated, not hidden)")
        out.append("")
        for c in unsupported:
            out.append(f"- {c.statement} — {c.note or 'no supporting evidence found'}")
        out.append("")
    if report.gaps:
        out.append("### Gaps noted during research")
        out.append("")
        for g in report.gaps[:20]:
            out.append(f"- {g}")
        out.append("")

    conflicts = [c for c in report.claims if c.verdict is Verdict.CONTRADICTED]
    if conflicts or report.conflicts:
        out.append("## Conflicts in the evidence")
        out.append("")
        if report.conflicts:
            out.append("The following verified claims cannot both be true:")
            out.append("")
            seen_pair: set[tuple[str, str]] = set()
            for conflict in report.conflicts:
                a, b = conflict.get("a"), conflict.get("b")
                if a and b:
                    key = tuple(sorted((a, b)))
                    if key in seen_pair:
                        continue
                    seen_pair.add(key)
                    out.append(f"- “{a}”")
                    out.append(f"  - vs “{b}”")
            out.append("")
        for c in conflicts:
            out.append(f"- {c.statement} (rejected: contradicted by its cited source)")
            if c.note:
                out.append(f"  - Verifier: {c.note}")
        out.append("")

    out.append("## Cross-check")
    out.append("")
    xc = report.crosscheck or {}
    if xc:
        out.append(f"- Independent second pass overview: {xc.get('overview') or '—'}")
        out.append(f"- Independent claims produced: {xc.get('cross_claims', 0)}; "
                   f"corroborated primary claims: {xc.get('corroborated', 0)}")
        out.append(f"- Final confidence distribution: "
                   + ", ".join(f"{k}={v}" for k, v in (xc.get("confidence_counts") or {}).items()))
    else:
        out.append("- (independent cross-check disabled for this run)")
    out.append("")
    return "\n".join(out)
