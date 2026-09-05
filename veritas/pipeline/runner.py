"""Runner: orchestrate one research mission end-to-end.

mission flow
------------
1. plan          — decompose the query (or use the user-provided plan hooks)
2. research      — per sub-question: search each surface, fetch full text,
                   researcher notes
3. claims        — evidence-bound claim extraction per sub-question
4. verify        — per claim: refetch sources, LLM judge verdict + confidence
5. cross-check   — independent second plan+research+claims, reconcile
6. synthesize    — prose answer from verified claims + deterministic report

Artifacts are written to the output directory: ``report.md`` (human) and
``ledger.json`` (machine-readable evidence + claims + verdicts).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..config import settings
from ..llm import BaseLLM, FakeLLM
from ..schema import Claim, Plan, Query, Report, Verdict
from .claims import extract_claims
from .crosscheck import run_crosscheck
from .research import make_plan, research_subquestion, researcher_notes
from .synthesize import render_report, synth_prose

LogFn = Callable[[str], None]


def quiet_log(msg: str) -> None:
    pass


def stderr_log(msg: str) -> None:
    print(f"[veritas] {msg}", file=sys.stderr, flush=True)


class Runner:
    def __init__(
        self,
        llm: BaseLLM | None = None,
        *,
        providers: list | None = None,
        log: LogFn = stderr_log,
        max_subquestions: int = 5,
        evidence_per_subquestion: int = 8,
        fetch_top: int = 3,
        max_claims: int = 40,
        enable_crosscheck: bool = True,
        outdir: Path | str | None = None,
    ) -> None:
        self.llm = llm or FakeLLM({})
        self.providers = providers or []
        self.log = log
        self.max_subquestions = max_subquestions
        self.evidence_per_subquestion = evidence_per_subquestion
        self.fetch_top = fetch_top
        self.max_claims = max_claims
        self.enable_crosscheck = enable_crosscheck
        self.outdir = Path(outdir) if outdir else Path.cwd() / "out"

    # ------------------------------------------------------------------ run
    def run(self, query: Query) -> Report:
        self.log(f"mission: {query.text[:120]}")
        self.log(f"surfaces: {', '.join(query.surface_names())}")
        providers = self.providers or _fresh_providers(query)
        for p in providers:
            self.log(f"  provider: {p.describe()}")

        plan = make_plan(self.llm, query)
        self.log(f"plan: {len(plan.subquestions)} sub-question(s) — {plan.overview[:100]}")

        # ---- 2+3. research + claims per sub-question (sequential, cost-bounded)
        all_claims: list[Claim] = []
        gaps: list[str] = []
        for sub in plan.subquestions[: self.max_subquestions]:
            self.log(f"  research: {sub.text[:90]}")
            sub_providers = _fresh_providers(query) if not self.providers else _copy_providers(self.providers)
            evidence, warnings = research_subquestion(
                self.llm, sub_providers, sub.text,
                limit=self.evidence_per_subquestion, fetch_top=self.fetch_top,
            )
            for w in warnings:
                self.log(f"    warn: {w}")
            if not evidence:
                gaps.append(f"no evidence found for: {sub.text}")
                continue
            notes = researcher_notes(self.llm, sub.text, evidence)
            claims, sub_gaps = extract_claims(self.llm, sub.text, evidence, researcher=notes)
            gaps.extend(sub_gaps)
            all_claims.extend(claims)
            self.log(f"    evidence {len(evidence)}, claims {len(claims)}")
            if len(all_claims) >= self.max_claims:
                self.log("  claim cap reached; stopping research")
                break

        all_claims = all_claims[: self.max_claims]
        if not all_claims:
            self.log("no claims extracted — nothing to verify")
            return self._finalize(query, plan, [], gaps, {}, [])

        # collapse cross-sub-question duplicates before paying verification cost:
        # keep the copy with the richest evidence, renumber globally
        all_claims = _dedupe_claims(all_claims)
        gaps = _dedupe(gaps)
        self.log(f"claims after dedupe: {len(all_claims)}")

        # ---- 4. verify each claim
        provider_by_surface = {p.surface: p for p in _fresh_providers(query)} \
            if not self.providers else {p.surface: p for p in self.providers}
        verified: list[Claim] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {}
            for c in all_claims:
                futs[pool.submit(_verify_one, self.llm, c, provider_by_surface)] = c
            for fut in as_completed(futs):
                claim = fut.result()
                verified.append(claim)
        self.log(f"verified: {len(verified)} claims — "
                 f"supported {sum(1 for c in verified if c.verdict is Verdict.SUPPORTED)}, "
                 f"partial {sum(1 for c in verified if c.verdict is Verdict.PARTIAL)}, "
                 f"contradicted {sum(1 for c in verified if c.verdict is Verdict.CONTRADICTED)}, "
                 f"unsupported {sum(1 for c in verified if c.verdict is Verdict.UNSUPPORTED)}")

        # ---- 5. independent cross-check
        cross_summary: dict = {}
        primary_before_cross = len(verified)
        if self.enable_crosscheck:
            try:
                xc_providers = _fresh_providers(query) if not self.providers else _copy_providers(self.providers)
                cross_summary = run_crosscheck(
                    self.llm, query, xc_providers, verified,
                    plan.crosscheck_seed_note or "independent angle",
                    subquestion_limit=max(2, self.max_subquestions // 2),
                    evidence_limit=max(4, self.evidence_per_subquestion // 2),
                )
                self.log("cross-check: corroborated "
                         f"{cross_summary.get('corroborated')} of {primary_before_cross} "
                         "primary claims, appended "
                         f"{cross_summary.get('appended', 0)} candidate claim(s)")
            except Exception as e:
                self.log(f"cross-check failed (continuing): {type(e).__name__}: {e}")

        # ---- 6. synthesize
        return self._finalize(query, plan, verified, gaps, cross_summary,
                              [c for c in verified if c.verdict in (Verdict.SUPPORTED, Verdict.PARTIAL)])

    # --------------------------------------------------------------- helpers
    def _finalize(
        self,
        query: Query,
        plan: Plan,
        claims: list[Claim],
        gaps: list[str],
        cross_summary: dict,
        assertable: list[Claim],
    ) -> Report:
        for g in cross_summary.get("cross_gaps", []):
            if g not in gaps:
                gaps.append(g)

        conflicts: list[dict] = []
        if len(assertable) >= 2:
            try:
                from .crosscheck import detect_contradictions
                for i, j in detect_contradictions(self.llm, assertable):
                    conflicts.append({"a": assertable[i - 1].statement,
                                      "b": assertable[j - 1].statement,
                                      "basis": "model-pairing"})
            except Exception as e:
                self.log(f"contradiction detection skipped: {type(e).__name__}: {e}")

        groups = []
        by_q: dict[str, list[Claim]] = {}
        for c in assertable:
            by_q.setdefault(c.subquestion or "General", []).append(c)
        for q, cs in by_q.items():
            groups.append({
                "question": q,
                "claims": [{
                    "statement": c.statement,
                    "confidence": c.confidence,
                    "evids": ", ".join(f"[{i}]" for i in range(1, len(c.evidence) + 1)),
                } for c in cs],
            })
        answer = ""
        try:
            answer = synth_prose(self.llm, {"query": query.text, "groups": groups})
            answer = answer.strip()
        except Exception as e:
            self.log(f"prose synthesis failed: {type(e).__name__}: {e}")

        report = Report(
            query=query.text,
            answer=answer,
            claims=claims,
            gaps=gaps,
            conflicts=conflicts,
            crosscheck=cross_summary,
            surfaces_used=query.surface_names(),
        )
        self._write_artifacts(report)
        return report

    def _write_artifacts(self, report: Report) -> None:
        self.outdir.mkdir(parents=True, exist_ok=True)
        ledger = {
            "query": report.query,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "surfaces": report.surfaces_used,
            "confidence_counts": report.confidence_counts(),
            "claims": [c.to_dict() for c in report.claims],
            "gaps": report.gaps,
            "conflicts": report.conflicts,
            "crosscheck": report.crosscheck,
        }
        (self.outdir / "ledger.json").write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False))
        (self.outdir / "report.md").write_text(render_report(report), encoding="utf-8")
        self.log(f"artifacts: {self.outdir/'report.md'}, {self.outdir/'ledger.json'}")


def _fresh_providers(query: Query) -> list:
    from ..connectors import build_providers
    return build_providers(query.surfaces)


def _copy_providers(providers: list) -> list:
    from copy import copy
    return [copy(p) for p in providers]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _claim_key(claim: "Claim") -> str:
    return " ".join(claim.statement.lower().split())


def _dedupe_claims(claims: list["Claim"]) -> list["Claim"]:
    """Collapse duplicate statements, keeping the copy with the richest
    evidence (most sources / longest passages), then renumber c1..cn."""
    best: dict[str, "Claim"] = {}
    for c in claims:
        key = _claim_key(c)
        prev = best.get(key)
        if prev is None:
            best[key] = c
            continue
        def _richness(x: "Claim") -> int:
            return (len(x.evidence), sum(len(e.passage) for e in x.evidence))
        if _richness(c) > _richness(prev):
            best[key] = c
    ordered = sorted(best.values(), key=lambda c: int(c.id[1:]) if c.id[1:].isdigit() else 0)
    for i, c in enumerate(ordered, start=1):
        c.id = f"c{i}"
    return ordered


def _verify_one(llm: BaseLLM, claim: Claim, provider_by_surface: dict) -> Claim:
    from .verify import verify_claim
    return verify_claim(llm, claim, provider_by_surface)
