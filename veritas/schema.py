"""Domain model for veritas research missions.

The data model is the contract between pipeline stages. Every factual unit a
researcher produces must be traceable back to an :class:`Evidence` item; every
:class:`Claim` in the final report carries confidence and verdict from the
verification stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum


# ---------------------------------------------------------------------------
# Surface + evidence
# ---------------------------------------------------------------------------


class Surface(str, Enum):
    """Where research can gather evidence from."""

    WEB = "web"
    LOCAL = "local"
    CODE = "code"


@dataclass
class Source:
    """A retrievable origin for evidence.

    Exactly one locator is normally set: ``url`` for web evidence, ``path``
    for local-file/codebase evidence (path may include ``#L12-L20`` style
    line ranges). ``title`` is a human-readable label.
    """

    url: str | None = None
    path: str | None = None
    title: str = ""
    surface: Surface = Surface.WEB
    anchor: str = ""  # for files: "L12-L20" line range of the quoted passage

    def locator(self) -> str:
        base = self.url or self.path or "<unknown>"
        if self.path and self.anchor:
            return f"{base}#{self.anchor}"
        return base

    def to_dict(self) -> dict:
        d = asdict(self)
        d["surface"] = self.surface.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Source":
        return Source(
            url=d.get("url"),
            path=d.get("path"),
            title=d.get("title", ""),
            surface=Surface(d.get("surface", "web")),
            anchor=d.get("anchor", ""),
        )


@dataclass
class Evidence:
    """A concrete retrievable piece of support: a source plus an exact
    passage window (so verification can re-fetch the *same* text)."""

    source: Source
    passage: str  # quoted snippet, ideally verbatim from the source
    retrieved_at: float = field(default_factory=time.time)
    kind: str = "search"  # search | fetch | file

    def to_dict(self) -> dict:
        return {"source": self.source.to_dict(), "passage": self.passage,
                "retrieved_at": self.retrieved_at, "kind": self.kind}

    @staticmethod
    def from_dict(d: dict) -> "Evidence":
        return Evidence(
            source=Source.from_dict(d["source"]),
            passage=d["passage"],
            retrieved_at=d.get("retrieved_at", 0.0),
            kind=d.get("kind", "search"),
        )


# ---------------------------------------------------------------------------
# Mission + plan
# ---------------------------------------------------------------------------


@dataclass
class Query:
    """The original user request plus allowed research surfaces."""

    text: str
    surfaces: list[Surface] = field(default_factory=lambda: [Surface.WEB])
    sources: list[str] = field(default_factory=list)  # user-supplied urls/paths

    def surface_names(self) -> list[str]:
        return [s.value for s in self.surfaces]


@dataclass
class SubQuestion:
    """One decomposed research question assigned to a researcher."""

    text: str
    rationale: str = ""
    surfaces: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | done | failed


@dataclass
class Plan:
    """Planner output: decomposition of the query + role strategy."""

    overview: str
    subquestions: list[SubQuestion]
    crosscheck_seed_note: str = ""  # different framing for the independent pass

    def to_dict(self) -> dict:
        return {
            "overview": self.overview,
            "subquestions": [asdict(s) for s in self.subquestions],
            "crosscheck_seed_note": self.crosscheck_seed_note,
        }


# ---------------------------------------------------------------------------
# Claims + verdicts
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"  # no evidence found (kept, flagged, marked low confidence)


CONFIDENCE_ORDER = ["high", "medium", "low", "unsupported"]


@dataclass
class Claim:
    """A single factual assertion with evidence and a verification verdict.

    ``confidence`` is derived after verification + cross-check:
      high    — independently corroborated, consistent evidence
      medium  — supported by evidence, single source or partial
      low     — partially supported, or weak/indirect evidence
      unsupported — explicitly NOT asserted as fact; reported as a gap
    """

    id: str
    statement: str
    subquestion: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    verdict: Verdict = Verdict.UNSUPPORTED
    confidence: str = "unsupported"
    crosschecked: bool = False   # seen in the independent pass
    conflicts: list[str] = field(default_factory=list)  # statements that contradict this
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "subquestion": self.subquestion,
            "evidence": [e.to_dict() for e in self.evidence],
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "crosschecked": self.crosschecked,
            "conflicts": self.conflicts,
            "note": self.note,
        }

    @staticmethod
    def from_dict(d: dict) -> "Claim":
        return Claim(
            id=d["id"],
            statement=d["statement"],
            subquestion=d.get("subquestion", ""),
            evidence=[Evidence.from_dict(e) for e in d.get("evidence", [])],
            verdict=Verdict(d.get("verdict", "unsupported")),
            confidence=d.get("confidence", "unsupported"),
            crosschecked=d.get("crosschecked", False),
            conflicts=d.get("conflicts", []),
            note=d.get("note", ""),
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """Final artifact of a mission."""

    query: str
    answer: str  # synthesized prose, per-claim confidence annotated
    claims: list[Claim] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)   # things sources could not establish
    conflicts: list[dict] = field(default_factory=list)  # {a, b, resolution, resolved}
    crosscheck: dict = field(default_factory=dict)  # summary of independent pass
    surfaces_used: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def confidence_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in CONFIDENCE_ORDER}
        for cl in self.claims:
            counts[cl.confidence] = counts.get(cl.confidence, 0) + 1
        return counts
