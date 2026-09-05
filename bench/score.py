"""R1 benchmark scoring — pure logic, no network, no LLM.

Consumes a mission ledger (the JSON ``Runner`` writes as ``ledger.json``) and
an optional gold sheet (``bench/gold/<id>.json``) and computes the R1 metrics
and gate checks defined in the benchmark spec (``docs/R1-BENCHMARK.md``, §5–6
— private planning doc). Token-matching semantics deliberately mirror the
pipeline's own cross-check (``veritas/pipeline/crosscheck.py``) so a "match"
here means the same thing it means there.

This module is hermetic and unit-tested; the owner-run orchestrator lives in
``run_benchmark.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# --------------------------------------------------------------------------
# Cost estimation (ESTIMATE ONLY) — VERITAS_LLM_LOG chars -> USD.
# deepseek-chat blended price; update when provider pricing changes.
# --------------------------------------------------------------------------

USD_PER_1M_TOKENS = 0.55
CHARS_PER_TOKEN = 4.0


def est_cost_usd(log_text: str) -> float:
    """Rough USD cost of one mission's LLM traffic from its audit log."""
    tokens = len(log_text) / CHARS_PER_TOKEN
    return tokens * USD_PER_1M_TOKENS / 1_000_000


# --------------------------------------------------------------------------
# Claim <-> gold matching (same conservative token semantics as cross-check)
# --------------------------------------------------------------------------

_MATCH_JACCARD = 0.5        # pipeline claim vs a gold expected claim
_RERUN_JACCARD = 0.6        # same-query rerun: claim pairs to compare

_GOLD_LABELS = ("correct", "incorrect", "contested")
_VERDICTS = ("supported", "partial", "contradicted", "unsupported")
CONFIDENCE_ORDER = ["high", "medium", "low", "unsupported"]
CLASSES = ("F", "C", "D", "U")

# Truth-critical disagreement: token overlap alone must never certify a
# claim that contradicts gold on quantities or polarity ("in 1958" vs
# "1957", "port 444" vs "445", "did not launch" vs "launched"). Digits
# fall out of the significant-token set, so quantities are compared
# explicitly: a match is allowed only when either side cites none or one
# side's quantity set contains the other's (omission tolerated: "150
# countries" vs gold "...150 countries in May 2017"; contradiction
# rejected: 1958 vs 1957, 444 vs 445, 100,000/15 vs 200,000/150).
_QUANTITY = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Spelled-out quantities participate in disagreement checks too: 'three
# months' vs gold 'two months' must reject like 1958 vs 1957.
_NUMWORD = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000", "billion": "1000000000",
    "hundreds": "100", "thousands": "1000", "millions": "1000000",
    "billions": "1000000000",
}
_NEGATION = re.compile(
    r"\b(?:not|no|never|without|nor|nothing|nobody|nowhere|neither|"
    r"hardly|barely|unlikely)\b")


def _quantities(text: str) -> set[str]:
    digits = {q.replace(",", "") for q in _QUANTITY.findall(text)}
    words = {v for w, v in _NUMWORD.items()
             if re.search(rf"\b{w}\b", text.lower())}
    return digits | words


def _quantity_anchors(text: str) -> dict[str, set[str]]:
    """Map each quantity (digit run or spelled number word) to the content
    word it modifies — the following non-numeric word, else the preceding
    one. Role-aware: 'roughly 150 computers' anchors 150 to 'computers', so
    a claim swapping 150 computers for 200,000 is caught even though
    {150, 2017} is a subset of gold's {200000, 150, 2017}; 'three months'
    anchors 3 to 'months' and disagrees with gold's two months."""
    words = [(m.group(0), m.start())
             for m in re.finditer(r"[a-z]{3,}", text.lower())]
    anchors: dict[str, set[str]] = {}

    def add(val: str, pos: int) -> None:
        nxt = next((w for w, p in words
                    if p > pos and w not in _NUMWORD), None)
        prv = next((w for w, p in reversed(words)
                    if p < pos and w not in _NUMWORD), None)
        anchor = nxt or prv
        if anchor:
            anchors.setdefault(anchor, set()).add(val)

    low = text.lower()
    for w, v in _NUMWORD.items():
        for m in re.finditer(rf"\b{w}\b", low):
            add(v, m.start())
    for m in _QUANTITY.finditer(text):
        add(m.group(0).replace(",", ""), m.end())
    return anchors


def _quantity_conflict(statement: str, gold: str) -> bool:
    """True when both statements quantify the SAME anchored thing with
    values where neither side's value set contains the other's (different
    year/port/count for the same anchor, e.g. 'in 1958' vs 'in 1957',
    'port 444' vs 'port 445', '150 computers' vs '200,000 computers').
    Subset additions are tolerated per anchor ('May 12, 2017' adds a day
    to gold's 'May 2017' without contradicting it)."""
    sa, ga = _quantity_anchors(statement), _quantity_anchors(gold)
    for anchor in sa.keys() & ga.keys():
        if not (sa[anchor] <= ga[anchor] or ga[anchor] <= sa[anchor]):
            return True
    return False


def _negated(text: str) -> bool:
    return bool(_NEGATION.search(text.lower()))


def _sig_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{4,}", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _sig_tokens(a), _sig_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def best_gold_match(statement: str, expected: list[dict]) -> dict | None:
    """Highest-overlap gold expected claim, if it is a genuine match.

    Overlap must survive truth-critical checks: a claim that contradicts
    gold on a quantity (different year, port, or count for the same thing)
    or has opposite polarity ("did not launch" vs "launched") is a
    different claim and never scores as the gold statement — while a claim
    that merely omits a gold detail still matches. Ties in overlap resolve
    toward the less credit-worthy label (contested/incorrect over correct)
    so ambiguity never certifies credit."""
    sq, sneg = _quantities(statement), _negated(statement)
    best, best_sim = None, 0.0
    for exp in expected:
        st = exp["statement"]
        eq, eneg = _quantities(st), _negated(st)
        if sq and eq and not (sq <= eq or eq <= sq):
            continue  # disjoint quantity sets: explicit disagreement
        if _quantity_conflict(statement, st):
            continue  # same anchored quantity, different value
        if sneg != eneg:
            continue  # opposite polarity is a different claim
        sim = _jaccard(statement, st)
        if sim > best_sim or (
                sim == best_sim and sim > 0.0 and best is not None
                and exp.get("gold_label") != "correct"
                and best.get("gold_label") == "correct"):
            best, best_sim = exp, sim
    return best if best_sim >= _MATCH_JACCARD else None


def gold_verdict(statement: str, expected: list[dict]) -> str:
    """'correct' | 'incorrect' | 'contested' | 'unmatched' for a claim."""
    hit = best_gold_match(statement, expected)
    if hit is None:
        return "unmatched"
    return hit["gold_label"] if hit["gold_label"] in _GOLD_LABELS else "unmatched"


# --------------------------------------------------------------------------
# Ledger / gold loading
# --------------------------------------------------------------------------

def load_json(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def query_class(gold: dict | None) -> str | None:
    if gold is None:
        return None
    cls = gold.get("class")
    return cls if cls in CLASSES else None


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def reliability(claims: list[dict], gold: dict) -> dict[str, float | int]:
    """P(gold-correct | confidence=b) over claims with an unambiguous gold
    label (correct/incorrect; contested excluded), verdict supported/partial."""
    buckets = {b: [0, 0] for b in ("high", "medium", "low")}  # correct, total
    for c in claims:
        if c["verdict"] not in ("supported", "partial"):
            continue
        gv = gold_verdict(c["statement"], gold["expected_claims"])
        if gv not in ("correct", "incorrect"):
            continue
        b = c["confidence"]
        if b not in buckets:
            continue
        buckets[b][1] += 1
        buckets[b][0] += 1 if gv == "correct" else 0
    out: dict[str, float | int] = {}
    for b, (ok, total) in buckets.items():
        out[b] = (ok / total) if total else None
        out[f"{b}_n"] = total
    return out


def compute_query_metrics(ledger: dict, gold: dict | None) -> dict:
    """Metrics for one query run (spec §5). Gold-less runs report structure
    only; every *_n field is 0 and pass-relevant values are None."""
    claims = ledger.get("claims", [])
    cls = query_class(gold)
    m: dict = {
        "class": cls,
        "n_claims": len(claims),
        "verdict_counts": _counts(claims, "verdict"),
        "confidence_counts": _counts(claims, "confidence"),
        "n_gaps": len(ledger.get("gaps", [])),
        "conflict_pairs": sum(1 for c in claims if c.get("conflicts")),
        "crosschecked": sum(1 for c in claims if c.get("crosschecked")),
        "asserted_n": 0,          # claims the pipeline stands behind
        "high_asserted_n": 0,     # asserted + confidence high (cross-check win)
        # default nulls
        "precision_supported": None, "precision_supported_n": 0,
        "recall_gold": None, "recall_gold_n": 0,
        "reliability": {"high": None, "high_n": 0, "medium": None,
                        "medium_n": 0, "low": None, "low_n": 0},
        "unsupported_share_U": None, "unsupported_share_U_n": 0,
        "fabrication_U": None,
        "contradiction_fires_D": None,
    }
    asserted = [c for c in claims if c["verdict"] in ("supported", "partial")]
    m["asserted_n"] = len(asserted)
    m["high_asserted_n"] = sum(1 for c in asserted
                                if c["confidence"] == "high")
    if gold is None or not claims:
        return m

    expected = gold.get("expected_claims", [])
    if cls in ("F", "C"):
        supported = [c for c in claims if c["verdict"] == "supported"]
        scored = [c for c in supported]
        correct = sum(1 for c in scored
                      if gold_verdict(c["statement"], expected) == "correct")
        m["precision_supported"] = correct / len(scored) if scored else None
        m["precision_supported_n"] = len(scored)
        # Recall counts distinct base facts. A 'variant_of' entry is an
        # alternate phrasing of a base fact, not a second required fact — it
        # would otherwise inflate the recall denominator.
        bases = [e for e in expected
                 if e.get("gold_label") == "correct" and not e.get("variant_of")]
        def _covered(base: dict) -> bool:
            cands = [base] + [v for v in expected
                              if v.get("variant_of") == base["statement"]]
            return any(best_gold_match(c["statement"], cands) is not None
                       for c in claims
                       if c["verdict"] in ("supported", "partial"))
        covered = sum(1 for b in bases if _covered(b))
        m["recall_gold"] = covered / len(bases) if bases else None
        m["recall_gold_n"] = len(bases)
    if cls in ("F", "C", "D"):
        m["reliability"] = reliability(claims, gold)
    if cls == "U":
        asserted = claims
        low_or_un = [c for c in asserted
                     if c["confidence"] in ("low", "unsupported")]
        m["unsupported_share_U"] = (len(low_or_un) / len(asserted)
                                    if asserted else None)
        m["unsupported_share_U_n"] = len(asserted)
        m["fabrication_U"] = sum(
            1 for c in asserted
            if c["verdict"] == "supported" and c["confidence"] == "high")
    if cls == "D":
        # conflict pairs recorded on claims (ledger keeps report-level
        # conflicts inside claim.conflicts when reconciled) or as top-level.
        fires = m["conflict_pairs"] + len(ledger.get("conflicts", []))
        m["contradiction_fires_D"] = 1 if fires else 0
    return m


def _counts(items: list[dict], key: str) -> dict:
    out: dict = {}
    for it in items:
        out[it.get(key, "")] = out.get(it.get(key, ""), 0) + 1
    return out


# --------------------------------------------------------------------------
# Cross-run helpers (determinism arm, cross-check paired arm)
# --------------------------------------------------------------------------

def flip_rate(ledger_a: dict, ledger_b: dict) -> float | None:
    """Share of matched same-query claims flipping supported<->contradicted.
    Claims pair up by statement overlap (conservative: Jaccard >= .6)."""
    pairs = _pair_claims(ledger_a.get("claims", []), ledger_b.get("claims", []))
    if not pairs:
        return None
    flips = sum(1 for a, b in pairs
                if {a["verdict"], b["verdict"]}
                == {"supported", "contradicted"})
    return flips / len(pairs)


def _pair_claims(ca: list[dict], cb: list[dict]) -> list[tuple[dict, dict]]:
    used: set[int] = set()
    pairs = []
    for a in ca:
        best, best_sim = None, _RERUN_JACCARD
        for j, b in enumerate(cb):
            if j in used:
                continue
            sim = _jaccard(a["statement"], b["statement"])
            if sim >= best_sim:
                best, best_sim, best_j = b, sim, j
        if best is not None:
            pairs.append((a, best))
            used.add(best_j)
    return pairs


# --------------------------------------------------------------------------
# Gates (spec §6) — value may be None (not applicable) for a metric.
# --------------------------------------------------------------------------

def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def gates(q_metrics: list[dict], *,
          q_metrics_nocc: list[dict] | None = None,
          relevance_judgements: list[int] | None = None,
          flip_pairs: list[tuple[dict, dict]] | None = None) -> dict:
    """A1–A6 over the per-query metric list. Each gate:
    {ok: bool|None (None = not applicable), value, detail}.

    ``q_metrics`` is the mainline (cross-check on) arm; A4 additionally needs
    ``q_metrics_nocc`` (the paired cross-check-off arm) and stays ``None``
    without it — a mainline-only contradiction fire is not the A4 gate."""
    fc = [m for m in q_metrics if m.get("class") in ("F", "C")
          and m.get("precision_supported_n")]
    prec = (sum(m["precision_supported"] * m["precision_supported_n"]
                for m in fc) / sum(m["precision_supported_n"] for m in fc)
            if fc else None)

    u = [m for m in q_metrics if m.get("class") == "U"
         and m.get("unsupported_share_U_n")]
    u_share = (sum(m["unsupported_share_U"] * m["unsupported_share_U_n"]
                   for m in u) / sum(m["unsupported_share_U_n"] for m in u)
               if u else None)
    fab = sum((m["fabrication_U"] or 0) for m in q_metrics
              if m.get("class") == "U")

    d = [m for m in q_metrics if m.get("class") == "D"]
    d_fire = (all(m["contradiction_fires_D"] for m in d) if d else None)

    # A4 needs BOTH arms: contradiction fires on the mainline run AND the
    # cross-check benefit delta (with vs without). No paired data -> None.
    def _fc_prec(ms: list[dict]) -> float | None:
        f = [m for m in ms if m.get("class") in ("F", "C")
             and m.get("precision_supported_n")]
        return (sum(m["precision_supported"] * m["precision_supported_n"]
                    for m in f) / sum(m["precision_supported_n"] for m in f)
                if f else None)

    def _high_share(ms: list[dict]) -> float | None:
        an = sum(m.get("asserted_n", 0) for m in ms)
        hn = sum(m.get("high_asserted_n", 0) for m in ms)
        return (hn / an) if an else None

    prec_nocc = _fc_prec(q_metrics_nocc) if q_metrics_nocc is not None else None
    hs_with = _high_share(q_metrics)
    hs_nocc = _high_share(q_metrics_nocc) if q_metrics_nocc is not None else None
    delta_ok = None
    if q_metrics_nocc is not None and d and hs_with is not None \
            and hs_nocc is not None:
        delta_ok = (d_fire is True and prec_nocc is not None
                    and prec is not None and prec >= prec_nocc
                    and hs_with > hs_nocc)
    a4_ok = None if q_metrics_nocc is None else delta_ok

    rel = {b: (sum(m["reliability"].get(b) * m["reliability"].get(f"{b}_n", 0)
                   for m in q_metrics
                   if m["reliability"].get(f"{b}_n"))
                / sum(m["reliability"].get(f"{b}_n", 0) for m in q_metrics
                      if m["reliability"].get(f"{b}_n"))
                if any(m["reliability"].get(f"{b}_n") for m in q_metrics)
                else None)
           for b in ("high", "medium", "low")}
    rel_n = sum(m["reliability"].get(f"{b}_n", 0)
                for m in q_metrics for b in ("high", "medium", "low"))

    # A2 needs all three buckets sampled; never compare None against a bar.
    rel_ok = None
    if rel_n:
        rel_ok = (
            rel["high"] is not None and rel["high"] >= 0.85
            and rel["medium"] is not None and rel["medium"] >= 0.70
            and rel["low"] is not None
            and rel["high"] >= rel["medium"] >= rel["low"]
            and (rel["high"] - rel["low"]) >= 0.15
        )

    med = _median([float(j) for j in relevance_judgements]) \
        if relevance_judgements else None
    a5_ok = None if med is None else med >= 0.70

    flips: list[float | None] = []
    if flip_pairs:
        flips = [flip_rate(a, b) for a, b in flip_pairs]
        flips = [f for f in flips if f is not None]
    flip = max(flips) if flips else None
    a6_ok = None if flip is None else flip == 0.0

    # A1 needs both halves to be assessable: precision on F/C and fabrication
    # on U. Either half missing -> not applicable (None), never a silent pass.
    a1_ok = None if (prec is None or not u) else (prec >= 0.90 and fab == 0)

    return {
        "A1_precision_fabrication": {
            "ok": a1_ok,
            "value": {"precision_supported": prec, "fabrication_U": fab},
            "detail": "A1 needs precision_n>0 on F/C (or U queries for the fabrication half).",
        },
        "A2_calibration": {
            "ok": rel_ok,
            "value": {"reliability": rel, "n_scored": rel_n},
            "detail": "needs high/medium/low samples",
        },
        "A3_honest_failure_U": {
            "ok": (u_share is not None and u_share >= 0.60) if u else None,
            "value": {"unsupported_share_U": u_share},
            "detail": "needs U queries with claims",
        },
        "A4_crosscheck_benefit": {
            "ok": a4_ok,
            "value": {"all_D_fired": d_fire, "n_D": len(d),
                      "precision_with": prec, "precision_without": prec_nocc,
                      "high_share_with": hs_with,
                      "high_share_without": hs_nocc},
            "detail": "needs the paired cross-check-off arm (q_metrics_nocc)",
        },
        "A5_relevance": {
            "ok": a5_ok,
            "value": {"median_relevance": med},
            "detail": "needs rubric judgements input",
        },
        "A6_determinism": {
            "ok": a6_ok,
            "value": {"flip_rate": flip},
            "detail": "needs rerun ledger pair(s)",
        },
    }
