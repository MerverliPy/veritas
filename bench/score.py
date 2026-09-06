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
from itertools import combinations
from pathlib import Path
# Protocol constant shared with bench/judge.py (kept local to avoid an import
# cycle): a judge OUTAGE that could not place a claim. Scored as not-correct,
# never excluded like a judge 'off-topic'.
FALLBACK_UNMATCHED = "__fallback_unmatched__"


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
# months' vs gold 'two months' must reject like 1958 vs 1957. Named months
# become month numbers so 'May 2017' vs gold 'April 2017' conflicts while
# a claim that omits the month still matches.
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
    "january": "1", "february": "2", "march": "3", "april": "4",
    "may": "5", "june": "6", "july": "7", "august": "8",
    "september": "9", "october": "10", "november": "11",
    "december": "12", "jan": "1", "feb": "2", "mar": "3",
    "apr": "4", "jun": "6", "jul": "7", "aug": "8", "sep": "9",
    "oct": "10", "nov": "11", "dec": "12",
}
# Synonym roles: quantity anchors are normalized so 'machines' vs gold
# 'computers' (or 'hosts'/'devices'/'systems') compare the same role.
_ANCHOR_ROLES = {
    "computer": "computers", "computers": "computers",
    "machine": "computers", "machines": "computers",
    "device": "computers", "devices": "computers",
    "host": "computers", "hosts": "computers",
    "system": "computers", "systems": "computers",
    "pc": "computers", "pcs": "computers", "node": "computers",
    "nodes": "computers",
    "country": "countries", "countries": "countries",
    "nation": "countries", "nations": "countries",
}
_NEGATION = re.compile(
    r"\b(?:not|no|never|without|nor|nothing|nobody|nowhere|neither|"
    r"hardly|barely|unlikely|\w+n't)\b")

# 'No.' used as a patent/case-number abbreviation ('No. 763,772') is not a
# negation. Require the period so a genuine quantified negation ('no 150
# countries') still counts: an abbreviation always prints 'No. <number>',
# while 'no <count> <noun>' carries no period.
_NO_ABBREV_FOLLOW = re.compile(r"^\.\s*\d")

# Status predicates whose antonyms are truth-critical: a claim that swaps
# one for the other states the opposite fact and must never match.
# Grant-status synonyms normalized onto 'granted' so a true synonym claim
# is never equidistant from a correct 'granted' entry and its mirror
# 'rejected' guard (which would tie-break toward the incorrect label).
# 'invalidated' is the verb of invalidity and folds onto 'invalid'.
_STATUS_SYNONYMS = {"issued": "granted", "awarded": "granted",
                     "allowed": "granted", "approved": "granted",
                     "patented": "granted", "invalidated": "invalid",
                     "invalidates": "invalid", "invalidating": "invalid",
                     "invalidate": "invalid"}

# Adverse dispositions (voided, unenforceable, ...) are NOT synonyms of
# 'invalid': unenforceability and invalidity are distinct patent
# dispositions, so a claim asserting one must not be credited as the
# other. They fold onto a separate 'void' marker that conflicts with
# 'valid', 'invalid', and 'granted' alike.
_VOID_STATUSES = ("unenforceable", "voided", "vacated", "revoked",
                   "overturned", "cancelled", "canceled")

_STATUS_TERMS = {"granted": "granted", "rejected": "rejected",
                 "valid": "valid", "invalid": "invalid"}

# Pairs of canonical statuses that cannot both be asserted about the same
# thing: claiming one must never match gold stating the other.
_CONFLICTING_STATUSES = (("granted", "rejected"), ("valid", "invalid"),
                         ("granted", "void"), ("valid", "void"),
                         ("invalid", "void"), ("rejected", "void"))


_STATUS_TERMS.update({w: "void" for w in _VOID_STATUSES})


def _negation_count(text: str) -> int:
    """Number of negation markers. Double negation ('did not spread without
    requiring...') must not collapse to the same polarity as a single
    'without'. A 'no' that is a patent/case-number abbreviation ('patent
    No. 763,772' — period followed by a digit) is not counted; a genuine
    quantified negation ('no 150 countries') still is."""
    count = 0
    for m in _NEGATION.finditer(text.lower()):
        if m.group(0) == "no" and _NO_ABBREV_FOLLOW.match(text[m.end():]):
            continue  # 'No. <number>' abbreviation, not a negation
        count += 1
    return count


def _status_set(text: str) -> set[str]:
    """Canonical status predicates asserted in ``text`` (after synonym and
    void-family folding). Empty when the text asserts no status term."""
    toks = _sig_tokens(text)
    out: set[str] = set()
    for t in toks:
        c = _STATUS_SYNONYMS.get(t, t)
        if c in _STATUS_TERMS:
            out.add(_STATUS_TERMS[c])
    return out


def _antonym_conflict(statement: str, gold: str) -> bool:
    """True when the two statements assert conflicting status predicates
    ('granted in 1900' vs 'rejected in 1900', 'claims ... valid' vs
    'claims ... invalid', 'claim 16 valid and infringed' vs 'claim 16
    unenforceable'). Word-level similarity cannot see the opposition, and
    adverse dispositions like unenforceable/voided are distinct from
    invalid, so an explicit canonical-status conflict stops any of these
    directions from matching."""
    sa, ga = _status_set(statement), _status_set(gold)
    if not sa or not ga:
        return False
    for a, b in _CONFLICTING_STATUSES:
        if (a in sa and b in ga) or (b in sa and a in ga):
            return True
    return False


def _quantities(text: str) -> set[str]:
    digits = {q.replace(",", "") for q in _QUANTITY.findall(text)}
    words = {v for w, v in _NUMWORD.items()
             if re.search(rf"\b{w}\b", text.lower())}
    return digits | words


def _quantity_anchors(text: str) -> dict[str, set[str]]:
    """Map each quantity (digit run or spelled number/month word) to the role
    of the content word it modifies — the following non-numeric word, else
    the preceding one — with synonym roles normalized (machines ==
    computers). Role-aware: 'roughly 150 computers' anchors 150 to
    'computers', so a claim swapping 150 machines for 200,000 computers is
    caught even though {150, 2017} is a subset of gold's
    {200000, 150, 2017}."""
    words = [(m.group(0), m.start())
             for m in re.finditer(r"[a-z]{3,}", text.lower())]
    anchors: dict[str, set[str]] = {}

    def role(w: str) -> str:
        return _ANCHOR_ROLES.get(w, w)

    def add(val: str, pos: int) -> None:
        nxt = next((w for w, p in words
                    if p > pos and w not in _NUMWORD), None)
        prv = next((w for w, p in reversed(words)
                    if p < pos and w not in _NUMWORD), None)
        anchor = nxt or prv
        if anchor:
            anchors.setdefault(role(anchor), set()).add(val)

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


def _sig_tokens(text: str) -> set[str]:
    return {_STATUS_SYNONYMS.get(t, t)
            for t in re.findall(r"[a-z0-9_]{4,}", text.lower())}


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
    sq, sneg = _quantities(statement), _negation_count(statement)
    best, best_sim = None, 0.0
    for exp in expected:
        st = exp["statement"]
        eq, eneg = _quantities(st), _negation_count(st)
        if sq and eq and not (sq <= eq or eq <= sq):
            continue  # disjoint quantity sets: explicit disagreement
        if _quantity_conflict(statement, st):
            continue  # same anchored quantity, different value
        if _antonym_conflict(statement, st):
            continue  # status predicate swapped for its antonym
        if sneg != eneg:
            continue  # different negation scope/count is a different claim
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

def reliability(claims: list[dict], gold: dict,
                label_fn=gold_verdict) -> dict[str, float | int]:
    """P(gold-correct | confidence=b) over claims with an unambiguous gold
    label (correct/incorrect; contested/off-topic excluded), verdict
    supported/partial. ``label_fn`` maps a statement to its gold label
    (lexical by default, LLM judge when injected)."""
    buckets = {b: [0, 0] for b in ("high", "medium", "low")}  # correct, total
    for c in claims:
        if c["verdict"] not in ("supported", "partial"):
            continue
        gv = label_fn(c["statement"])
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


def compute_query_metrics(ledger: dict, gold: dict | None,
                          claim_judge=None) -> dict:
    """Metrics for one query run (spec §5). Gold-less runs report structure
    only; every *_n field is 0 and pass-relevant values are None.

    ``claim_judge`` (optional) labels each claim against gold facts
    (bench/judge.py) for precision/calibration — the lexical matcher cannot
    credit the pipeline's generative claims. When absent, lexical
    ``gold_verdict`` is used. Recall stays lexical in both cases."""
    claims = ledger.get("claims", [])
    cls = query_class(gold)
    m: dict = {
        "class": cls,
        "query_id": gold.get("query_id") if gold else None,
        "n_claims": len(claims),
        "verdict_counts": _counts(claims, "verdict"),
        "confidence_counts": _counts(claims, "confidence"),
        "n_gaps": len(ledger.get("gaps", [])),
        "conflict_pairs": sum(1 for c in claims if c.get("conflicts")),
        "crosschecked": sum(1 for c in claims if c.get("crosschecked")),
        "asserted_n": 0,          # claims the pipeline stands behind
        "high_asserted_n": 0,     # asserted + confidence high (cross-check win)
        "corroborated_n": 0,      # asserted + crosschecked/high (A2 finding)
        "corroboration_rate": None,
        "judge_counts": {},       # judge label distribution when judging
        # default nulls
        "precision_supported": None, "precision_supported_n": 0,
        "precision_unscored_n": 0,
        "recall_gold": None, "recall_gold_n": 0,
        "reliability": {"high": None, "high_n": 0, "medium": None,
                        "medium_n": 0, "low": None, "low_n": 0},
        "unsupported_share_U": None, "unsupported_share_U_n": 0,
        "subquestion_unresolved_U": None,  # re-spec A3 (sub-question level)
        "subquestion_unresolved_n": 0,
        "subquestion_total_n": 0,
        "fabrication_U": None,
        "contradiction_fires_D": None,
    }
    asserted = [c for c in claims if c["verdict"] in ("supported", "partial")]
    m["asserted_n"] = len(asserted)
    m["high_asserted_n"] = sum(1 for c in asserted
                                if c["confidence"] == "high")
    # Corroboration rate (re-spec A2 finding): share of asserted claims the
    # independent cross-check pass saw (crosschecked flag) or that reached
    # high confidence (the cross-check win). Ledger-only; no gold needed.
    m["corroborated_n"] = sum(1 for c in asserted
                               if c.get("crosschecked")
                               or c.get("confidence") == "high")
    m["corroboration_rate"] = (m["corroborated_n"] / len(asserted)
                                if asserted else None)
    if gold is None or not claims:
        # A claims-less U ledger still registers gap-named sub-questions as
        # honestly unresolved (re-spec A3) — the pipeline admitted failure.
        if cls == "U" and gold is not None:
            _subquestion_honest_failure(m, ledger)
        return m

    expected = gold.get("expected_claims", [])
    _label_cache: dict[str, str] = {}

    def claim_label(statement: str) -> str:
        # Memoize: each claim is judged once even when several metrics
        # (precision, calibration) consume the same label.
        if statement in _label_cache:
            return _label_cache[statement]
        if claim_judge is not None:
            label = claim_judge(statement, expected, ledger.get("query"))
            if label == FALLBACK_UNMATCHED:
                # Judge outage: the claim could not be placed. Treat it as
                # 'incorrect' everywhere (precision denominator AND the A2
                # calibration buckets) — confidence placed in an unverifiable
                # claim is a conservative calibration miss, never a credit.
                _label_cache[statement] = "incorrect"
                return "incorrect"
            m["judge_counts"][label] = m["judge_counts"].get(label, 0) + 1
            if label == "correct":
                out = "correct"
            elif label in ("incorrect",):
                out = "incorrect"
            elif label == "contested":
                out = "contested"
            else:
                out = "unmatched"  # judge off-topic: no gold coverage
        else:
            out = gold_verdict(statement, expected)
        _label_cache[statement] = out
        return out

    expected = gold.get("expected_claims", [])
    if cls in ("F", "C"):
        supported = [c for c in claims if c["verdict"] == "supported"]
        labels = [claim_label(c["statement"]) for c in supported]
        correct = sum(1 for l in labels if l == "correct")
        if claim_judge is not None:
            # Judge semantics: gold is a SAMPLE of checkable facts, so claims
            # it cannot place (off-topic true context) are excluded and
            # reported, never scored wrong. Fabrication is caught by the
            # judge labeling demonstrably false claims 'incorrect'.
            placed = sum(1 for l in labels if l in ("correct", "incorrect"))
            m["precision_supported"] = correct / placed if placed else None
            m["precision_supported_n"] = placed
            m["precision_unscored_n"] = len(supported) - placed
        else:
            # Lexical semantics (fallback / --no-judge): the matcher cannot
            # tell a true paraphrase from a falsehood, so every supported
            # claim counts and unmatched means not-correct (conservative).
            m["precision_supported"] = (correct / len(supported)
                                         if supported else None)
            m["precision_supported_n"] = len(supported)
            m["precision_unscored_n"] = 0
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
        m["reliability"] = reliability(claims, gold, claim_label)
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
        _subquestion_honest_failure(m, ledger)
    if cls == "D":
        # conflict pairs recorded on claims (ledger keeps report-level
        # conflicts inside claim.conflicts when reconciled) or as top-level.
        fires = m["conflict_pairs"] + len(ledger.get("conflicts", []))
        m["contradiction_fires_D"] = 1 if fires else 0
    return m


def _subquestion_honest_failure(m: dict, ledger: dict) -> None:
    """Re-spec A3: honest failure at SUB-QUESTION level. A U sub-question is
    honestly unresolved when every claim attached to it is (unsupported|low)
    — the pipeline did not confidently assert the asked quantity — or it
    produced no claims and is named in a gap (runner emits 'no evidence found
    for: <sub.text>'). The universe is the recoverable one: sub-questions that
    produced claims (their text rides on each claim) plus gap-named ones.
    Judge off-topic labeling of U claims is the documented optional
    refinement (U claims are not judged today); until then confidence
    low/unsupported is the signal. Mutates ``m`` in place; safe for
    claims-less U ledgers (gap-named sub-questions still register)."""
    claims = ledger.get("claims", [])
    by_sq: dict[str, list[dict]] = {}
    for c in claims:
        key = (c.get("subquestion") or "").strip()
        by_sq.setdefault(key, []).append(c)
    gap_named = _gap_named_subquestions(ledger)
    names = set(by_sq) | {g for g in gap_named if g}
    unresolved = 0
    for name in names:
        cs = by_sq.get(name, [])
        if not cs:
            unresolved += 1            # zero claims, gap-named
        elif all(c.get("confidence") in ("low", "unsupported")
                 for c in cs):
            unresolved += 1
    total = len(names)
    m["subquestion_unresolved_n"] = unresolved
    m["subquestion_total_n"] = total
    m["subquestion_unresolved_U"] = (unresolved / total if total
                                      else None)


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


def _confidence_proportions(ledger: dict) -> dict[str, float] | None:
    """Normalized confidence-count distribution over CONFIDENCE_ORDER for a
    rerun ledger. Real ledgers carry ``confidence_counts``; when absent
    (tests, hand-built ledgers) the counts are derived from claims' own
    confidence field. Returns None when the run has no claims to form a
    distribution."""
    cc = ledger.get("confidence_counts")
    if not isinstance(cc, dict) or not any(cc.get(k) for k in CONFIDENCE_ORDER):
        cc = _counts(ledger.get("claims", []), "confidence")
    total = sum(cc.get(k, 0) for k in CONFIDENCE_ORDER)
    if total <= 0:
        return None
    return {k: cc.get(k, 0) / total for k in CONFIDENCE_ORDER}


def has_usable_distribution(ledger: dict) -> bool:
    """True when the ledger can contribute to the A6 confidence-distance
    gate (it carries a usable confidence-count distribution). Claims-less
    reruns return False — they must never count toward the >=3 rerun
    minimum that the distribution gate needs."""
    return _confidence_proportions(ledger) is not None


def plan_subquestions(ledger: dict) -> set[str]:
    """The run's plan as a sub-question set: claim-backed sub-questions plus
    gap-named ones (a planned sub-question that found no evidence still
    exists in the plan). Shared by the A3 honest-failure universe, the A6
    plan-overlap Jaccard, and the rerun-collector's usability test."""
    named = {c.get("subquestion", "").strip()
             for c in ledger.get("claims", [])
             if (c.get("subquestion") or "").strip()}
    named |= {g for g in _gap_named_subquestions(ledger) if g}
    return named


def normalized_conf_l1(ledger_a: dict, ledger_b: dict) -> float | None:
    """Normalized L1 (total-variation distance) between two reruns'
    confidence-count distributions, in [0, 1] (0 = identical distributions).
    None when either run has no claims. This is re-spec A6(a)."""
    pa, pb = _confidence_proportions(ledger_a), _confidence_proportions(ledger_b)
    if pa is None or pb is None:
        return None
    return 0.5 * sum(abs(pa[k] - pb[k]) for k in CONFIDENCE_ORDER)


def _gap_named_subquestions(ledger: dict) -> list[str]:
    """Sub-question texts the runner recorded as finding no evidence
    (``gaps = ["no evidence found for: <sub.text>"]``) — a planned
    sub-question that produced no claims. Shared by the A3 honest-failure
    universe and the A6 plan-overlap sets so both see the same plan."""
    return [
        g.split("no evidence found for: ", 1)[1].strip()
        for g in ledger.get("gaps", [])
        if g.startswith("no evidence found for: ")
    ]


def subquestion_jaccard(ledger_a: dict, ledger_b: dict) -> float | None:
    """Jaccard of the two reruns' sub-question statement sets (plan overlap,
    re-spec A6(b)). The universe includes both claim-backed sub-questions and
    gap-named ones (a planned sub-question that found no evidence still
    exists in the plan). None when either run has no sub-question text."""
    def _sq(ledger: dict) -> set[str]:
        named = {c.get("subquestion", "").strip()
                 for c in ledger.get("claims", [])
                 if (c.get("subquestion") or "").strip()}
        named |= {g for g in _gap_named_subquestions(ledger) if g}
        return named
    sa, sb = _sq(ledger_a), _sq(ledger_b)
    if not sa or not sb:
        return None
    return len(sa & sb) / len(sa | sb)


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
          flip_pairs: list[tuple[dict, dict]] | None = None,
          rerun_groups: list[list[dict]] | None = None) -> dict:
    """A1–A6 over the per-query metric list. Each gate:
    {ok: bool|None (None = not applicable), value, detail}.

    Re-spec semantics (spec §13, owner-approved 2026-09-06):
    - A2 gates POPULATED buckets only: medium >= 0.70 AND medium >= low when
      low is sampled; low has no floor; high reliability is REPORTED only (a
      near-empty high bucket is a documented limitation, not a silent pass).
      A new corroboration-rate finding (asserted & crosschecked/high) gates
      at >= 0.05 so a working cross-check pass is observable.
    - A3 measures U honest failure at SUB-QUESTION level: a sub-question is
      honestly unresolved when every claim attached is (unsupported|low) or
      it produced no claims and appears in a gap; gate >= 0.6 of U
      sub-questions.
    - A4 evaluates ONLY on same-query paired arms (>= 2 paired queries,
      matched by query_id): (a) contradiction fires on >= half the paired D
      queries in the with-arm, (b) with-arm high_share > without-arm
      high_share, (c) with-arm precision >= without-arm precision - 0.05
      tolerance. Placed-claim populations are reported so a confounded pair
      is visible.
    - A6 is distribution-level determinism over >= 3 reruns of a query
      (``rerun_groups``): median pairwise normalized L1 of confidence-count
      distributions <= 0.30 gates; median pairwise sub-question Jaccard
      >= 0.40 is reported; statement-level ``flip_rate`` (via
      ``flip_pairs``) stays informational.

    ``q_metrics`` is the mainline (cross-check on) arm."""
    fc = [m for m in q_metrics if m.get("class") in ("F", "C")
          and m.get("precision_supported_n")]
    prec = (sum(m["precision_supported"] * m["precision_supported_n"]
                for m in fc) / sum(m["precision_supported_n"] for m in fc)
            if fc else None)

    fab = sum((m["fabrication_U"] or 0) for m in q_metrics
              if m.get("class") == "U")
    u_present = any(m.get("class") == "U" for m in q_metrics)

    # A1 needs both halves to be assessable: precision on F/C and fabrication
    # on U. Either half missing -> not applicable (None), never a silent pass.
    a1_ok = None if (prec is None or not u_present) \
        else (prec >= 0.90 and fab == 0)

    # ---- A2: populated-bucket calibration + corroboration floor -----------
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
    med_n = sum(m["reliability"].get("medium_n", 0) for m in q_metrics)
    low_n = sum(m["reliability"].get("low_n", 0) for m in q_metrics)
    high_n = sum(m["reliability"].get("high_n", 0) for m in q_metrics)

    # Corroboration rate: share of asserted claims the cross-check pass saw
    # (crosschecked flag) or that reached high (the cross-check win).
    asserted_total = sum(m.get("asserted_n", 0) for m in q_metrics)
    corr_n = sum(m.get("corroborated_n", 0) for m in q_metrics)
    corr_rate = (corr_n / asserted_total) if asserted_total else None

    # Gate on POPULATED buckets: medium must be sampled; low is reported
    # with no floor (ordering only checked when low is sampled).
    rel_ok = None
    if rel_n and med_n:
        med = rel["medium"]
        ok = (med is not None and med >= 0.70)
        if low_n:
            ok = ok and rel["low"] is not None and med >= rel["low"]
        if corr_rate is not None:
            ok = ok and corr_rate >= 0.05
        rel_ok = ok
    high_note = ("high bucket near-empty (n=%d): high reliability is reported "
                 "only until cross-check corroboration improves" % high_n
                 if high_n == 0 else "")

    # ---- A3: U honest failure at sub-question level -----------------------
    u_sub = [m for m in q_metrics if m.get("class") == "U"
             and m.get("subquestion_total_n")]
    unres_n = sum(m.get("subquestion_unresolved_n", 0) for m in u_sub)
    total_n = sum(m.get("subquestion_total_n", 0) for m in u_sub)
    u_share = (unres_n / total_n) if total_n else None
    # claim-level share stays informational alongside the sub-question metric
    u_cl = [m for m in q_metrics if m.get("class") == "U"
            and m.get("unsupported_share_U_n")]
    cl_share = (sum(m["unsupported_share_U"] * m["unsupported_share_U_n"]
                    for m in u_cl)
                / sum(m["unsupported_share_U_n"] for m in u_cl)
                if u_cl else None)
    a3_ok = None if not u_sub else (u_share is not None and u_share >= 0.60)

    # ---- A4: same-query paired cross-check benefit ------------------------
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

    def _placed(ms: list[dict]) -> int:
        return sum(m.get("precision_supported_n", 0) for m in ms
                   if m.get("class") in ("F", "C"))

    a4_ok = None
    a4_value: dict = {}
    if q_metrics_nocc is not None:
        # Same-query pairing: only queries present in BOTH arms count toward
        # A4, so an independent-mission population confound is visible (each
        # arm's placed-claim count is reported), never silently compared.
        with_by_id = {m.get("query_id"): m for m in q_metrics
                      if m.get("query_id")}
        nocc_by_id = {m.get("query_id"): m for m in q_metrics_nocc
                      if m.get("query_id")}
        paired_ids = sorted(set(with_by_id) & set(nocc_by_id))
        pw = [with_by_id[i] for i in paired_ids if i in with_by_id]
        po = [nocc_by_id[i] for i in paired_ids if i in nocc_by_id]
        d_paired = [m for m in pw if m.get("class") == "D"]
        fires = sum(1 for m in d_paired if m.get("contradiction_fires_D"))
        hs_with, hs_without = _high_share(pw), _high_share(po)
        pr_with, pr_without = _fc_prec(pw), _fc_prec(po)
        a4_value = {
            "n_paired": len(paired_ids),
            "paired_ids": paired_ids,
            "n_D_paired": len(d_paired),
            "D_fired_with": fires,
            "high_share_with": hs_with,
            "high_share_without": hs_without,
            "precision_with": pr_with,
            "precision_without": pr_without,
            "placed_with": _placed(pw),
            "placed_without": _placed(po),
        }
        if len(paired_ids) < 2 or not d_paired:
            a4_ok = None  # under-powered / no D query to fire
        elif pr_with is None or pr_without is None:
            # Paired set with no usable F/C precision population on either
            # arm: the precision condition was never measured -> n/a, never a
            # FAIL on an unevaluated condition (Codex P2).
            a4_ok = None
        else:
            fires_ok = fires * 2 >= len(d_paired)      # >= half the D queries
            hs_ok = (hs_with is not None and hs_without is not None
                     and hs_with > hs_without)
            pr_ok = pr_with >= pr_without - 0.05
            a4_ok = fires_ok and hs_ok and pr_ok

    # ---- A5: relevance ----------------------------------------------------
    med = _median([float(j) for j in relevance_judgements]) \
        if relevance_judgements else None
    a5_ok = None if med is None else med >= 0.70

    # ---- A6: distribution-level determinism -------------------------------
    flips: list[float | None] = []
    if flip_pairs:
        flips = [flip_rate(a, b) for a, b in flip_pairs]
        flips = [f for f in flips if f is not None]
    flip = max(flips) if flips else None   # informational only

    l1_vals: list[float] = []
    jac_vals: list[float] = []
    n_groups_ge3 = 0
    for group in rerun_groups or []:
        # L1 (the gate) needs >=3 reruns with a usable confidence
        # distribution. A claims-less rerun has no distribution and must not
        # count toward that minimum (a phantom third would fabricate a pass).
        usable = [ledger for ledger in group
                  if has_usable_distribution(ledger)]
        if len(usable) >= 3:
            n_groups_ge3 += 1
            for a, b in combinations(usable, 2):
                d = normalized_conf_l1(a, b)
                if d is not None:
                    l1_vals.append(d)
        # Plan-overlap Jaccard is a REPORTED diagnostic over >=3 plan-bearing
        # reruns (the same rerun minimum as the distribution gate). It is
        # computed over ALL plan-bearing ledgers in the group — including a
        # claims-less rerun that recorded its planned questions as gaps.
        # Dropping it (filtering by the confidence distribution first) would
        # hide a measurable plan-overlap signal whenever a rerun honestly
        # found no evidence (Codex P2).
        plan_bearing = [ledger for ledger in group
                        if plan_subquestions(ledger)]
        if len(plan_bearing) >= 3:
            for a, b in combinations(plan_bearing, 2):
                j = subquestion_jaccard(a, b)
                if j is not None:
                    jac_vals.append(j)
    med_l1 = _median(l1_vals) if l1_vals else None
    med_jac = _median(jac_vals) if jac_vals else None
    a6_ok = None
    if n_groups_ge3 and med_l1 is not None:
        a6_ok = med_l1 <= 0.30

    return {
        "A1_precision_fabrication": {
            "ok": a1_ok,
            "value": {"precision_supported": prec, "fabrication_U": fab},
            "detail": "A1 needs precision_n>0 on F/C (or U queries for the fabrication half).",
        },
        "A2_calibration": {
            "ok": rel_ok,
            "value": {"reliability": rel, "n_scored": rel_n,
                      "corroboration_rate": corr_rate,
                      "corroborated_n": corr_n, "asserted_n": asserted_total},
            "detail": "gates populated buckets: medium>=0.70 and medium>=low "
                       "(when low sampled); corroboration rate >=0.05; high is "
                       "reported only" + (high_note if high_n == 0 else ""),
        },
        "A3_honest_failure_U": {
            "ok": a3_ok,
            "value": {"honest_unresolved_U_subquestions": u_share,
                      "n_subquestions_total": total_n,
                      "n_subquestions_unresolved": unres_n,
                      "unsupported_share_U_claim_level": cl_share},
            "detail": "sub-question level: needs U sub-questions; "
                       ">= 0.6 honestly unresolved",
        },
        "A4_crosscheck_benefit": {
            "ok": a4_ok,
            "value": a4_value,
            "detail": "needs the paired cross-check-off arm (q_metrics_nocc) "
                       "with >=2 same-query pairs incl. a D query; fires on "
                       ">=half the paired D queries, with-arm high_share > "
                       "without, precision within 0.05 tolerance",
        },
        "A5_relevance": {
            "ok": a5_ok,
            "value": {"median_relevance": med},
            "detail": "needs rubric judgements input",
        },
        "A6_determinism": {
            "ok": a6_ok,
            "value": {"median_pairwise_l1": med_l1,
                      "median_pairwise_subq_jaccard": med_jac,
                      "flip_rate": flip,
                      "n_rerun_groups_ge3": n_groups_ge3},
            "detail": "needs >=3 reruns of a query (rerun_groups): "
                       "distribution L1 <= 0.30 gates; sub-question Jaccard "
                       ">= 0.40 reported; statement flip_rate informational",
        },
    }
