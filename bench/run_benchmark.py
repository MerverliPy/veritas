#!/usr/bin/env python3
"""R1 benchmark runner — OWNER-RUN ONLY.

Live network + paid LLM backend (deepseek-chat, keyless web engines). Never
run in CI; never required by the hermetic test suite. See ``bench/README.md``.

Run one mission per query through the real CLI (``python -m veritas.cli run``),
meter cost from ``VERITAS_LLM_LOG``, stop when the per-run USD cap is hit, and
write a scorecard (metrics + A1–A6 gates) for queries that have a gold sheet.

Usage:
    python3 bench/run_benchmark.py                        # all queries, cap $1
    python3 bench/run_benchmark.py --ids f1-wannacry,u1  # subset
    python3 bench/run_benchmark.py --no-crosscheck        # paired-arm runs
    python3 bench/run_benchmark.py --cap-usd 0.25         # pilot budget
    # evaluate A4/A6 on EXISTING runs (no new missions):
    python3 bench/run_benchmark.py --rescore out/bench/cc-1 \
        --paired-arm out/bench/nocc-1 --rerun-dirs out/bench/det-1,out/bench/det-2,out/bench/det-3

Scoring is done by ``bench/score.py`` (pure, tested). The re-spec A4 (same-
query paired cross-check delta) and A6 (distribution determinism over >=3
reruns) gates need inputs from OTHER runs: pass the cross-check-off arm's
run dir with ``--paired-arm`` and >=3 determinism run dirs with
``--rerun-dirs`` (each dir holds per-query ``<id>/ledger.json`` subdirs).
Without them the scorecard marks those gates ``n/a``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # allow running from anywhere

# Scorer semantics revision. Bumped whenever bench/score.py metric/gate
# semantics change; recorded in scorecard provenance so A4 never pairs a
# main arm scored under one scorer revision with a paired arm scored under
# another (judge-vs-lexical mode is already checked separately).
SCORER_REVISION = "r1-gate-respec-1"

from bench.score import (  # noqa: E402
    CLASSES,
    CONFIDENCE_ORDER,
    compute_query_metrics,
    est_cost_usd,
    gates,
    gold_verdict as _lexical_gold,
    load_json,
)


_GOLD_LABELS = ("correct", "incorrect", "contested")


def gold_revision(gold_dir: Path, query_ids: list[str]) -> str:
    """Deterministic revision of the gold sheets backing ``query_ids``
    (sha1 over sorted ids + sheet bytes; a missing sheet hashes a marker so
    presence changes also alter the revision). Recorded in scorecard
    provenance at scoring time and re-derived from the CURRENT sheets when a
    paired arm is loaded, so A4 can never compare metrics produced under two
    different gold revisions."""
    h = hashlib.sha1()
    for qid in sorted(set(query_ids)):
        h.update(qid.encode("utf-8"))
        p = Path(gold_dir) / f"{qid}.json"
        h.update(p.read_bytes() if p.exists() else b"<no-sheet>")
    return h.hexdigest()[:12]


def _scored_gold_ids(entries: list[dict], gold_dir: Path) -> list[str]:
    """Ids of ok, metric-bearing scorecard entries whose gold sheet exists
    (structure-only no-sheet queries are excluded — they carry no gold to
    bind). Used to record the gold revision a scorecard was scored under."""
    return sorted({e["id"] for e in entries
                   if e.get("ok") and e.get("metrics")
                   and (Path(gold_dir) / f"{e['id']}.json").exists()})


def _load_optional_paired(paired_arm: str | None, *, parser=None,
                          require_crosscheck: str | None = None,
                          require_gold_judge_on: bool | None = None,
                          expected: list[dict] | None = None,
                          gold_dir: Path | None = None) \
        -> list[dict] | None:
    """Resolve --paired-arm to the other arm's metric list (A4 same-query
    pairing) or None. Errors abort loudly — a silently-ignored paired arm
    would report A4 as n/a and look like a real result."""
    if not paired_arm:
        return None
    try:
        out = load_paired_metrics(paired_arm,
                                  require_crosscheck=require_crosscheck,
                                  require_gold_judge_on=require_gold_judge_on,
                                  expected=expected,
                                  gold_dir=gold_dir)
    except (ValueError, OSError) as e:
        if parser is not None:
            parser.error(f"paired arm rejected: {e}")
        raise
    if not out and parser is not None:
        parser.error("paired arm has no scored queries to compare against")
    return out


def _load_optional_reruns(rerun_dirs: str | None, *, parser=None,
                          queries: list[dict] | None = None) \
        -> list[list[dict]] | None:
    """Resolve --rerun-dirs into A6 rerun groups (>=3 same-query ledgers).
    Duplicate paths are deduplicated and at least THREE DISTINCT run dirs are
    required — passing the same run three times must not fabricate reruns."""
    if not rerun_dirs:
        return None
    seen: set[Path] = set()
    dirs: list[Path] = []
    for s in rerun_dirs.split(","):
        if not s.strip():
            continue
        p = Path(s).resolve()
        if p not in seen:
            seen.add(p)
            dirs.append(p)
    if len(dirs) < 3 and parser is not None:
        parser.error("determinism needs at least 3 DISTINCT rerun dirs "
                     "(--rerun-dirs)")
    return collect_rerun_groups(dirs, queries or [])


def load_paired_metrics(run_dir: Path, *,
                        require_crosscheck: str | None = None,
                        require_gold_judge_on: bool | None = None,
                        expected: list[dict] | None = None,
                        gold_dir: Path | None = None) -> list[dict]:
    """Load the OTHER arm's per-query metric list for a same-query A4
    comparison (re-spec A4). Returns the metric dicts of that run's ok,
    scored queries, with ``query_id`` injected from the scorecard entry when
    the scorer predates the query_id field so pairing still works.

    Validation (Codex): the paired arm must be the OPPOSITE cross-check arm
    (``require_crosscheck``), scored under the SAME gold-judge mode
    (``require_gold_judge_on``; judge vs lexical compute precision
    differently), each paired query's recorded text/class must match the
    main arm's expected query, and when ``gold_dir`` is given the paired
    run must have been scored under the SAME gold revision and scorer
    revision as the current run — pairing unrelated missions, two enabled
    arms, mixed scoring modes, or stale gold/scorer semantics must never
    produce a gate.

    A ``scorecard-rescore.json`` (a re-score of this run under the current
    gold/scorer) is preferred over the original ``scorecard.json`` when
    present — the owner can refresh a stale paired arm without new paid
    missions."""
    rescore_sc = Path(run_dir) / "scorecard-rescore.json"
    sc = rescore_sc if rescore_sc.exists() \
        else Path(run_dir) / "scorecard.json"
    if not sc.exists():
        raise ValueError(f"paired-arm run has no scorecard.json: {run_dir}")
    data = load_json(sc)
    prov = data.get("provenance", {})
    if require_crosscheck is not None:
        cc = prov.get("crosscheck")
        if cc is None:
            raise ValueError("paired-arm scorecard records no crosscheck arm; "
                             "re-run or --rescore the paired arm under the "
                             "current code so A4 can verify it is the OTHER arm")
        if cc != require_crosscheck:
            raise ValueError(f"paired-arm run crosscheck is "
                             f"{cc!r}, expected "
                             f"{require_crosscheck!r} (it must be the OTHER arm)")
    if require_gold_judge_on is not None:
        gj = prov.get("gold_judge")
        if gj is None:
            raise ValueError("paired-arm scorecard records no gold_judge mode; "
                             "cannot verify scoring-mode parity")
        if gj.startswith("on") is not require_gold_judge_on:
            raise ValueError(f"paired-arm gold_judge {gj!r} does not match the "
                             f"main arm's scoring mode (judge vs lexical "
                             f"compute precision differently)")
    if gold_dir is not None:
        # Gold + scorer binding: the paired metrics were computed against the
        # gold sheets and scorer semantics recorded in ITS provenance. If the
        # sheets or score.py semantics changed since that run, A4 would compare
        # current main-arm precision against stale paired-arm precision — reject
        # loudly and tell the owner how to refresh (no paid missions needed).
        rec_scorer = prov.get("scorer_rev")
        if rec_scorer != SCORER_REVISION:
            raise ValueError(
                f"paired-arm run was scored under scorer revision "
                f"{rec_scorer!r}, current is {SCORER_REVISION!r}; re-run or "
                f"--rescore the paired arm under the current code so A4 "
                f"compares identical scorer semantics")
        recorded_gold = prov.get("gold_rev")
        paired_scored = _scored_gold_ids(data.get("queries", []), gold_dir)
        current_gold = gold_revision(gold_dir, paired_scored)
        if recorded_gold != current_gold:
            raise ValueError(
                f"paired-arm run was scored under gold revision "
                f"{recorded_gold!r}, current gold revision is "
                f"{current_gold!r}; gold sheets changed since that run — "
                f"re-run or --rescore the paired arm under the current gold "
                f"so A4 compares identical gold semantics")
    expected_by_id = {q.get("id"): q for q in (expected or [])
                      if q.get("id")}
    out: list[dict] = []
    for e in data.get("queries", []):
        if not isinstance(e, dict) or not e.get("ok") or not e.get("metrics"):
            continue
        qid = e.get("id")
        exp = expected_by_id.get(qid) if expected_by_id else None
        if expected_by_id and exp is None:
            raise ValueError(f"paired-arm query {qid!r} is not in the main "
                             f"arm's query set")
        if exp is not None:
            if e.get("query") and exp.get("query") \
                    and e["query"] != exp["query"]:
                raise ValueError(f"paired-arm query {qid!r} text drifted "
                                 f"from the main arm")
            if e.get("class") and exp.get("class") \
                    and e["class"] != exp["class"]:
                raise ValueError(f"paired-arm query {qid!r} class drifted "
                                 f"from the main arm")
        m = dict(e["metrics"])
        m.setdefault("query_id", qid)
        out.append(m)
    return out


def collect_rerun_groups(run_dirs: list[Path],
                         queries: list[dict]) -> list[list[dict]]:
    """Collect determinism rerun ledgers (re-spec A6): for every query that
    has a ledger in >= 3 of ``run_dirs``, group those ledgers. Each returned
    group is one query's >= 3 same-query reruns, ready for ``gates(
    rerun_groups=...)``. Fewer than 3 usable reruns -> the query contributes
    nothing (the gate stays n/a for it).

    Validation (Codex): a rerun ledger whose embedded ``query`` text differs
    from the expected query is NOT a rerun of that question — it is skipped
    like a corrupt ledger, so A6 can never pass on mismatched missions."""
    groups: list[list[dict]] = []
    for q in queries:
        qid = q.get("id")
        qtext = q.get("query")
        if not qid:
            continue
        ledgers = []
        for d in run_dirs:
            lp = Path(d) / qid / "ledger.json"
            if not lp.exists():
                continue
            try:
                ledger = load_json(lp)
            except Exception:  # noqa: BLE001 - one corrupt rerun must not
                continue       # drop the whole determinism arm
            if qtext and ledger.get("query") and ledger["query"] != qtext:
                continue      # different mission, not a rerun of this query
            ledgers.append(ledger)
        if len(ledgers) >= 3:
            groups.append(ledgers)
    return groups


def preflight_errors(queries: list[dict], gold_dir: Path) -> list[str]:
    """Validate query list + every selected gold sheet BEFORE any paid mission
    runs. A malformed sheet must fail loudly up front — failing mid-run after
    spending money (and discarding earlier progress) is unacceptable."""
    errs: list[str] = []
    ids = [q.get("id") for q in queries]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    for i in dups:
        errs.append(f"queries.json: duplicate query id {i!r}")
    for q in queries:
        if not q.get("id") or not q.get("query"):
            errs.append("queries.json: entry missing id or query text")
        elif q.get("class") not in CLASSES:
            errs.append(f"queries.json: {q.get('id')!r} has unknown class "
                        f"{q.get('class')!r} (expected one of {CLASSES})")
    for q in queries:
        qid = q.get("id")
        gpath = gold_dir / f"{qid}.json"
        if not gpath.exists():
            continue  # missing sheet = structure-only metrics, allowed
        try:
            g = load_json(gpath)
        except Exception as e:  # noqa: BLE001 - report the filename
            errs.append(f"gold/{qid}.json: unreadable JSON: {e}")
            continue
        if g.get("query_id") != qid:
            errs.append(f"gold/{qid}.json: query_id {g.get('query_id')!r} "
                        f"!= query id {qid!r}")
        if g.get("class") != q.get("class"):
            errs.append(f"gold/{qid}.json: class {g.get('class')!r} "
                        f"!= query class {q.get('class')!r}")
        expected = g.get("expected_claims")
        if not isinstance(expected, list) or not expected:
            errs.append(f"gold/{qid}.json: expected_claims must be a "
                        f"non-empty list")
            continue
        for i, e in enumerate(expected):
            stmt = e.get("statement") if isinstance(e, dict) else None
            if not isinstance(stmt, str) or not stmt.strip():
                errs.append(f"gold/{qid}.json: expected_claims[{i}] missing "
                            f"statement")
            if e.get("gold_label") not in _GOLD_LABELS:
                errs.append(f"gold/{qid}.json: expected_claims[{i}] "
                            f"gold_label {e.get('gold_label')!r} not in "
                            f"{_GOLD_LABELS}")
            if e.get("confidence_class") not in CONFIDENCE_ORDER:
                errs.append(f"gold/{qid}.json: expected_claims[{i}] "
                            f"confidence_class {e.get('confidence_class')!r} "
                            f"not in {CONFIDENCE_ORDER}")
        stmts = [e.get("statement") for e in expected
                 if isinstance(e, dict) and e.get("statement")]
        for i, e in enumerate(expected):
            if isinstance(e, dict) and e.get("variant_of"):
                if e["variant_of"] not in stmts:
                    errs.append(f"gold/{qid}.json: expected_claims[{i}] "
                                f"variant_of does not match any statement "
                                f"in this sheet")
                elif e["statement"] == e["variant_of"]:
                    errs.append(f"gold/{qid}.json: expected_claims[{i}] "
                                f"variant_of points at itself")
    return errs


def _assess(q: dict, qtext: str, qout: Path, gold_dir: Path,
             judge_enabled: bool, judge_log: Path) -> dict:
    """Score one query's artifacts in ``qout`` (ledger.json) against its gold
    sheet, judging each claim when enabled. Judge traffic appends to
    ``judge_log`` and the estimate is read from that file only, so judging
    cost stays visible and separate from the mission's own llm.log when
    requested (rescore uses a dedicated llm-rescore.log; the execute path
    passes the mission llm.log after the child has appended to it)."""
    from bench.judge import make_claim_judge
    from veritas.llm import DeepSeekClient

    entry: dict = {"id": q["id"], "class": q.get("class"), "query": qtext,
                   "ok": True, "est_cost_usd": 0.0, "metrics": {}}
    gold_path = gold_dir / f"{q['id']}.json"
    gold = load_json(gold_path) if gold_path.exists() else None
    claim_judge = None
    judge_state = None
    try:
        ledger = load_json(qout / "ledger.json")
    except Exception as e:  # noqa: BLE001
        entry["ok"] = False
        entry["error"] = f"ledger unreadable: {e}"
    else:
        if judge_enabled and gold:
            claim_judge, judge_state = make_claim_judge(
                DeepSeekClient(log=str(judge_log)))
        try:
            entry["metrics"] = compute_query_metrics(
                ledger, gold, claim_judge=claim_judge)
        except Exception as e:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = f"score parse failed: {e}"
        entry["judge_fallbacks"] = judge_state["fallbacks"] if judge_state else 0
        entry["judge_mode"] = ("judge" if judge_state else
                               "lexical" if gold is not None else "no-gold")
        if gold is None:
            entry["note"] = "no gold sheet — structure-only metrics"
    log_text = judge_log.read_text(encoding="utf-8", errors="replace") \
        if judge_log.exists() else ""
    entry["est_cost_usd"] = round(est_cost_usd(log_text), 6)
    return entry


def _rescore_main(run_dir: Path, queries: list[dict], gold_dir: Path,
                  judge_enabled: bool, relevance: list[int] | None,
                  no_crosscheck: bool, cap_usd: float,
                  crosscheck: str = "on",
                  paired_metrics: list[dict] | None = None,
                  rerun_groups: list[list[dict]] | None = None) -> int:
    """Score an existing run dir without new missions. Relevance judgements
    are bound to this run's sample by the caller; the judge spend respects
    the cap. ``paired_metrics`` (the cross-check-off arm's metrics) and
    ``rerun_groups`` (>=3 same-query rerun ledgers) feed the re-spec A4/A6
    gates when the owner supplies them. ``crosscheck`` is the ORIGINAL
    run's arm (from its scorecard provenance), recorded in the rescore
    artifact so a rescored run is self-describing when later used as a
    paired arm; gold/scorer revisions are bound the same way."""
    per_query = []
    capped = False
    total_usd = 0.0
    for idx, q in enumerate(queries):
        if capped:
            per_query.append({"id": q["id"], "class": q.get("class"),
                              "query": q.get("query", ""), "ok": False,
                              "skipped_cap": True, "metrics": {}})
            continue
        if q.get("_unresolved"):
            per_query.append({"id": q["id"], "class": q.get("class"),
                              "query": q.get("query", ""), "ok": False,
                              "error": q.get("_reason", "unresolved query"),
                              "metrics": {}})
            continue
        qout = run_dir / q["id"]
        if not (qout / "ledger.json").exists():
            per_query.append({"id": q["id"], "class": q.get("class"),
                              "query": q.get("query", ""), "ok": False,
                              "error": "no ledger in run dir", "metrics": {}})
            continue
        entry = _assess(q, q.get("query") or "", qout, gold_dir, judge_enabled,
                        qout / "llm-rescore.log")
        per_query.append(entry)
        total_usd += entry.get("est_cost_usd", 0.0)
        if total_usd >= cap_usd and queries[idx + 1:]:
            capped = True
            print(f"[bench] rescore cap ${cap_usd:.2f} reached "
                  f"(cumulative ${total_usd:.4f}) — skipping the remaining "
                  f"queries' judging (scorecard capped-partial)")
    completed = [e for e in per_query if e.get("ok") and e.get("metrics")]
    agg = gates([e["metrics"] for e in completed],
                relevance_judgements=relevance,
                q_metrics_nocc=paired_metrics,
                rerun_groups=rerun_groups)
    n_failed = sum(1 for e in per_query if not e.get("ok")
                   and not e.get("skipped_cap"))
    valid = n_failed == 0 and not capped and len(completed) == len(queries) \
        and not no_crosscheck
    out_path = run_dir / "scorecard-rescore.json"
    scorecard = {
        "provenance": {
            "mode": "rescore",
            "of_run": run_dir.name,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
                .isoformat(),
            "crosscheck": crosscheck,
            "scorer_rev": SCORER_REVISION,
            "gold_rev": gold_revision(gold_dir, _scored_gold_ids(
                per_query, gold_dir)),
            "gold_judge": ("on" if judge_enabled
                           else "off(--no-judge or no backend)"),
            "relevance_judgements": len(relevance) if relevance else 0,
            "judge_note": "judge cost in each query's llm-rescore.log",
            "paired_arm": ("on" if paired_metrics else "off"),
            "rerun_groups": len(rerun_groups) if rerun_groups else 0,
            "capped_partial": capped,
        },
        "valid": valid,
        "n_failed": n_failed,
        "queries": per_query,
        "gates": agg,
    }
    out_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False))
    print(f"[bench] rescored {len(per_query)} query(s) from {run_dir}")
    print(f"[bench] scorecard: {out_path}  "
          f"({'VALID' if valid else 'INVALID (advisory)'})")
    for gate, g in agg.items():
        ok = "PASS" if g["ok"] is True else ("FAIL" if g["ok"] is False else "n/a")
        print(f"  {gate}: {ok}  value={g['value']}")
    return 0


def read_keyed_relevance(path: str | Path) -> tuple[list[int], str | None]:
    """Read a judgements file: the collector's keyed object
    {"sample_sha": ..., "judgements": [0/1...]} or a legacy plain list
    (sha None). Garbage fails loudly."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        judgements = data.get("judgements")
        sha = data.get("sample_sha")
        if not isinstance(judgements, list) or not isinstance(sha, str):
            raise ValueError(f"relevance file {path}: keyed judgements need "
                             f"'judgements' list + 'sample_sha'")
    elif isinstance(data, list):
        judgements, sha = data, None
    else:
        raise ValueError(f"relevance file {path}: must be a list or a keyed "
                         f"object")
    out = []
    for i, v in enumerate(judgements):
        if isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1):
            raise ValueError(f"relevance file {path}: item {i} is not 0/1 "
                             f"(got {v!r})")
        out.append(v)
    return out, sha


def relevance_binding_error(judgements_sha: str | None,
                            sample_path: Path) -> str | None:
    """A5 judgements MUST be bound to THIS run's relevance-sample.json by
    sha — same-length samples from different runs are different sources, so
    length alone never suffices and a plain un-keyed list cannot be bound."""
    if not sample_path.exists():
        return (f"relevance judgements given but no relevance-sample.json in "
                f"the run dir — collect one first (collect_relevance.py)")
    try:
        sample_sha = hashlib.sha1(
            sample_path.read_text(encoding="utf-8").encode()).hexdigest()[:16]
    except OSError:
        return f"run relevance-sample.json unreadable: {sample_path}"
    if judgements_sha is None:
        return (f"relevance judgements are not bound to this run (plain "
                f"list) — regenerate with collect_relevance.py so they carry "
                f"the run's sample_sha")
    if judgements_sha != sample_sha:
        return (f"relevance judgements sample_sha {judgements_sha} does not "
                f"match this run's {sample_sha} — they describe different "
                f"run samples")
    return None


def select_queries(queries: list[dict], ids: str) -> tuple[list[dict], list[str]]:
    """Filter ``queries`` by comma-separated ``ids``; return (chosen, unknown).
    Unknown ids must be rejected by the caller — silently dropping a typo'd
    mission would let the reduced run look like a complete benchmark."""
    want = [s.strip() for s in ids.split(",") if s.strip()]
    by_id = {q["id"]: q for q in queries}
    chosen = [by_id[w] for w in want if w in by_id]
    unknown = [w for w in want if w not in by_id]
    return chosen, unknown


def parse_relevance(path: str | Path) -> list[int]:
    """Load and validate a binary 0/1 relevance-judgement list.
    Raises ValueError on malformed input — garbage must fail loudly, never
    skew A5 into a misleading pass (e.g. a stray 2 median >= 0.7)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"relevance file {path} must be a JSON list of 0/1")
    out: list[int] = []
    for i, v in enumerate(data):
        if isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1):
            raise ValueError(f"relevance file {path}: item {i} is not 0/1 "
                             f"(got {v!r})")
        out.append(v)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", default=str(REPO / "bench" / "queries.json"))
    p.add_argument("--out", default=str(Path.cwd() / "out" / "bench"),
                   help="parent dir; each invocation writes its own <run-id>/ "
                        "subdir (scorecard + per-query ledgers + llm.log)")
    p.add_argument("--run-id", default=None,
                   help="label this invocation (arm/repeat). Default: "
                        "<cc|nocc>-<timestamp> so paired/determinism arms "
                        "never clobber each other's artifacts")
    p.add_argument("--ids", default=None,
                   help="comma list of query ids to run (default: all)")
    p.add_argument("--cap-usd", type=float, default=1.0,
                   help="stop after cumulative estimated cost passes this USD")
    p.add_argument("--no-crosscheck", action="store_true",
                   help="add --no-crosscheck to each mission (paired arm)")
    p.add_argument("--no-judge", action="store_true",
                   help="skip the LLM gold judge; use lexical gold matching "
                        "(default: judge on when a reasoning backend exists)")
    p.add_argument("--rescore", metavar="RUN_DIR", default=None,
                   help="score an EXISTING run dir (no new missions): load "
                        "its ledgers, apply gold + judge (unless --no-judge) "
                        "+ --relevance judgements, write scorecard-rescore.json")
    p.add_argument("--paired-arm", metavar="RUN_DIR", default=None,
                   help="cross-check-off arm's run dir for the A4 same-query "
                        "paired comparison (its scorecard metrics pair with "
                        "this run by query_id; rescore + execute)")
    p.add_argument("--rerun-dirs", default=None,
                   help="comma-separated determinism run dirs (>=3 per query) "
                        "whose same-query ledgers feed the re-spec A6 "
                        "distribution gate (rescore + execute)")
    p.add_argument("--relevance", default=None,
                   help="optional JSON file: list of 0/1 rubric judgements")
    args = p.parse_args()

    spec = load_json(args.queries)
    queries = spec["queries"]
    if args.ids:
        queries, unknown = select_queries(queries, args.ids)
        if unknown:
            p.error(f"unknown query id(s): {', '.join(unknown)}")
    gold_dir = REPO / "bench" / "gold"
    errs = preflight_errors(queries, gold_dir)
    if errs:
        p.error("pre-flight validation failed:\n  " + "\n  ".join(errs))

    # LLM gold judge (default on when a reasoning backend exists). The judge
    # appends to the query's llm.log so judging cost stays inside the meter.
    judge_enabled = False
    if not args.no_judge:
        try:
            from veritas.config import settings as _settings
            judge_enabled = _settings.has_reasoning_backend()
        except Exception:  # noqa: BLE001 - settings import must not kill the run
            judge_enabled = False

    if args.rescore:
        run_dir = Path(args.rescore)
        orig_scorecard = run_dir / "scorecard.json"
        if not orig_scorecard.exists():
            p.error(f"rescore requires the run's scorecard.json "
                    f"(not found in {run_dir})")
        orig = load_json(orig_scorecard)
        prov = orig.get("provenance", {})
        nocc_orig = prov.get("crosscheck") == "off"
        recorded = {e["id"]: e for e in orig.get("queries", [])
                    if isinstance(e, dict)}
        orig_ids = prov.get("query_ids") or list(recorded)

        # resolve the queries to rescore (run set, or --ids subset) against
        # the CURRENT spec, rejecting ids that cannot be scored faithfully
        want = [i.strip() for i in args.ids.split(",")] if args.ids \
            else orig_ids
        by_id = {q["id"]: q for q in queries}
        resolved = []
        for i in want:
            rec = recorded.get(i)
            if rec is None:
                resolved.append({"id": i, "class": None, "query": "",
                                 "_unresolved": True,
                                 "_reason": "not part of this run"})
                continue
            cur = by_id.get(i)
            if cur is None:
                resolved.append({"id": i, "class": None, "query": "",
                                 "_unresolved": True,
                                 "_reason": "missing from queries spec"})
                continue
            if rec.get("query") and rec["query"] != cur["query"]:
                resolved.append({"id": i, "class": cur.get("class"),
                                 "query": cur["query"], "_unresolved": True,
                                 "_reason": "query text drifted since the run"})
                continue
            if rec.get("class") and rec["class"] != cur.get("class"):
                resolved.append({"id": i, "class": cur.get("class"),
                                 "query": cur["query"], "_unresolved": True,
                                 "_reason": "query class drifted since the run"})
                continue
            resolved.append(cur)

        # A5 judgements must be bound to this exact run's sample
        relevance_list = None
        if args.relevance:
            judgements, sha = read_keyed_relevance(args.relevance)
            err = relevance_binding_error(sha, run_dir / "relevance-sample.json")
            if err:
                p.error(f"relevance rejected: {err}")
            relevance_list = judgements
        # A4's main position is the cross-check-ON arm: gates() reads
        # ``q_metrics`` as the with-arm. Rescoring an off-arm run with a
        # paired cc-arm would evaluate the deltas in reverse (a genuinely
        # beneficial pair reported as failure) — reject it (Codex).
        main_cc = prov.get("crosscheck") != "off"
        if not main_cc and args.paired_arm:
            p.error("--paired-arm with a crosscheck=off main run would "
                    "reverse A4; rescore the crosscheck=on run instead and "
                    "pass the off-arm as --paired-arm")
        paired_metrics = _load_optional_paired(
            args.paired_arm, parser=p,
            require_crosscheck="off" if main_cc else "on",
            require_gold_judge_on=judge_enabled,
            expected=resolved,
            gold_dir=gold_dir)
        rerun_groups = _load_optional_reruns(args.rerun_dirs, parser=p,
                                             queries=resolved)
        if args.rerun_dirs and not rerun_groups:
            p.error("--rerun-dirs produced no usable determinism group: no "
                    "query has >=3 readable same-query rerun ledgers in the "
                    "given dirs")
        return _rescore_main(run_dir, resolved, gold_dir, judge_enabled,
                             relevance_list, args.no_crosscheck or nocc_orig,
                             args.cap_usd,
                             crosscheck="off" if nocc_orig else "on",
                             paired_metrics=paired_metrics,
                             rerun_groups=rerun_groups)

    # execute path: --relevance is a plain 0/1 list (no sample to bind)
    relevance = None
    if args.relevance:
        try:
            relevance = parse_relevance(args.relevance)
        except ValueError as e:
            p.error(str(e))

    arm = "nocc" if args.no_crosscheck else "cc"
    run_id = args.run_id or f"{arm}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    out_root = Path(args.out) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    extra = ["--no-crosscheck"] if args.no_crosscheck else []

    # Resolve comparison inputs BEFORE any paid mission runs (Codex): a
    # malformed paired scorecard or rerun dir must fail before spending the
    # budget, never after the loop has consumed it.
    if args.no_crosscheck and args.paired_arm:
        # A fresh --no-crosscheck mission IS the off-arm; pairing it with a
        # cc run would reverse A4 (q_metrics is always the with-arm).
        p.error("--paired-arm with a --no-crosscheck run would reverse A4; "
                "run the mainline (crosscheck on) run and pass the off-arm "
                "as --paired-arm")
    paired_metrics = _load_optional_paired(
        args.paired_arm, parser=p,
        require_crosscheck="off" if not args.no_crosscheck else "on",
        require_gold_judge_on=judge_enabled,
        expected=queries,
        gold_dir=gold_dir)
    rerun_groups = _load_optional_reruns(args.rerun_dirs, parser=p,
                                         queries=queries)
    if args.rerun_dirs and not rerun_groups:
        p.error("--rerun-dirs produced no usable determinism group: no query "
                "has >=3 readable same-query rerun ledgers in the given dirs")

    print(f"[bench] {len(queries)} query(s), cap ${args.cap_usd:.2f}, "
          f"crosscheck={'off' if args.no_crosscheck else 'on'}, "
          f"gold_judge={'on' if judge_enabled else 'off'}")
    total_usd = 0.0
    per_query = []
    capped = False  # True only once a remaining query is actually skipped
    for idx, q in enumerate(queries):
        if capped:
            per_query.append(_skipped(q))
            continue
        qid, qtext = q["id"], q["query"]
        qout = out_root / qid
        qout.mkdir(parents=True, exist_ok=True)
        llm_log = qout / "llm.log"
        if llm_log.exists():
            llm_log.unlink()  # fresh file: meter only this invocation
        env = dict(os.environ, VERITAS_LLM_LOG=str(llm_log))
        cmd = [sys.executable, "-m", "veritas.cli", "run", qtext,
               "--surfaces", "web", "--outdir", str(qout), "--quiet", *extra]
        print(f"[bench] {qid} ({q.get('class')}): {qtext[:70]}...")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              cwd=str(REPO))  # run the checked-out code
        entry: dict = {
            "id": qid,
            "class": q.get("class"),
            "query": qtext,
            "ok": proc.returncode == 0,
            "est_cost_usd": 0.0,
            "metrics": {},
        }
        gold_path = gold_dir / f"{qid}.json"
        gold = load_json(gold_path) if gold_path.exists() else None
        claim_judge = None
        judge_state = None
        if proc.returncode != 0:
            entry["error"] = (proc.stderr or proc.stdout)[-500:]
        else:
            try:
                ledger = load_json(qout / "ledger.json")
            except Exception as e:  # noqa: BLE001 - never lose the mission record
                entry["ok"] = False
                entry["error"] = f"ledger unreadable: {e}"
            else:
                if judge_enabled and gold:
                    # Judge each claim against the gold facts (temp 0, same
                    # backend); transport failures fall back to the lexical
                    # matcher per claim and are counted.
                    from bench.judge import make_claim_judge
                    from veritas.llm import DeepSeekClient

                    claim_judge, judge_state = make_claim_judge(
                        DeepSeekClient(log=str(llm_log)))

                try:
                    entry["metrics"] = compute_query_metrics(
                        ledger, gold, claim_judge=claim_judge)
                except Exception as e:  # noqa: BLE001
                    entry["ok"] = False
                    entry["error"] = f"score parse failed: {e}"
                entry["judge_fallbacks"] = (
                    judge_state["fallbacks"] if judge_state else 0)
                entry["judge_mode"] = (
                    "judge" if judge_state else
                    "lexical" if gold is not None else "no-gold")
                if gold is None:
                    entry["note"] = "no gold sheet — structure-only metrics"
        log_text = llm_log.read_text(encoding="utf-8", errors="replace") \
            if llm_log.exists() else ""
        entry["est_cost_usd"] = round(est_cost_usd(log_text), 6)
        total_usd += entry["est_cost_usd"]
        per_query.append(entry)

        remaining = queries[idx + 1:]
        if total_usd >= args.cap_usd and remaining:
            capped = True
            print(f"[bench] cap ${args.cap_usd:.2f} reached "
                  f"(cumulative ${total_usd:.4f}) — skipping "
                  f"{len(remaining)} remaining (scorecard capped-partial)")
        else:
            print(f"[bench]   est ${entry['est_cost_usd']:.5f} "
                  f"(cumulative ${total_usd:.5f})")

    completed = [e for e in per_query if e.get("ok") and e.get("metrics")]
    agg = gates([e["metrics"] for e in completed],
                relevance_judgements=relevance,
                q_metrics_nocc=paired_metrics,
                rerun_groups=rerun_groups)
    n_failed = sum(1 for e in per_query if not e.get("ok")
                   and not e.get("skipped_cap"))
    n_skipped = sum(1 for e in per_query if e.get("skipped_cap"))
    # Gates over a subset are advisory only: a failed/skipped mission must
    # not let the rest silently pass as a complete benchmark. Cross-check-off
    # runs are the paired arm — their A1-A6 lines are never authoritative.
    valid = (not capped and n_failed == 0 and len(completed) == len(queries)
             and not args.no_crosscheck)

    reasons = []
    if n_failed:
        reasons.append(f"{n_failed} failed")
    if n_skipped:
        reasons.append(f"{n_skipped} skipped")
    if capped:
        reasons.append("capped-partial")
    if args.no_crosscheck:
        reasons.append("no-crosscheck arm (paired comparison only)")
    invalid_reason = ", ".join(reasons) if reasons else ""

    provenance = {
        "run_id": run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "queries_revision": spec.get("revision", "?"),
        "query_ids": [q["id"] for q in queries],
        "queries_sha": hashlib.sha1(
            json.dumps(queries, sort_keys=True).encode()).hexdigest()[:12],
        "scorer_rev": SCORER_REVISION,
        "gold_rev": gold_revision(gold_dir, _scored_gold_ids(
            per_query, gold_dir)),
        "crosscheck": "off" if args.no_crosscheck else "on",
        "gold_judge": ("off(--no-judge)" if args.no_judge
                        else "off(no reasoning backend)" if not judge_enabled
                        else "on"),
        "backend_note": "deepseek-chat (see bench/README.md)",
        "cap_usd": args.cap_usd,
        "capped_partial": capped,
        "total_est_cost_usd": round(total_usd, 6),
        "cost_note": "estimate from VERITAS_LLM_LOG chars; "
                     "see bench/score.py constants",
    }
    scorecard = {"provenance": provenance,
                 "valid": valid,
                 "n_failed": n_failed,
                 "n_skipped_cap": n_skipped,
                 "queries": per_query,
                 "gates": agg}
    out_path = out_root / "scorecard.json"
    out_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False))
    print(f"[bench] run dir: {out_root}")
    print(f"[bench] scorecard: {out_path}")
    print(f"[bench] cumulative est cost: ${total_usd:.4f} "
          f"(cap ${args.cap_usd:.2f}{' — CAPPED-PARTIAL' if capped else ''})")
    if not valid:
        print(f"[bench] scorecard INVALID ({invalid_reason}) — "
              f"A1-A6 lines below are advisory only")
    else:
        print("[bench] scorecard VALID — A1-A6 lines below are authoritative")
    for gate, g in agg.items():
        ok = "PASS" if g["ok"] is True else ("FAIL" if g["ok"] is False else "n/a")
        print(f"  {gate}: {ok}  value={g['value']}")
    return 0


def _skipped(q: dict) -> dict:
    return {"id": q["id"], "class": q.get("class"), "query": q["query"],
            "ok": False, "skipped_cap": True, "est_cost_usd": 0.0,
            "metrics": {}}


if __name__ == "__main__":
    raise SystemExit(main())
