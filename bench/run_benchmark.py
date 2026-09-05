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

Scoring is done by ``bench/score.py`` (pure, tested); the paired cross-check
delta (A4 second half) and determinism reruns need repeated invocations with
``--no-crosscheck`` / a manual rerun — the scorecard marks those gates
``na`` until the inputs exist.
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
                  no_crosscheck: bool) -> int:
    """Score an existing run dir without new missions: relevance judgements
    must be scored against the run that produced the sources they describe."""
    if not run_dir.is_dir():
        print(f"[bench] rescore: no such run dir: {run_dir}", file=sys.stderr)
        return 2
    if relevance is not None:
        err = relevance_binding_error(len(relevance), run_dir /
                                      "relevance-sample.json")
        if err:
            print(f"[bench] rescore aborted: {err}", file=sys.stderr)
            return 2
    per_query = []
    for q in queries:
        if q.get("_unresolved"):
            per_query.append({"id": q["id"], "class": None, "query": "",
                              "ok": False,
                              "error": "query id recorded in the run is "
                                       "missing from the queries spec",
                              "metrics": {}})
            continue
        qout = run_dir / q["id"]
        if not (qout / "ledger.json").exists():
            per_query.append({"id": q["id"], "class": q.get("class"),
                              "query": q["query"], "ok": False,
                              "error": "no ledger in run dir", "metrics": {}})
            continue
        per_query.append(_assess(q, q["query"], qout, gold_dir, judge_enabled,
                                 qout / "llm-rescore.log"))
    completed = [e for e in per_query if e.get("ok") and e.get("metrics")]
    agg = gates([e["metrics"] for e in completed],
                relevance_judgements=relevance)
    n_failed = sum(1 for e in per_query if not e.get("ok"))
    valid = n_failed == 0 and len(completed) == len(queries) \
        and not no_crosscheck
    out_path = run_dir / "scorecard-rescore.json"
    scorecard = {
        "provenance": {
            "mode": "rescore",
            "of_run": run_dir.name,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
                .isoformat(),
            "gold_judge": ("on" if judge_enabled else "off(--no-judge or no backend)"),
            "relevance_judgements": len(relevance) if relevance else 0,
            "judge_note": "judge cost in each query's llm-rescore.log",
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



def relevance_binding_error(n_judgements: int, sample_path: Path) -> str | None:
    """A5 judgements describe THIS run's relevance-sample.json, in its order
    and length — a stray [1] must never make A5 pass on an unrelated run."""
    if not sample_path.exists():
        return (f"relevance judgements given but no relevance-sample.json in "
                f"the run dir — collect one first (collect_relevance.py)")
    try:
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"run relevance-sample.json is unreadable: {sample_path}"
    if len(sample) != n_judgements:
        return (f"relevance judgements ({n_judgements}) do not match the "
                f"run's {len(sample)}-source sample — collect/refill "
                f"relevance-sample.json for this run")
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
    p.add_argument("--relevance", default=None,
                   help="optional JSON file: list of 0/1 rubric judgements")
    args = p.parse_args()

    spec = load_json(args.queries)
    queries = spec["queries"]
    if args.ids:
        queries, unknown = select_queries(queries, args.ids)
        if unknown:
            p.error(f"unknown query id(s): {', '.join(unknown)}")
    relevance = None
    if args.relevance:
        try:
            relevance = parse_relevance(args.relevance)
        except ValueError as e:
            p.error(str(e))

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
        nocc_orig = False
        orig_ids = []
        if orig_scorecard.exists():
            prov = load_json(orig_scorecard).get("provenance", {})
            nocc_orig = prov.get("crosscheck") == "off"
            orig_ids = prov.get("query_ids", [])
        if args.ids is None and orig_ids:
            # The run's recorded query set is authoritative: a recorded id
            # missing from the current spec, or whose query text/class has
            # drifted, makes the rescore unresolvable — the old ledger would
            # be scored against a definition that never ran it.
            recorded = {e["id"]: e for e in
                        load_json(orig_scorecard).get("queries", [])}
            by_id = {q["id"]: q for q in queries}
            resolved = []
            for i in orig_ids:
                if i not in by_id:
                    resolved.append({"id": i, "class": None, "query": "",
                                     "_unresolved": True,
                                     "_reason": "missing from queries spec"})
                    continue
                rec = recorded.get(i) or {}
                if rec.get("query") and rec["query"] != by_id[i]["query"]:
                    resolved.append({"id": i, "class": by_id[i].get("class"),
                                     "query": by_id[i]["query"],
                                     "_unresolved": True,
                                     "_reason": "query text drifted since "
                                               "the run"})
                    continue
                resolved.append(by_id[i])
            queries = resolved
        return _rescore_main(run_dir, queries, gold_dir, judge_enabled,
                             relevance, args.no_crosscheck or nocc_orig)

    arm = "nocc" if args.no_crosscheck else "cc"
    run_id = args.run_id or f"{arm}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    out_root = Path(args.out) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    extra = ["--no-crosscheck"] if args.no_crosscheck else []

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
                relevance_judgements=relevance)
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
