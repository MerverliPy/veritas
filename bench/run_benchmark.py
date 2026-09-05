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
    compute_query_metrics,
    est_cost_usd,
    gates,
    load_json,
)


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
    p.add_argument("--relevance", default=None,
                   help="optional JSON file: list of 0/1 rubric judgements")
    args = p.parse_args()

    spec = load_json(args.queries)
    queries = spec["queries"]
    if args.ids:
        queries, unknown = select_queries(queries, args.ids)
        if unknown:
            p.error(f"unknown query id(s): {', '.join(unknown)}")
    relevance = parse_relevance(args.relevance) if args.relevance else None

    arm = "nocc" if args.no_crosscheck else "cc"
    run_id = args.run_id or f"{arm}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    out_root = Path(args.out) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    gold_dir = REPO / "bench" / "gold"
    extra = ["--no-crosscheck"] if args.no_crosscheck else []

    print(f"[bench] {len(queries)} query(s), cap ${args.cap_usd:.2f}, "
          f"crosscheck={'off' if args.no_crosscheck else 'on'}")
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
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        log_text = llm_log.read_text(encoding="utf-8", errors="replace") \
            if llm_log.exists() else ""
        cost = est_cost_usd(log_text)
        total_usd += cost

        gold_path = gold_dir / f"{qid}.json"
        gold = None
        if gold_path.exists():
            gold = load_json(gold_path)
            if (gold.get("query_id") != qid
                    or gold.get("class") != q.get("class")):
                print(f"[bench] WARN gold/{qid}.json mismatch "
                      f"(expected query_id={qid}, class={q.get('class')}; "
                      f"got {gold.get('query_id')}/{gold.get('class')}) "
                      f"— ignoring sheet")
                gold = None
        entry: dict = {
            "id": qid,
            "class": q.get("class"),
            "query": qtext,
            "ok": proc.returncode == 0,
            "est_cost_usd": round(cost, 6),
            "metrics": {},
        }
        if proc.returncode != 0:
            entry["error"] = (proc.stderr or proc.stdout)[-500:]
        else:
            ledger = load_json(qout / "ledger.json")
            entry["metrics"] = compute_query_metrics(ledger, gold)
            if gold is None:
                entry["note"] = "no gold sheet — structure-only metrics"
        per_query.append(entry)

        remaining = queries[idx + 1:]
        if total_usd >= args.cap_usd and remaining:
            capped = True
            print(f"[bench] cap ${args.cap_usd:.2f} reached "
                  f"(cumulative ${total_usd:.4f}) — skipping "
                  f"{len(remaining)} remaining (scorecard capped-partial)")
        else:
            print(f"[bench]   est ${cost:.5f} "
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
