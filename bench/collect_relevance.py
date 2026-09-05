#!/usr/bin/env python3
"""A5 relevance-sample collector — OWNER-RUN ONLY (no network, no LLM).

Extracts the A5 rubric sample (spec §5): for a subset of sub-questions per
query, the distinct sources the pipeline cited, so the owner can judge each
source 0/1 "does this source help answer the sub-question?" The marked
judgements feed the benchmark driver's ``--relevance`` file.

Usage:
    python3 bench/collect_relevance.py --run-dir out/bench/cc-judge2
    # reads the sample (relevance-sample.json) + fills relevance-judgements.json
    # (replace each null with 0 or 1, keep order), then score THE SAME run:
    python3 bench/run_benchmark.py --rescore out/bench/cc-judge2 --relevance out/bench/cc-judge2/relevance-judgements.json

Sampling is deterministic (query order from bench/queries.json, then
sub-question text, then source URL), so reruns produce the same sheet
regardless of within-claim evidence order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PASSAGE_LIMIT = 600


def _source_locator(source: dict) -> str:
    return source.get("url") or source.get("path") or "(no locator)"


def extract_sample(ledger: dict, *,
                   max_subquestions: int = 2,
                   sources_per_subquestion: int = 3) -> list[dict]:
    """Deterministic relevance sample from one query's ledger.

    For the first ``max_subquestions`` distinct sub-questions (in claim
    order) with evidence, take up to ``sources_per_subquestion`` distinct
    cited sources (first-seen order). Each entry is one source for one
    sub-question.
    """
    claims = ledger.get("claims", [])
    seen_sub: dict[str, list[dict]] = {}
    for c in claims:
        subq = (c.get("subquestion") or "").strip()
        if not subq or not c.get("evidence"):
            continue
        bucket = seen_sub.setdefault(subq, [])
        for e in c["evidence"]:
            src = (e.get("source") or {})
            loc = _source_locator(src)
            if loc and not any(x["url"] == loc for x in bucket):
                bucket.append({
                    "url": loc,
                    "title": src.get("title") or "",
                    "passage": (e.get("passage") or "")[:PASSAGE_LIMIT],
                })

    out: list[dict] = []
    for subq in list(seen_sub)[:max_subquestions]:
        # Stable order regardless of within-claim evidence order.
        for s in sorted(seen_sub[subq], key=lambda x: x["url"])[
                :sources_per_subquestion]:
            out.append({
                "subquestion": subq,
                "url": s["url"],
                "title": s["title"],
                "passage": s["passage"],
            })
    return out


def judgements_action(prev_sample, new_entries, prev_judgements_filled,
                     force: bool) -> str:
    """Decide what to do with an existing judgements file:
    'preserve' when the sample is unchanged and judgements are filled,
    'write' when there is nothing filled or the sample changed and --force
    is set, 'abort' when the sample changed under a filled file without
    --force (silently keeping judgements for different sources would skew
    A5)."""
    if not prev_judgements_filled:
        return "write"
    if prev_sample == new_entries:
        return "preserve"
    return "write" if force else "abort"


def _latest_run_dir(out_root: Path) -> Path:
    """Newest run by its scorecard file mtime — run-id names mix prefixes
    (cc-/nocc-/det-...), so name ordering is not chronological."""
    dirs = [p for p in out_root.iterdir()
            if p.is_dir() and (p / "scorecard.json").exists()]
    if not dirs:
        raise SystemExit(f"no run dirs (with scorecard.json) under {out_root}")
    return max(dirs, key=lambda p: (p / "scorecard.json").stat().st_mtime)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None,
                   help="run dir (default: newest out/bench/<run-id> with a "
                        "scorecard)")
    p.add_argument("--queries", default=str(REPO / "bench" / "queries.json"),
                   help="query order + text source (bench/queries.json)")
    p.add_argument("--max-subquestions", type=int, default=2)
    p.add_argument("--sources-per-subquestion", type=int, default=3)
    p.add_argument("--sample-out", default=None,
                   help="where to write relevance-sample.json (default: "
                        "next to the run scorecard)")
    p.add_argument("--judgements-out", default=None,
                   help="where to write the null-filled judgements list "
                        "(default: <sample dir>/relevance-judgements.json)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing, already-filled judgements "
                        "file with nulls")
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else \
        _latest_run_dir(Path.cwd() / "out" / "bench")
    spec = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    if args.sample_out:
        sample_out = Path(args.sample_out)
    else:
        sample_out = run_dir / "relevance-sample.json"
    judgements_out = Path(args.judgements_out) if args.judgements_out else \
        sample_out.with_name("relevance-judgements.json")

    entries: list[dict] = []
    for q in spec["queries"]:
        ledger_path = run_dir / q["id"] / "ledger.json"
        if not ledger_path.exists():
            print(f"[collect] skip {q['id']}: no ledger", file=sys.stderr)
            continue
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        for e in extract_sample(ledger,
                                max_subquestions=args.max_subquestions,
                                sources_per_subquestion=args.sources_per_subquestion):
            entries.append({"query_id": q["id"], "class": q.get("class"),
                            "query": q["query"], **e})

    if not entries:
        print("[collect] no sample entries found — nothing to judge",
              file=sys.stderr)
        return 1

    prev_sample = None
    if sample_out.exists():
        try:
            prev_sample = json.loads(sample_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev_sample = None
    prev_filled = False
    malformed = False
    if judgements_out.exists():
        try:
            prev_filled = any(v is not None for v in json.loads(
                judgements_out.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            # Malformed content is NOT evidence the file is safe to reset.
            malformed = True
    if malformed and not args.force:
        print(f"[collect] {judgements_out} is malformed JSON — refusing to "
              f"reset it (fix the file or rerun with --force)",
              file=sys.stderr)
        return 1
    action = judgements_action(prev_sample, entries, prev_filled, args.force)
    if action == "abort":
        print(f"[collect] sample changed since {judgements_out} was filled "
              f"— judgements would describe different sources. Rerun with "
              f"--force to reset them.", file=sys.stderr)
        return 1
    if action == "preserve":
        print(f"[collect] sample unchanged — keeping filled "
              f"{judgements_out}", file=sys.stderr)
    else:
        judgements_out.write_text(json.dumps([None] * len(entries), indent=2))
    sample_out.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"[collect] sample ({len(entries)} sources): {sample_out}")
    print(f"[collect] judgements to fill (null -> 0/1, keep order): "
          f"{judgements_out}")
    print("[collect] then score THE SAME run dir with: run_benchmark.py "
          "--rescore <run-dir> --relevance <filled judgements>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
