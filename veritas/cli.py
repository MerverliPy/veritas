"""Command line interface: run a research mission.

Examples
--------
    veritas run "Why did the WannaCry worm spread so fast?" --surfaces web
    veritas run "What does this repo do and how is auth handled?" \
        --surfaces code --code-root . --outdir out/repo-audit
    veritas run "summarise my notes on the agent-ecosystem plan" \
        --surfaces local --local-root ~/notes
    veritas run "Q?" --surfaces web,local,code --no-crosscheck --quiet
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .llm import DeepSeekClient, FakeLLM, LLMError
from .pipeline.runner import Runner, quiet_log, stderr_log
from .schema import Query, Surface


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="veritas", description=__doc__)
    p.add_argument("--version", action="version", version=f"veritas {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a research mission")
    run.add_argument("query", help="the question / request to research")
    run.add_argument("--surfaces", default="web",
                     help="comma list of web,local,code (default: web)")
    run.add_argument("--local-root", default=None,
                     help="directory to search for local-files surface (default: cwd)")
    run.add_argument("--code-root", default=None,
                     help="repository path for code surface (default: cwd)")
    run.add_argument("--source", action="append", default=[],
                     help="user-provided starting URL or file path (repeatable)")
    run.add_argument("--outdir", default=None, help="output dir (default: ./out)")
    run.add_argument("--no-crosscheck", action="store_true",
                     help="skip the independent cross-check pass (faster/cheaper)")
    run.add_argument("--max-subquestions", type=int, default=5,
                     help="cap planner decomposition")
    run.add_argument("--quiet", action="store_true", help="suppress progress logs")
    run.add_argument("--fake", action="store_true",
                     help=argparse.SUPPRESS)  # FakeLLM offline demo/tests
    return p


def _scripted_defaults() -> dict[str, str]:
    """Canned role responses so `--fake` demos run fully offline."""
    def j(o):
        import json as _json
        return _json.dumps(o)
    return {
        "You are Veritas Planner.": j({"overview": "offline demo plan",
            "subquestions": [{"text": "What does the evidence say about the topic?",
                               "rationale": "demo"}],
            "crosscheck_seed_note": "re-examine from the opposite angle"}),
        "You are Veritas CrossCheck Planner.": j({"overview": "independent",
            "subquestions": [{"text": "Independent look at the same evidence",
                               "rationale": "demo"}]}),
        "You are Veritas Researcher.": j({"key_points": [], "conflicts": [],
                                           "uncertainties": []}),
        "You are Veritas Claim Extractor.": j({"claims": [
            {"statement": "The evidence says the topic is described in the local files.",
             "evidence_idx": [1]}], "noted_gaps": []}),
        "You are Veritas Verifier.": j({"verdict": "supported",
            "reason": "scripted offline demo", "better_statement": ""}),
        "You are Veritas Synthesizer.": "Offline demo run: connect a DeepSeek key "
            "(.env) for real research, verification and cross-checking.",
        "You are Veritas Conflict Detector.": j({"contradicting_pairs": []}),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        return 1

    try:
        surfaces = [Surface(s.strip().lower()) for s in args.surfaces.split(",") if s.strip()]
    except ValueError as e:
        print(f"bad surface list: {e}", file=sys.stderr)
        return 2

    query = Query(text=args.query, surfaces=surfaces, sources=args.source)
    log = quiet_log if args.quiet else stderr_log

    if args.fake:
        llm = FakeLLM(_scripted_defaults())
    else:
        try:
            llm = DeepSeekClient()
        except Exception as e:
            print(f"failed to init LLM client: {e}", file=sys.stderr)
            return 3
        if not llm.api_key:
            print("no DEEPSEEK_API_KEY in .env or environment; "
                  "run with --fake for an offline skeleton or add the key.",
                  file=sys.stderr)
            return 3

    runner = Runner(
        llm=llm, log=log,
        outdir=args.outdir,
        max_subquestions=args.max_subquestions,
        enable_crosscheck=not args.no_crosscheck,
    )
    try:
        report = runner.run(query)
    except LLMError as e:
        print(f"mission aborted: {e}", file=sys.stderr)
        return 4

    counts = report.confidence_counts()
    print(f"done: {len(report.claims)} claims "
          f"(high {counts.get('high',0)}, medium {counts.get('medium',0)}, "
          f"low {counts.get('low',0)}, unsupported {counts.get('unsupported',0)})")
    outdir = runner.outdir
    print(f"report: {outdir / 'report.md'}")
    print(f"ledger: {outdir / 'ledger.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
