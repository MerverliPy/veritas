# R1 benchmark (`bench/`)

Owner-run harness for the R1 live-web benchmark. **Never run in CI and never
required by the hermetic test suite** — missions need the public web, the
deepseek-chat backend (`.env`), and spend real money.

- `queries.json` — fixed query set, **seed starting points — owner curates**.
- `gold/` — per-query gold sheets (schema + instructions in `gold/README.md`).
  `example-wannacry.json` illustrates the shape (not owner-verified).
- `score.py` — pure, unit-tested scoring (metrics + A1–A6 gates). Hermetic.
- `run_benchmark.py` — orchestrator: runs each query through the real CLI,
  meters cost from `VERITAS_LLM_LOG`, enforces a per-run USD cap, writes
  `out/bench/<run-id>/scorecard.json`.
- `collect_relevance.py` — A5 rubric-sample collector (owner-run, no LLM):
  extracts ~2 sub-questions × top sources per query from a run's ledgers
  into `relevance-sample.json` + a null-filled `relevance-judgements.json`
  you mark 0/1 (keep order). Filled files are preserved across reruns
  (--force resets). Latest run is picked by scorecard mtime, not name.
- `run_benchmark.py --rescore <run-dir>` — score an EXISTING run (no new
  missions): re-applies gold + judge + `--relevance` judgements and writes
  `scorecard-rescore.json`. Relevance judgements MUST be scored against the
  run whose sources produced them, so A5 always goes through --rescore. Every invocation gets its own
  `<run-id>/` subdir (default `cc|nocc-<timestamp>`, or pass `--run-id`) so
  the paired/determinism arms never clobber each other's scorecard, ledgers,
  or `llm.log` (and each arm's cost meters only its own traffic).

## Run a pilot

```bash
# 1. curate: pick queries (bench/queries.json) and fill gold/<id>.json
# 2. execute (mainline arm — cross-check on):
python3 bench/run_benchmark.py --ids f1-wannacry,f2-sputnik --cap-usd 0.25
# 3. read the scorecard (path printed; under out/bench/<run-id>/):
cat out/bench/<run-id>/scorecard.json
```

Outputs land under `out/bench/` which is gitignored (`out/`). Cost is an
*estimate* (log chars × blended rate — constants in `score.py`); the driver
stops after the query that crosses `--cap-usd` and marks the scorecard
`capped-partial`.

## Arms (spec §4)

| Arm | Command | Feeds |
|---|---|---|
| Mainline (cross-check on) | `run_benchmark.py` (default) | A1–A3, A5, A6, A4 (contradiction fires) |
| Paired (cross-check off) | `run_benchmark.py --no-crosscheck --run-id nocc-pilot` on the same subset | A4 cross-check delta (compare the two scorecards) |
| Determinism | run a query twice with distinct `--run-id`s | A6 flip rate (pass both ledgers via `score.flip_rate`) |

## Boundaries

- Scores come only from `ledger.json` + gold — the driver never inspects
  `report.md` prose, so "report says X" is not scored as evidence.
- Benchmark numbers are **not** a quality claim until R1's acceptance
  criteria pass and the owner publishes them (positioning guardrail).
- Tavily/Brave/Serper parity is out of scope until validated (provider parity
  note in the audit); mainline uses keyless engines only.
