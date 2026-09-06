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
  into `relevance-sample.json` + a keyed, null-filled
  `relevance-judgements.json` ({sample_sha, judgements}) you mark 0/1 (keep
  order inside judgements). The sample_sha binds labels to this exact run —
  rescore rejects mismatched or un-keyed files. Filled files are preserved
  across reruns (--force resets); malformed files abort. Latest run is
  picked by scorecard mtime, not name.
- `run_benchmark.py --rescore <run-dir>` — score an EXISTING run (no new
  missions): re-applies gold + judge + `--relevance` judgements and writes
  `scorecard-rescore.json`. Requires the run's scorecard.json; rejects query
  ids that drifted (text/class) or are missing from the spec; judge spend
  respects --cap-usd; nocc runs stay advisory; relevance judgements MUST be
  sha-bound to this run's sample (keyed file from collect_relevance.py). Every invocation gets its own
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
| Mainline (cross-check on) | `run_benchmark.py` (default) | A1–A3, A5; A4 + A6 with the inputs below |
| Paired (cross-check off) | `run_benchmark.py --no-crosscheck --run-id nocc-pilot` on the same subset | A4 same-query delta (see below) |
| Determinism | run a query >=3 times with distinct `--run-id`s (e.g. `det-1/2/3`) | A6 distribution gate (see below) |

## Evaluating A4 and A6 on existing runs (re-spec)

A4 and A6 compare inputs from OTHER runs, so a single mainline scorecard
marks them `n/a` until you supply those inputs. Both are scored WITHOUT new
missions via `--rescore`:

```bash
# A4: rescore the mainline run with the cross-check-off arm alongside it.
# Gates on the SAME query ids in both arms (>= 2 pairs, incl. a D query);
# reports placed-claim populations so a confounded pair is visible.
python3 bench/run_benchmark.py --rescore out/bench/full-1 \
    --paired-arm out/bench/full-1-nocc

# A6: >=3 rerun dirs of the same queries feed the distribution gate
# (median pairwise normalized L1 of confidence counts <= 0.30).
python3 bench/run_benchmark.py --rescore out/bench/full-1 \
    --rerun-dirs out/bench/det-1,out/bench/det-2,out/bench/det-3
```

Each rerun/paired dir holds per-query `out/bench/<run>/<query-id>/ledger.json`
subdirs (the normal layout). Statement-level `flip_rate` is informational
under the re-spec; the A6 gate is distribution-level.

## Boundaries

- Scores come only from `ledger.json` + gold — the driver never inspects
  `report.md` prose, so "report says X" is not scored as evidence.
- Benchmark numbers are **not** a quality claim until R1's acceptance
  criteria pass and the owner publishes them (positioning guardrail).
- Tavily/Brave/Serper parity is out of scope until validated (provider parity
  note in the audit); mainline uses keyless engines only.
