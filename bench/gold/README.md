# Gold sheets — schema & instructions

Gold sheets are the benchmark's ground truth: for each query, the expected
claims and their labels. **Owner-curated and audited** — the benchmark is only
as honest as its gold. The pipeline never reads these; `run_benchmark.py`
scores each mission ledger against the sheet at `bench/gold/<query_id>.json`
(e.g. `f1-wannacry.json`).

## Schema

```json
{
  "query_id": "<must match bench/queries.json id>",
  "class": "F | C | D | U",
  "source_landscape": "short note on the evidence base (well-documented / contested / thin)",
  "expected_claims": [
    {
      "statement": "precise, independently verifiable claim",
      "gold_label": "correct | incorrect | contested",
      "confidence_class": "high | medium | low | unsupported",
      "note": "why this label (optional)"
    }
  ]
}
```

## Rules

- `gold_label` semantics vs pipeline claims:
  - `correct` — a pipeline claim matching this statement is true (A1/A2 credit).
  - `incorrect` — matching means the pipeline asserted a falsehood (scored as
    not-correct; a finding if verdict was `supported`).
  - `contested` — genuine disagreement; excluded from precision/calibration
    denominators (see spec §8), but `D` queries still must fire the
    contradiction pass (A4).
- `confidence_class` is the *gold expectation* for a correct claim (e.g. a
  single-source-but-solid fact is `medium` until cross-check corroborates) —
  used to sanity-check the pipeline's calibration, not to override it.
- Write statements that a verifier would agree on: self-contained, with the
  key figures/entities spelled out. Token matching is conservative (4+ char
  alphanumeric tokens, Jaccard ≥ 0.5) — near-duplicate statements still match.
- Every query in the executed set needs a sheet **before** its A1–A5 numbers
  are meaningful. The runner prints structure-only metrics without one.

## Worked example

`example-wannacry.json` shows the shape for the canonical F query. Copy to
`gold/f1-wannacry.json` and audit before relying on it — it is illustrative
of the schema, not owner-verified gold.
