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
      "note": "why this label (optional)",
      "named_winners": ["Guglielmo Marconi", "Karl Ferdinand Braun"] (optional)
    }
  ]
}

## Optional: `named_winners` (award/attribution facts)

For facts where a person's identity is truth-critical — who shared/won a
prize, who a record belongs to — add `"named_winners": ["Full Name", ...]`
to the expected claim. The lexical matcher then requires a claim's award
co-recipients (from an "X and Y won/shared/received" structure) to be a
subset of these names: a claim crediting a wrong co-recipient ("Marconi and
Popov won ...") can never match, however high the token overlap. The LLM
judge ignores the field (it reads the statement) but benefits from the
precise wording.`
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
  key figures/entities spelled out. Matching is conservative (4+ char
  alphanumeric tokens, Jaccard ≥ 0.5) and refuses matches that disagree on
  **years** or **polarity** ("did not launch" vs "launched") — a statement
  is only `correct` when it genuinely says the same thing as gold.
- **Atomicity**: state one fact per `expected_claim`. The pipeline emits
  atomic verifier-shaped claims; a fused sentence ("comets are ice and dust
  while asteroids are rocky") matches any single atomic entry poorly.
- **Granularity**: phrase gold at the query's requested level. Do not put
  unrequested specifics in a gold statement ("...in 1957" vs gold "...on
  4 October 1957") — a correct answer at the asked granularity would be
  blocked by the year/polarity checks and denied credit.
- **Completeness**: enumerate the claims a *correct* report would assert
  (the canonical run's supported claims are a good checklist). Gold
  incompleteness penalizes legitimate claims as unmatched.
- **Positive phrasing**: state facts positively. A gold entry containing a
  negation word ("was not vulnerable") flips the polarity check, so an
  equivalent positive claim ("were immune") would be rejected — phrase
  immunity/protection facts as "protected from ..." instead.
- **Variants**: when a natural alternate wording of a *central* fact would
  fall under the Jaccard bar, add it as a second entry with
  `"variant_of": "<exact base statement>"`. A variant credits the same
  fact when matched but never adds to the recall denominator.
- **Ambiguity order**: when two gold entries could tie for a claim, the
  scorer resolves ties toward the less credit-worthy label
  (contested/incorrect over correct); word entries so one-sided priority
  claims cannot token-match a neutral fact entry (e.g. a patent statement
  names the patent office and "granted a patent").
- Every query in the executed set needs a sheet **before** its A1–A5 numbers
  are meaningful. The runner prints structure-only metrics without one.
- `U` (hard/niche) sheets may legitimately contain a *verified anchor* claim
  plus a landscape note saying the target figure is not publicly documented —
  never invent the answer. The point of `U` is honest failure.

## Current sheets

All seven query sheets are **AUDITED (2026-09-06)**: labels and anchor
facts were independently re-verified against the public record (each sheet's
`note` cites the sources; the u1 anchor correction, the u2 added
54-locomotive anchor, the f1 contested-relabel rationale, and the d2-radio
1943-Supreme-Court anchors are recorded in the files). `d2-radio` was added
as the second class-D query (A4 re-spec: contradiction pass fires on >= half
the paired D queries) and mirrors the d1-telephone sheet shape. Sheets are
owner-curated scoring data the pipeline never reads; any future label change
should update the per-sheet `note` the same way. `f1-wannacry` supersedes
the earlier illustrative example.
