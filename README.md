# veritas

**Accurate and reliable agent research team** — a deterministic multi-role
research pipeline over the **public web**, **local files/notes**, and
**codebases**. Every factual claim is bound to retrievable evidence, verified
against re-fetched source text, and independently cross-checked; uncertainty is
stated, never hidden.

```text
plan ──▶ research ──▶ claims ──▶ verify ──▶ cross-check ──▶ synthesize
 │         │            │           │            │              │
 decompose parallel    atomic,    re-fetch     independent   prose from
 query     gather      evidence-   source,     2nd plan +    verified
 (LLM)     (search +   bound      LLM judge    reconcile     claims only
           fetch)      claims     verdict      (deterministic)
```

## Install & configure

```bash
git init . && cp .env.example .env   # then add DEEPSEEK_API_KEY (DeepSeek API)
pip install -e .                     # or: run via  python -m veritas.cli
```

No paid search key is required — web research uses keyless engines
(DuckDuckGo, Wikipedia, arXiv, Hacker News, GitHub). Set `TAVILY_API_KEY`
(optional) to prefer a cleaner paid engine.

## Usage

```bash
veritas run "Why did the WannaCry worm spread so fast?" --surfaces web

veritas run "Summarise my notes on the agent-ecosystem plan" \
    --surfaces local --local-root ~/notes

veritas run "How is auth handled in this repo?" \
    --surfaces code --code-root ~/agent-ecosystem

veritas run "Compare X and Y" --surfaces web,local,code \
    --outdir out/mission1 --no-crosscheck --quiet
```

Every run writes two artifacts to `--outdir` (default `./out`):

* **`report.md`** — human report: answer prose, per-claim confidence table,
  verified claims with numbered sources, *Not established* gaps, *Conflicts in
  the evidence*, and the cross-check summary.
* **`ledger.json`** — machine-readable evidence ledger: every claim with its
  verdict, confidence, and full evidence objects (source + quoted passage).

Confidence semantics:

| tag | meaning |
|-----|---------|
| `high` | claim supported by its sources **and** independently corroborated by the cross-check pass from different sources |
| `medium` | supported by re-fetched source text (single run / single source) |
| `low` | only partially supported (verifier supplies the corrected statement) |
| `unsupported` | **not asserted** — evidence could neither support nor refute it; reported under *Not established* |

## Reliability design (the short version)

1. **No claim without evidence.** The claim extractor must cite evidence by
   index; un-cited or out-of-range citations are rejected into a gaps list —
   the model can never inject support from thin air.
2. **Verification re-reads the source.** The verifier re-fetches each cited
   source (web pages re-downloaded; file/code sources re-read at their exact
   `#L#-#` anchor) and judges the claim against that text: supported / partial
   / contradicted / unsupported, with a corrected statement when partial.
3. **Independent cross-check.** A second, differently-planned research pass
   runs; its claims are reconciled deterministically. Agreement from
   *different* sources bumps confidence to `high`; unmatched findings are put
   through the verifier (never trusted unchecked) before being appended.
4. **Semantic conflict detection.** One dedicated pass pairs claims that
   cannot both be true — antonyms defeat token matching, so this is the one
   place an LLM judges *relationships between claims*, with index validation.
5. **Honest failure.** Claims the sources can't establish are reported as
   *Not established*, and contradictions are shown rather than smoothed over.

## Development

```bash
python3 -m pytest -q        # 52 hermetic tests, no network, FakeLLM-driven
```

`veritas run "..." --fake` runs the full pipeline offline with scripted model
responses, useful for a no-key demo of the report shape. Set
`VERITAS_LLM_LOG=/path/log.txt` to audit every prompt/response the team sends.

See `docs/DESIGN.md` for the full architecture, role contracts, and the
threats each stage defends against.
