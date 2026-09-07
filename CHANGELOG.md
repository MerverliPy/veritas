# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-06

Initial release.

### Added

- Evidence-bound research pipeline: deterministic multi-role orchestration
  (plan → research → claim extraction → verify → cross-check → contradiction
  detection → synthesize) over three surfaces — public web, local files/notes,
  and codebases.
- Research surfaces: keyless web engines (DuckDuckGo, Wikipedia, arXiv,
  Hacker News/Algolia, GitHub repo search) plus direct URL fetch, with
  optional Tavily when `TAVILY_API_KEY` is set; local file/notes search;
  git-aware code search.
- Per-claim evidence binding by index, verification against re-fetched source
  text, and deterministic confidence mapping (`high` only via independent
  cross-check corroboration from a different source).
- Independent cross-check pass (on by default; skip with `--no-crosscheck`)
  that re-plans and re-researches, promoting confidence only on agreement from
  different sources.
- Semantic contradiction detection over final assertable claims, with index
  validation.
- Deterministic report assembly: `report.md` (answer prose, per-claim
  confidence, numbered sources, *Not established*, research gaps, conflicts,
  cross-check summary) and machine-readable `ledger.json` per run.
- Offline demo mode via `veritas run ... --fake` and prompt auditing via
  `VERITAS_LLM_LOG`.
- Hermetic test suite (FakeLLM-driven, no network) under `tests/`.

### Fixed

- Shared global citation numbering between the synthesized answer and the
  report's numbered sources.
- Audit remediations: file-root containment for the local surface, strict
  1-based evidence indices, corrected `PARTIAL` statement handling, and
  deterministic verification order.
- URL fetch guard hardening: https-only fetches, public-host resolution
  checks, redirect re-validation, arXiv over HTTPS, and audit logs written
  with `0600` permissions; removed dead configuration knobs.
- Patent-infringement matcher: many hardening rounds over claim status,
  chronology, award, and coordination semantics (d2-radio rounds 3–27).
