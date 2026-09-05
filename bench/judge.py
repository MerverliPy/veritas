"""LLM gold judge for the R1 benchmark.

Statement-level lexical gold matching cannot credit the pipeline's
generative claims (a fresh run phrases correct facts differently from the
gold sheet every time — the pilot's claims averaged 0.17-0.22 Jaccard to
gold, scoring a good report at ~0% precision). The fix is a strict LLM
judge: each pipeline claim is judged against the gold facts only, at
temperature 0, reusing the pipeline's own client.

The judge is injected into ``bench/score.py`` as a ``claim_judge`` callable,
so the scorer stays pure and hermetic (tests script a FakeLLM judge). The
pipeline never sees the gold sheet; the gold sheet never affects a run.

Label semantics (mapped to metrics in score.py):
  correct    -> gold-verdict 'correct'   (credit in A1/A2)
  incorrect  -> 'incorrect'              (asserted a falsehood; no credit)
  contested  -> 'contested'              (disputed priority; no credit, and
                                          excluded from calibration)
  off-topic  -> 'unmatched'              (true or background but does not
                                          address any gold fact: a supported
                                          off-topic claim is exactly the U-class
                                          dodge the pilot exposed; no credit)
Anything else, or a judge failure, falls back to the lexical matcher so a
judge outage never silently flips a score.
"""

from __future__ import annotations

import json

from veritas.llm import BaseLLM, LLMError

GOLD_JUDGE_SYSTEM = """You are a strict gold-standard judge for a research benchmark. A research pipeline produced a CLAIM for the QUERY. Judge whether the claim, as stated, is CORRECT given ONLY the GOLD FACTS below. Gold facts are a SAMPLE of checkable facts, not an exhaustive list of everything true.

Labels:
- correct: the claim states a fact consistent with a gold 'correct' fact — paraphrase allowed. If the claim adds a conjunct or detail beyond the gold wording, credit it ONLY when that added assertion is also true and consistent (same entities/figures/scope). Example: gold 'researcher registered the kill-switch domain' is matched by 'a 22-year-old UK researcher discovered the kill switch accidentally'.
- incorrect: the claim contradicts a gold 'correct' fact (different figure/date/country/polarity for the same thing), matches a gold fact labeled 'incorrect', OR adds a conjunct that is demonstrably false (e.g. gold covers 'Sputnik launched in 1957' and the claim continues '...and the Moon is made of cheese' — the fused claim is not correct). A standalone atomic claim with NO gold coverage is also incorrect when it is demonstrably false by general knowledge (e.g. 'the Moon is made of cheese', 'the sky is green') — fabricated assertions must never be excluded as off-topic.
- contested: the claim asserts a one-sided priority/disputed statement that a gold fact labels 'contested', or asserts a figure within a range a gold 'contested' fact describes as disputed. No credit either way.
- off-topic: NO gold fact covers the claim's assertion and it is not demonstrably false — it may be true, about a source, an adjacent topic, or a comparison. No credit and no penalty — off-topic claims are neither correct nor incorrect for this benchmark.

Never call a claim incorrect for asserting something gold simply does not mention — that is off-topic, not incorrect. Ignore phrasing and the claim's own confidence.
Respond with JSON only: {"label": "correct|incorrect|contested|off-topic", "reason": "<one sentence>"}"""


def _label(raw: str) -> str | None:
    """Parse the judge's JSON; tolerate stray prose around the object."""
    text = raw.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    label = (obj.get("label") or "").strip().lower()
    return label if label in ("correct", "incorrect", "contested", "off-topic") \
        else None


def _judge_prompt(query: str | None, expected: list[dict], statement: str) -> str:
    lines = []
    if query:
        lines.append(f"QUERY: {query}")
    lines.append("GOLD FACTS:")
    for e in expected:
        label = e.get("gold_label", "?")
        note = e.get("note", "")
        lines.append(f"- [{label}] {e['statement']}"
                     + (f"  ({note})" if note else ""))
    lines.append("")
    lines.append("CLAIM TO JUDGE:")
    lines.append(statement)
    lines.append("")
    lines.append("Verdict JSON:")
    return "\n".join(lines)


def make_gold_judge(llm: BaseLLM) -> callable:
    """Build a claim judge: judge(statement, expected, query=None) -> label
    ('correct' | 'incorrect' | 'contested' | 'off-topic'). Falls back to the
    caller's lexical verdict via the return value convention below."""
    from bench.score import gold_verdict as _lexical  # local: avoid cycle

    def judge(statement: str, expected: list[dict], query: str | None = None) -> str:
        user = _judge_prompt(query, expected, statement)
        try:
            data = llm.complete_json(GOLD_JUDGE_SYSTEM, user)
        except (LLMError, OSError, ValueError) as e:  # noqa: BLE001
            raise JudgeError(f"gold judge failed: {e}") from e
        raw = (data.get("label") if isinstance(data, dict) else "") or ""
        label = raw.strip().lower()
        if label in ("correct", "incorrect", "contested", "off-topic"):
            return label
        # Refusal / schema drift (valid JSON, missing or unknown label): raise
        # so the driver's make_claim_judge counts this as a fallback instead of
        # silently swapping in the lexical verdict without a record.
        raise JudgeError(f"gold judge returned invalid label: {raw!r}")

    return judge


FALLBACK_UNMATCHED = "__fallback_unmatched__"


class JudgeError(RuntimeError):
    pass


def make_claim_judge(llm: BaseLLM) -> tuple[callable, dict]:
    """Driver-facing judge with a counted lexical fallback.

    Returns (judge_cb, state) where judge_cb(statement, expected, query)
    -> canonical label. Transport/parse failures fall back to the lexical
    matcher per claim and increment state['fallbacks'] — a judge outage
    never silently flips or drops a score."""
    from bench.score import gold_verdict as _lexical  # local: avoid cycle

    judge = make_gold_judge(llm)
    state = {"fallbacks": 0}

    def judge_cb(statement: str, expected: list[dict],
                 query: str | None = None) -> str:
        try:
            return judge(statement, expected, query)
        except Exception:  # noqa: BLE001 - outage must fall back, not fail
            state["fallbacks"] += 1
            lexical = _lexical(statement, expected)
            # Provenance matters: a judge 'off-topic' is excluded from
            # precision, but an outage fallback that cannot place the claim
            # must still count as not-correct (conservative) — never as
            # judge output.
            if lexical in ("correct", "incorrect", "contested"):
                return lexical
            return FALLBACK_UNMATCHED

    return judge_cb, state
