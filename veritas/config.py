"""Configuration: load .env, expose settings. No secrets ever printed."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (project root). Existing env vars win."""
    here = Path(__file__).resolve().parent.parent
    for candidate in (here / ".env",):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.deepseek_key: str = os.environ.get("DEEPSEEK_API_KEY", "")
        self.deepseek_base_url: str = os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.deepseek_model: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.tavily_key: str = os.environ.get("TAVILY_API_KEY", "")
        self.llm_log: str = os.environ.get("VERITAS_LLM_LOG", "")
        self.web_timeout_s: float = float(os.environ.get("VERITAS_WEB_TIMEOUT", "12"))
        # Optional extra HTTP headers for the chat endpoint (JSON object),
        # e.g. gateways that require a client User-Agent or a session header.
        # Parse failures leave it empty — a misconfigured optional header
        # must not kill the mission at import time.
        try:
            self.llm_headers: dict = {
                str(k): str(v) for k, v in
                json.loads(os.environ.get("VERITAS_LLM_HEADERS", "{}")).items()
            }
        except (ValueError, AttributeError):
            self.llm_headers = {}
        # Optional extra request-body fields (JSON object) merged into every
        # chat completion, e.g. {"reasoning_effort": "low"} or {"max_tokens":
        # 16384} for reasoning models whose default effort can consume the
        # whole token budget. NOTE: response_format belongs in
        # VERITAS_LLM_JSON_BODY_EXTRA (below), not here — body extras leak
        # into free-form prose calls, and JSON mode must not constrain them
        # (Codex P1, PR #16 round 2).
        # Parse failures leave it empty — never fatal at import time.
        try:
            self.llm_body_extra: dict = dict(
                json.loads(os.environ.get("VERITAS_LLM_BODY_EXTRA", "{}")))
        except (ValueError, TypeError):
            self.llm_body_extra = {}
        # Optional JSON-only extra body fields (JSON object), merged only into
        # structured (complete_json) requests — e.g.
        # {"response_format": {"type": "json_object"}} for gateways that need
        # explicit JSON mode. Never applied to free-form prose calls: a
        # provider enforcing JSON mode would reject the prose prompt or wrap
        # its answer in a JSON object (Codex P1, PR #16 round 2).
        try:
            self.llm_json_body_extra: dict = dict(
                json.loads(os.environ.get("VERITAS_LLM_JSON_BODY_EXTRA", "{}")))
        except (ValueError, TypeError):
            self.llm_json_body_extra = {}

    def has_reasoning_backend(self) -> bool:
        return bool(self.deepseek_key)


settings = Settings()
