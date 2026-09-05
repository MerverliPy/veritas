"""Configuration: load .env, expose settings. No secrets ever printed."""

from __future__ import annotations

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
        self.brave_key: str = os.environ.get("BRAVE_API_KEY", "")
        self.serper_key: str = os.environ.get("SERPER_API_KEY", "")
        self.llm_log: str = os.environ.get("VERITAS_LLM_LOG", "")
        # Quality/time knobs
        self.max_evidence_per_claim: int = int(os.environ.get("VERITAS_MAX_EVIDENCE", "6"))
        self.web_timeout_s: float = float(os.environ.get("VERITAS_WEB_TIMEOUT", "12"))

    def has_reasoning_backend(self) -> bool:
        return bool(self.deepseek_key)


settings = Settings()
