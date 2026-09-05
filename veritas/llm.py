"""LLM access.

Two implementations share one interface:

* :class:`DeepSeekClient` — OpenAI-compatible chat endpoint
  (https://api.deepseek.com/chat/completions), with lenient JSON extraction
  and a single retry on transient failures.
* :class:`FakeLLM` — scripted responses for deterministic tests. Responses
  are selected by an exact ``system``-prefix match, so pipeline stages can be
  exercised without a network or a key.

Every call may be appended to ``VERITAS_LLM_LOG`` when configured, giving an
audit trail of exactly what each role was asked and what it returned.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .config import settings


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """Lenient JSON-object extraction from model output."""
    t = text.strip()
    # strip markdown fences if present
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # fall back to the outermost balanced {...} region
    start = t.find("{")
    if start == -1:
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    break
    raise LLMError(f"could not parse JSON object from: {text[:200]!r}")


class BaseLLM:
    """Interface used by every pipeline role.

    ``complete`` returns free text; ``complete_json`` returns a parsed dict.
    Roles may pass ``json=True``-style hints; clients decide how to enforce.
    """

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        raise NotImplementedError

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        raw = self.complete(system, user, temperature=temperature, max_tokens=max_tokens)
        return extract_json(raw)


class DeepSeekClient(BaseLLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        log: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.deepseek_key
        self.base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self.model = model or settings.deepseek_model
        self.timeout = timeout if timeout is not None else settings.web_timeout_s + 40
        self.log = settings.llm_log if log is None else log

    # -- plumbing ----------------------------------------------------------
    def _chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        if not self.api_key:
            raise LLMError(
                "no DEEPSEEK_API_KEY configured — set it in .env or the environment"
            )
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err: Exception | None = None
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                last_err = LLMError(f"HTTP {e.code} from {self.model}: {detail}")
                if e.code in (429, 500, 502, 503, 504) and attempt == 1:
                    time.sleep(2.0)
                    continue
                raise last_err
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = LLMError(f"network error talking to {self.model}: {e}")
                if attempt == 1:
                    time.sleep(1.0)
                    continue
                raise last_err
        raise LLMError(str(last_err))

    def _audit(self, system: str, user: str, out: str) -> None:
        if not self.log:
            return
        try:
            with open(self.log, "a") as f:
                f.write("=== system ===\n%s\n=== user ===\n%s\n=== out ===\n%s\n\n"
                        % (system, user, out))
        except OSError:
            pass  # logging must never break a mission

    # -- interface ----------------------------------------------------------
    def complete(self, system, user, *, temperature=0.2, max_tokens=2048) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        out = self._chat(messages, temperature, max_tokens)
        self._audit(system, user, out)
        return out


class FakeLLM(BaseLLM):
    """Deterministic scripted LLM for tests and offline demo runs.

    Each entry maps a system-prompt *prefix* to either a literal string or a
    callable(user_text) -> string. The first matching prefix wins. A special
    key ``"*"`` is the fallback. This keeps test fixtures readable while
    letting pipeline behaviour (not model quality) be the thing under test.
    """

    def __init__(self, responses: dict[str, str | Callable[[str], str]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []  # (system_prefix, user) audit
        self._lock = threading.Lock()

    def complete(self, system, user, *, temperature=0.2, max_tokens=2048) -> str:
        with self._lock:
            for prefix, value in self.responses.items():
                if prefix != "*" and not system.startswith(prefix):
                    continue
                self.calls.append((prefix, user))
                if callable(value):
                    return value(user)
                return value
        raise LLMError(f"FakeLLM has no response for system prompt: {system[:80]!r}")

    def complete_json(self, system, user, *, temperature=0.0, max_tokens=2048) -> dict:
        return extract_json(self.complete(system, user, temperature=temperature))


def default_client() -> BaseLLM:
    """Create the configured client, or FakeLLM when no key is present."""
    if settings.has_reasoning_backend():
        return DeepSeekClient()
    return FakeLLM({})
