"""DeepSeekClient audit-log (VERITAS_LLM_LOG) behavior — hermetic, no network."""

from __future__ import annotations

import io
import json
import stat
import sys

from veritas.config import Settings
from veritas.llm import DeepSeekClient


def _client(log: str) -> DeepSeekClient:
    # __init__ only reads settings/env; it never opens a connection.
    return DeepSeekClient(log=log)


def test_audit_success_writes_entry_and_stays_quiet(tmp_path, capsys):
    log = tmp_path / "audit.log"
    _client(str(log))._audit("sys", "user", "out")
    assert log.read_text() == "=== system ===\nsys\n=== user ===\nuser\n=== out ===\nout\n\n"
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert capsys.readouterr().err == ""


def test_audit_write_failure_is_surfaced_not_silent(tmp_path, capsys):
    # Parent directory does not exist -> open("a") raises OSError.
    log = tmp_path / "missing" / "audit.log"
    # Must not raise: logging must never break a mission.
    _client(str(log))._audit("sys", "user", "out")
    err = capsys.readouterr().err
    assert "VERITAS_LLM_LOG" in err
    assert "audit.log" in err


def test_audit_disabled_writes_nothing(tmp_path, capsys):
    # log=None falls back to settings.llm_log; an empty/absent setting is fine,
    # but force an explicit empty value to keep the test environment-independent.
    _client("")._audit("sys", "user", "out")
    assert not list(tmp_path.iterdir())
    assert capsys.readouterr().err == ""


def test_audit_failure_with_broken_stderr_never_raises(tmp_path, monkeypatch):
    # A closed/unwritable stderr must not turn a surfaced warning into a raise:
    # audit logging can never break a mission (review follow-up).
    class _BrokenStderr:
        def write(self, _text: str) -> int:
            raise OSError("stderr closed")

        def flush(self) -> None:
            raise OSError("stderr closed")

    monkeypatch.setattr(sys, "stderr", _BrokenStderr())
    log = tmp_path / "missing" / "audit.log"
    _client(str(log))._audit("sys", "user", "out")  # must not raise


# ---------------------------------------------- gateway headers / body extra


def test_gateway_headers_and_body_extra_are_merged(monkeypatch):
    """Optional gateway support (opencode-go style): VERITAS_LLM_HEADERS land
    on the HTTP request and VERITAS_LLM_BODY_EXTRA merge into the JSON body,
    overriding the defaults (e.g. a reasoning model needing max_tokens=16384
    or response_format=json_object). Hermetic: urlopen is stubbed."""
    from veritas import llm as llm_mod

    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data.decode())
        payload = {"choices": [{"message": {"content": '{"ok": true}'}}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_mod.settings, "llm_headers",
                        {"User-Agent": "veritas-bench/0.1",
                         "x-opencode-session": "test-session"})
    monkeypatch.setattr(llm_mod.settings, "llm_body_extra",
                        {"response_format": {"type": "json_object"},
                         "max_tokens": 16384})
    client = DeepSeekClient(api_key="k", base_url="https://gw.example/v1",
                            model="glm-5.3-flash", log="")
    assert client.complete_json("sys", "user") == {"ok": True}
    assert captured["url"] == "https://gw.example/v1/chat/completions"
    assert captured["body"]["model"] == "glm-5.3-flash"
    # body extra overrides the request-level default (max_tokens=16384 wins)
    assert captured["body"]["max_tokens"] == 16384
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["headers"]["authorization"] == "Bearer k"
    assert captured["headers"]["user-agent"] == "veritas-bench/0.1"
    assert captured["headers"]["x-opencode-session"] == "test-session"


def test_malformed_optional_json_env_degrades_to_empty(monkeypatch):
    """A misconfigured VERITAS_LLM_HEADERS / VERITAS_LLM_BODY_EXTRA must never
    kill a mission at settings time — parse failures degrade to empty."""
    monkeypatch.setenv("VERITAS_LLM_HEADERS", "{not json")
    monkeypatch.setenv("VERITAS_LLM_BODY_EXTRA", "[1, 2]")  # valid JSON, not object
    s = Settings()
    assert s.llm_headers == {}
    assert s.llm_body_extra == {}
