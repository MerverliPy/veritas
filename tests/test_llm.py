"""DeepSeekClient audit-log (VERITAS_LLM_LOG) behavior — hermetic, no network."""

from __future__ import annotations

from veritas.llm import DeepSeekClient


def _client(log: str) -> DeepSeekClient:
    # __init__ only reads settings/env; it never opens a connection.
    return DeepSeekClient(log=log)


def test_audit_success_writes_entry_and_stays_quiet(tmp_path, capsys):
    log = tmp_path / "audit.log"
    _client(str(log))._audit("sys", "user", "out")
    assert log.read_text() == "=== system ===\nsys\n=== user ===\nuser\n=== out ===\nout\n\n"
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
