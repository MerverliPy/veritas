"""URL validation is hermetic and rejects non-public fetch targets."""

from __future__ import annotations

import socket

import pytest

import veritas.extract as extract


def _fake_getaddrinfo(host: str, port, *, type=None):
    del port, type
    if host in {"127.0.0.1", "169.254.169.254", "10.0.0.5"}:
        address = host
    else:
        address = "93.184.216.34"
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/file",
    "http://127.0.0.1/",
    "http://169.254.169.254/",
    "http://10.0.0.5/",
])
def test_guard_rejects_non_public_urls(monkeypatch, url):
    monkeypatch.setattr(extract.socket, "getaddrinfo", _fake_getaddrinfo)
    with pytest.raises(ValueError):
        extract._guard_url(url)


def test_guard_accepts_public_https_without_network(monkeypatch):
    monkeypatch.setattr(extract.socket, "getaddrinfo", _fake_getaddrinfo)
    extract._guard_url("https://example.com/")


def test_fetch_url_rejects_before_urlopen(monkeypatch):
    monkeypatch.setattr(extract.socket, "getaddrinfo", _fake_getaddrinfo)

    def unexpected_urlopen(*args, **kwargs):
        raise AssertionError("urlopen must not run for a private target")

    monkeypatch.setattr(extract.urllib.request, "urlopen", unexpected_urlopen)
    with pytest.raises(ValueError):
        extract.fetch_url("http://127.0.0.1/")


def test_fetch_url_rechecks_redirect_target(monkeypatch):
    monkeypatch.setattr(extract.socket, "getaddrinfo", _fake_getaddrinfo)

    class RedirectResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "http://127.0.0.1/"

    monkeypatch.setattr(extract.urllib.request, "urlopen", lambda *args, **kwargs: RedirectResponse())
    with pytest.raises(ValueError):
        extract.fetch_url("https://example.com/")
