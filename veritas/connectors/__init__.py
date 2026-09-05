"""Connector registry: build providers for the surfaces a mission enables."""

from __future__ import annotations

from pathlib import Path

from ..schema import Surface
from .base import Provider
from .files import CodeProvider, LocalProvider
from .web import WebProvider


def build_providers(
    surfaces: list[Surface],
    *,
    local_root: Path | str | None = None,
    code_root: Path | str | None = None,
) -> list[Provider]:
    """Instantiate one provider per enabled surface.

    ``local_root`` / ``code_root`` default to the current directory. Duplicate
    roots are allowed (the surfaces differ semantically: LOCAL = notes/docs,
    CODE = tracked source + structural overview).
    """
    providers: list[Provider] = []
    for s in surfaces:
        if s is Surface.WEB:
            providers.append(WebProvider())
        elif s is Surface.LOCAL:
            providers.append(LocalProvider(local_root or Path.cwd()))
        elif s is Surface.CODE:
            providers.append(CodeProvider(code_root or Path.cwd()))
    return providers
