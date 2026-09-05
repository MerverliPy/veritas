"""Provider interface: how a research surface answers searches.

A *provider* owns one surface (web / local / code). The researcher calls
``search`` per sub-question and gets evidence with quoted passages; the
verifier calls ``fetch`` to pull the full text behind a source again and
re-check claims against the *same* retrievable origin.

Connectors must be resilient: failures return empty lists and append a human
readable warning to ``warnings`` instead of raising into the pipeline.
"""

from __future__ import annotations

from ..schema import Evidence, Source, Surface


class Provider:
    surface: Surface = Surface.WEB

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def search(self, query: str, limit: int = 8) -> list[Evidence]:
        raise NotImplementedError

    def fetch(self, source: Source) -> str | None:
        """Return the full retrievable text behind a source, or None."""
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.surface.value} provider"
