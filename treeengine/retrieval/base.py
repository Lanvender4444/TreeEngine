"""Retriever helpers. The Retriever protocol itself lives in ``core.protocols``."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.models import Evidence
from ..core.protocols import Retriever


@runtime_checkable
class SemanticRetriever(Retriever, Protocol):
    """Reserved: added only when the benchmark shows semantic misses are a bottleneck."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def dedupe(evidence: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str | None, str | None]] = set()
    out = []
    for e in evidence:
        key = (e.block_id, e.node_id if e.block_id is None else None)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


@runtime_checkable
class AnchorRetriever(Protocol):
    """V0.9 retrieval contract: a retriever returns ranked document positions (``Anchor``):
    BlockAnchors (FTS, Vector, Graph-lite) or NodeAnchors (Tree)."""

    name: str

    def retrieve_anchors(
        self,
        query: str,
        *,
        document_id: str | None = None,
        limit: int = 50,
        trace: Any = None,
    ) -> list[Any]: ...
