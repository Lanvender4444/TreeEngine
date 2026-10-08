"""Retriever protocol. Every retriever (tree, fts, and later vector/web/graph) returns Evidence."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.models import Evidence


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int = 10,
    ) -> list[Evidence]: ...


@runtime_checkable
class SemanticRetriever(Retriever, Protocol):
    """Reserved for V0.2 (vector retrieval). Not implemented in V0.1."""

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
