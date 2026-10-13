"""Semantic retrieval signal: embedding hits as BlockAnchors."""

from __future__ import annotations

from .anchor import Anchor, anchors_from_evidence
from .result import Trace
from .vector import VectorRetriever


class SemanticAnchorRetriever:
    name = "vector"

    def __init__(self, vector: VectorRetriever) -> None:
        self.vector = vector

    def retrieve_anchors(
        self,
        query: str,
        *,
        document_id: str | None = None,
        limit: int = 50,
        trace: Trace | None = None,
    ) -> list[Anchor]:
        hits = self.vector.vector_search(query, document_id=document_id, limit=limit, trace=trace)
        return anchors_from_evidence(hits, self.name)


__all__ = ["SemanticAnchorRetriever"]
