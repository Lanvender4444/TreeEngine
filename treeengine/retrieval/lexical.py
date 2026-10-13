"""Lexical retrieval signal: FTS hits as BlockAnchors."""

from __future__ import annotations

from .anchor import Anchor, anchors_from_evidence
from .fts import FTSRetriever
from .result import Trace


class LexicalRetriever:
    name = "fts"

    def __init__(self, fts: FTSRetriever) -> None:
        self.fts = fts

    def retrieve_anchors(
        self,
        query: str,
        *,
        document_id: str | None = None,
        limit: int = 50,
        trace: Trace | None = None,
    ) -> list[Anchor]:
        hits = self.fts.fts_search(query, document_id=document_id, limit=limit, trace=trace)
        return anchors_from_evidence(hits, self.name)


__all__ = ["LexicalRetriever"]
