"""Corpus retrieval: which documents should we look in?

Coarse step of coarse-to-fine (Corpus -> Document -> Node -> Block). In-document retrievers
(tree / fts / vector) never decide this themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.protocols import Repository
from ..core.text import query_terms
from .result import Trace
from .vector import VectorRetriever


@dataclass
class DocumentCandidate:
    document_id: str
    score: float
    signals: dict[str, Any] = field(default_factory=dict)


class CorpusRetriever:
    """Document routing. FTS routing by default; add semantic routing with a VectorRetriever.

    Each signal ranks documents by the sum of their best 3 hits (not by volume: long documents
    would otherwise win on sheer number of weak matches); signals are fused with RRF."""

    def __init__(
        self,
        repo: Repository,
        vector: VectorRetriever | None = None,
        max_docs: int = 3,
        rrf_k: int = 60,
    ) -> None:
        self.repo = repo
        self.vector = vector
        self.max_docs = max_docs
        self.rrf_k = rrf_k

    def route(
        self, query: str, max_docs: int | None = None, trace: Trace | None = None
    ) -> list[DocumentCandidate]:
        max_docs = max_docs or self.max_docs
        docs = self.repo.list_documents()
        if len(docs) <= max_docs:
            chosen = [DocumentCandidate(d.id, 0.0, {"reason": "small corpus"}) for d in docs]
            if trace is not None:
                trace.add(
                    "document_routing", candidates=len(docs), chosen=[c.document_id for c in chosen]
                )
            return chosen

        rankings: dict[str, list[str]] = {}
        per_signal: dict[str, dict[str, float]] = {}
        terms = query_terms(query)
        if terms:
            per_doc: dict[str, list[float]] = {}
            for b, s in self.repo.fts_query(terms, mode="or", limit=50):
                per_doc.setdefault(b.document_id, []).append(s)
            per_signal["fts"] = {d: sum(sorted(v, reverse=True)[:3]) for d, v in per_doc.items()}
        if self.vector is not None:
            per_doc_v: dict[str, list[float]] = {}
            for e in self.vector.vector_search(query, limit=50):
                per_doc_v.setdefault(e.document_id, []).append(e.score or 0.0)
            per_signal["vector"] = {
                d: sum(sorted(v, reverse=True)[:3]) for d, v in per_doc_v.items()
            }
        for name, scores in per_signal.items():
            rankings[name] = sorted(scores, key=lambda d: -scores[d])

        fused: dict[str, float] = {}
        signals: dict[str, dict[str, Any]] = {}
        for name, ranked in rankings.items():
            for rank, d in enumerate(ranked):
                fused[d] = fused.get(d, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                signals.setdefault(d, {})[name] = {
                    "rank": rank + 1,
                    "score": round(per_signal[name][d], 4),
                }
        ranked_docs = sorted(fused, key=lambda d: -fused[d])[:max_docs]
        chosen = [DocumentCandidate(d, round(fused[d], 6), signals[d]) for d in ranked_docs]
        if not chosen:  # no lexical/semantic signal at all: fall back to the first documents
            chosen = [
                DocumentCandidate(d.id, 0.0, {"reason": "no signal"}) for d in docs[:max_docs]
            ]
        if trace is not None:
            trace.add(
                "document_routing",
                candidates=len(docs),
                signals=sorted(per_signal),
                chosen=[c.document_id for c in chosen],
                detail=[{"document_id": c.document_id, **c.signals} for c in chosen],
            )
        return chosen
