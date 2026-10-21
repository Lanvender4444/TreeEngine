"""A3: an external vector store as TreeEngine's own vector index.

``StoreVectorIndex`` implements TreeEngine's ``VectorIndex`` protocol on top of any adapter
(Qdrant, Chroma, LanceDB, FAISS, ...): ingestion writes block vectors there, hybrid / gated
retrieval, the reranker features, Evidence Search priors and Tree-RAG all read from it.

    from qdrant_client import QdrantClient
    engine = TreeEngine.open("kb.db", embedder=FastEmbedEmbedding(),
                             vector_store=QdrantStore(QdrantClient(url=...), "kb"))

With ``repo`` (the local wiring passes it) each record also carries the block text and the
mapping payload, so the same collection is directly usable by another RAG system - its hits
map back exactly. The embedding-model fingerprint (TE-012) is kept in the engine database
(table ``vector_backend_meta``) - external stores have no shared place for it.

``search(block_ids=...)`` (a tree scope) is exact: the vectors of those blocks are fetched
and compared locally.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .base import VectorRecord

_META = """
CREATE TABLE IF NOT EXISTS vector_backend_meta (
    target TEXT NOT NULL, name TEXT NOT NULL, value TEXT,
    PRIMARY KEY (target, name)
);
"""


class StoreVectorIndex:
    def __init__(self, store: Any, *, repo: Any = None, target: str | None = None) -> None:
        self.store = store
        self.repo = repo
        self.target = target or getattr(store, "name", type(store).__name__)
        self._meta: dict[str, str] = {}
        self.conn = getattr(repo, "conn", None)
        if self.conn is not None:
            self.conn.executescript(_META)

    # ------------------------------------------------------------------ fingerprint meta
    def meta(self, name: str) -> str | None:
        if self.conn is None:
            return self._meta.get(name)
        r = self.conn.execute(
            "SELECT value FROM vector_backend_meta WHERE target = ? AND name = ?",
            (self.target, name),
        ).fetchone()
        return None if r is None else r[0]

    def set_meta(self, name: str, value: str) -> None:
        if self.conn is None:
            self._meta[name] = value
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO vector_backend_meta VALUES (?, ?, ?)",
            (self.target, name, value),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ VectorIndex
    def _record(self, block_id: str, document_id: str, v: Sequence[float]) -> VectorRecord:
        meta: dict[str, Any] = {"document_id": document_id, "block_id": block_id}
        text = ""
        b = self.repo.get_block(block_id) if self.repo is not None else None
        if b is not None:
            text = b.content
            rec = (
                self.repo.get_index_status(document_id)
                if hasattr(self.repo, "get_index_status")
                else None
            )
            meta.update(
                node_id=b.node_id,
                page=b.page,
                start_offset=b.start_offset,
                end_offset=b.end_offset,
                block_type=b.block_type,
                source_revision=(rec or {}).get("source_revision"),
            )
        return VectorRecord(block_id, v, text, meta)

    def upsert(self, items: Sequence[tuple[str, str, Sequence[float]]]) -> None:
        self.store.upsert([self._record(b, d, v) for b, d, v in items])

    def delete(self, block_ids: Sequence[str]) -> None:
        if block_ids:
            self.store.delete(list(block_ids))

    def delete_document(self, document_id: str) -> None:
        self.store.delete(document_id=document_id)

    def rebuild(self) -> None:
        ids = self.store.ids()
        if ids:
            self.store.delete(ids)

    def vectors(self, block_ids: Sequence[str]) -> dict[str, list[float]]:
        return dict(self.store.vectors(list(block_ids)))

    def count(self) -> int:
        return int(self.store.count())

    def size_bytes(self) -> int:
        return 0  # held by the external store

    def search(
        self,
        vector: Sequence[float],
        *,
        k: int = 10,
        document_id: str | None = None,
        block_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        if block_ids is not None:
            got = self.store.vectors(list(block_ids))
            qn = math.sqrt(sum(float(x) * float(x) for x in vector)) or 1.0
            scored = []
            for b, v in got.items():
                vn = math.sqrt(sum(x * x for x in v)) or 1.0
                scored.append((b, sum(a * c for a, c in zip(v, vector, strict=False)) / (qn * vn)))
            return sorted(scored, key=lambda x: -x[1])[:k]
        return [
            (h.id, float(h.score)) for h in self.store.search(vector, k, document_id=document_id)
        ]

    def close(self) -> None:
        pass  # the store (and its client) belongs to the caller


__all__ = ["StoreVectorIndex"]
