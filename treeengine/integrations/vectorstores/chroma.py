"""Chroma adapter (``chromadb``; ephemeral, persistent or HTTP client).

    import chromadb
    store = ChromaStore(chromadb.PersistentClient("./chroma"), "treeengine")

Chroma metadata accepts only scalars: None values are left out and lists / dicts are stored as
``json:`` strings (decoded again on the way out). Cosine space.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import StoreHit, VectorRecord, batched, scalar_meta, unscalar_meta

try:  # pragma: no cover - import guard
    import chromadb
except ImportError as e:  # pragma: no cover
    from ...api.errors import DependencyMissingError

    raise DependencyMissingError("pip install 'treeengine[chroma]'", extra="chroma") from e


class ChromaStore:
    name = "chroma"

    def __init__(self, client: Any = None, collection: str = "treeengine") -> None:
        self.client = client if client is not None else chromadb.EphemeralClient()
        self.col = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for part in batched(list(records), 1000):
            self.col.upsert(
                ids=[r.id for r in part],
                embeddings=[[float(x) for x in r.vector] for r in part],
                documents=[r.text for r in part],
                metadatas=[scalar_meta(r.metadata) or {"_": 1} for r in part],
            )

    def delete(self, ids: Sequence[str] | None = None, *, document_id: str | None = None) -> int:
        before = self.col.count()
        if ids:
            self.col.delete(ids=list(ids))
        if document_id is not None:
            self.col.delete(where={"document_id": document_id})
        return before - self.col.count()

    def search(
        self, vector: Sequence[float], k: int = 10, *, document_id: str | None = None
    ) -> list[StoreHit]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(
            query_embeddings=[[float(x) for x in vector]],
            n_results=min(k, n),
            where={"document_id": document_id} if document_id is not None else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = (res.get("documents") or [[]])[0] or [""] * len(ids)
        metas = (res.get("metadatas") or [[]])[0] or [{}] * len(ids)
        dists = (res.get("distances") or [[]])[0] or [0.0] * len(ids)
        return [
            StoreHit(
                i,
                1.0 - float(d),
                t or "",
                {k2: v for k2, v in unscalar_meta(m).items() if k2 != "_"},
            )
            for i, t, m, d in zip(ids, docs, metas, dists, strict=False)
        ]

    def vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        res = self.col.get(ids=list(ids), include=["embeddings"])
        embs = res.get("embeddings")
        if embs is None:
            return {}
        return {i: [float(x) for x in e] for i, e in zip(res["ids"], embs, strict=False)}

    def count(self) -> int:
        return int(self.col.count())

    def ids(self, document_id: str | None = None) -> list[str]:
        res = self.col.get(
            where={"document_id": document_id} if document_id is not None else None, include=[]
        )
        return sorted(res["ids"])
