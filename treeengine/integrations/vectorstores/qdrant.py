"""Qdrant adapter (``qdrant-client``; local ``:memory:`` / path mode or a server).

    from qdrant_client import QdrantClient
    store = QdrantStore(QdrantClient(url="http://localhost:6333"), "treeengine")

Point ids are UUIDv5 of the record id (Qdrant accepts only UUIDs / integers); the record id,
text and TreeEngine payload live in the point payload. The collection is created on the first
upsert (cosine distance) with a keyword index on ``document_id``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import StoreHit, VectorRecord, batched, point_uuid

try:  # pragma: no cover - import guard
    from qdrant_client import QdrantClient, models
except ImportError as e:  # pragma: no cover
    from ...api.errors import DependencyMissingError

    raise DependencyMissingError("pip install 'treeengine[qdrant]'", extra="qdrant") from e


class QdrantStore:
    name = "qdrant"

    def __init__(self, client: Any = None, collection: str = "treeengine") -> None:
        self.client = client if client is not None else QdrantClient(":memory:")
        self.collection = collection

    def _exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    def _ensure(self, dim: int) -> None:
        if self._exists():
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        import warnings

        try:
            with warnings.catch_warnings():  # local mode: "payload indexes have no effect"
                warnings.simplefilter("ignore")
                self.client.create_payload_index(
                    self.collection, "document_id", models.PayloadSchemaType.KEYWORD
                )
        except Exception:  # noqa: BLE001 - not supported by this deployment
            pass

    @staticmethod
    def _filter(document_id: str | None) -> Any:
        if document_id is None:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ]
        )

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        self._ensure(len(records[0].vector))
        for part in batched(list(records), 256):
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=point_uuid(r.id),
                        vector=[float(x) for x in r.vector],
                        payload={**r.metadata, "_id": r.id, "_text": r.text},
                    )
                    for r in part
                ],
            )

    def delete(self, ids: Sequence[str] | None = None, *, document_id: str | None = None) -> int:
        if not self._exists():
            return 0
        before = self.count()
        if ids:
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=[point_uuid(i) for i in ids]),
            )
        if document_id is not None:
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(filter=self._filter(document_id)),
            )
        return before - self.count()

    def search(
        self, vector: Sequence[float], k: int = 10, *, document_id: str | None = None
    ) -> list[StoreHit]:
        if not self._exists():
            return []
        res = self.client.query_points(
            self.collection,
            query=[float(x) for x in vector],
            limit=k,
            query_filter=self._filter(document_id),
            with_payload=True,
        ).points
        out = []
        for p in res:
            pl = dict(p.payload or {})
            rid, text = str(pl.pop("_id", p.id)), str(pl.pop("_text", ""))
            out.append(StoreHit(rid, float(p.score), text, pl))
        return out

    def vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        if not self._exists() or not ids:
            return {}
        pts = self.client.retrieve(
            self.collection,
            ids=[point_uuid(i) for i in ids],
            with_payload=["_id"],
            with_vectors=True,
        )
        return {str((p.payload or {}).get("_id")): [float(x) for x in p.vector] for p in pts}

    def count(self) -> int:
        if not self._exists():
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def ids(self, document_id: str | None = None) -> list[str]:
        if not self._exists():
            return []
        out: list[str] = []
        offset = None
        while True:
            pts, offset = self.client.scroll(
                self.collection,
                scroll_filter=self._filter(document_id),
                limit=512,
                offset=offset,
                with_payload=["_id"],
                with_vectors=False,
            )
            out += [str((p.payload or {}).get("_id", p.id)) for p in pts]
            if offset is None:
                return sorted(out)
