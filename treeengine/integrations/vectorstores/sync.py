"""A2: keep an external vector store in step with TreeEngine's blocks.

TreeEngine stays the source of truth (parsing, structure, blocks, revisions); the external
store (the one your RAG stack already queries) gets one record per block - its vector, text
and the payload that maps a hit back exactly (``document_id``, ``block_id``, ``node_id``,
``page``, offsets, ``source_revision``, ``uri``, ``section``).

    sync = BlockSync(engine, QdrantStore(client, "kb"), embedder=their_embedder)
    report = sync.sync()          # incremental: only documents whose revision changed
    hits = sync.search("why did margin fall?", k=8)        # -> external hits
    spans = engine.build_context(hits, strict_mapping=True)

Incremental state is kept in the engine database (table ``vector_sync``: target, document,
source revision, index revision, block count). A document whose revision changed is replaced
(``delete(document_id=...)`` then upsert); a document deleted from the engine is deleted from
the store. ``embedder`` defaults to the engine's own; if the engine already has vectors from
the same model they are copied instead of re-embedding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .base import StoreHit, VectorRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vector_sync (
    target TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_revision TEXT,
    index_revision INTEGER,
    blocks INTEGER NOT NULL,
    synced_at TEXT NOT NULL,
    PRIMARY KEY (target, document_id)
);
"""


@dataclass
class SyncReport:
    target: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    records: int = 0
    embedded: int = 0
    copied: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "added": len(self.added),
            "updated": len(self.updated),
            "removed": len(self.removed),
            "unchanged": len(self.unchanged),
            "records": self.records,
            "embedded": self.embedded,
            "copied_from_engine": self.copied,
        }


class BlockSync:
    def __init__(
        self,
        engine: Any,
        store: Any,
        *,
        embedder: Any = None,
        target: str | None = None,
        batch: int = 64,
        with_title: bool = True,
    ) -> None:
        self.engine = engine
        self.repo = engine.repo
        self.store = store
        own = getattr(engine.components.pipeline.indexer, "embedder", None)
        self.embedder: Any = embedder or own
        if self.embedder is None:
            raise ValueError("BlockSync needs an embedder (the engine has none)")
        self.reuse = own is not None and (embedder is None or embedder is own)
        self.target = target or f"{getattr(store, 'name', type(store).__name__)}"
        self.batch = batch
        self.with_title = with_title
        self.conn = self.repo.conn
        self.conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ state
    def state(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT document_id, source_revision, index_revision, blocks FROM vector_sync "
            "WHERE target = ?",
            (self.target,),
        ).fetchall()
        return {
            r[0]: {"source_revision": r[1], "index_revision": r[2], "blocks": r[3]} for r in rows
        }

    def _mark(self, doc: str, rec: dict[str, Any] | None, n: int) -> None:
        with self.repo.transaction():
            self.conn.execute(
                "INSERT OR REPLACE INTO vector_sync VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.target,
                    doc,
                    (rec or {}).get("source_revision"),
                    (rec or {}).get("index_revision"),
                    n,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def _unmark(self, doc: str) -> None:
        with self.repo.transaction():
            self.conn.execute(
                "DELETE FROM vector_sync WHERE target = ? AND document_id = ?", (self.target, doc)
            )

    # ------------------------------------------------------------------ sync
    def sync(self, document_ids: Sequence[str] | None = None, *, force: bool = False) -> SyncReport:
        rep = SyncReport(self.target)
        state = self.state()
        docs = {d.id: d for d in self.repo.list_documents()}
        wanted = list(document_ids) if document_ids is not None else list(docs)
        for doc_id in wanted:
            if doc_id not in docs:
                continue
            rec = (
                self.repo.get_index_status(doc_id)
                if hasattr(self.repo, "get_index_status")
                else None
            )
            old = state.get(doc_id)
            same = (
                old is not None
                and rec is not None
                and old["source_revision"] == rec.get("source_revision")
                and old["index_revision"] == rec.get("index_revision")
            )
            if same and not force:
                rep.unchanged.append(doc_id)
                continue
            if old is not None:
                self.store.delete(document_id=doc_id)
            n = self._push(docs[doc_id], rec, rep)
            self._mark(doc_id, rec, n)
            (rep.updated if old is not None else rep.added).append(doc_id)
        if document_ids is None:
            for doc_id in set(state) - set(docs):
                self.store.delete(document_id=doc_id)
                self._unmark(doc_id)
                rep.removed.append(doc_id)
        return rep

    def remove(self, document_id: str) -> int:
        n = self.store.delete(document_id=document_id)
        self._unmark(document_id)
        return n

    def _push(self, doc: Any, rec: dict[str, Any] | None, rep: SyncReport) -> int:
        from ...retrieval.vector import VectorIndexer, block_embedding_text

        blocks = self.repo.get_document_blocks(doc.id)
        titles: dict[str, str] = {}
        for b in blocks:
            if b.node_id and b.node_id not in titles:
                n = self.repo.get_node(b.node_id)
                titles[b.node_id] = n.title if n else ""
        vi = self.engine.components.vector_index if self.reuse else None
        rev = (rec or {}).get("source_revision")
        for i in range(0, len(blocks), self.batch):
            part = blocks[i : i + self.batch]
            have = (
                vi.vectors([b.id for b in part])
                if vi is not None and hasattr(vi, "vectors")
                else {}
            )
            todo = [b for b in part if b.id not in have]
            if todo:
                texts = [
                    block_embedding_text(
                        b, VectorIndexer._title(b, titles) if self.with_title else None
                    )
                    for b in todo
                ]
                have.update(
                    zip([b.id for b in todo], self.embedder.embed_texts(texts), strict=True)
                )
                rep.embedded += len(todo)
            rep.copied += len(part) - len(todo)
            self.store.upsert(
                [
                    VectorRecord(
                        b.id,
                        have[b.id],
                        b.content,
                        {
                            "document_id": b.document_id,
                            "block_id": b.id,
                            "node_id": b.node_id,
                            "page": b.page,
                            "start_offset": b.start_offset,
                            "end_offset": b.end_offset,
                            "source_revision": rev,
                            "uri": doc.uri,
                            "section": titles.get(b.node_id or ""),
                            "block_type": b.block_type,
                        },
                    )
                    for b in part
                ]
            )
            rep.records += len(part)
        return len(blocks)

    # ------------------------------------------------------------------ query
    def search(
        self, query: str, k: int = 10, *, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Query the external store with the sync's embedder; external hits for build_context."""
        v = self.embedder.embed_query(query)
        return [h.to_hit() for h in self.store.search(v, k, document_id=document_id)]

    def raw_search(
        self, query: str, k: int = 10, *, document_id: str | None = None
    ) -> list[StoreHit]:
        return self.store.search(self.embedder.embed_query(query), k, document_id=document_id)


__all__ = ["BlockSync", "SyncReport"]
