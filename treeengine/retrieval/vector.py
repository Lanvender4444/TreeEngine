"""Semantic retrieval over block embeddings.

Unit of embedding is the Block (nothing else in V0.3). The embedded text is the block content
prefixed with its section title, so a block knows which section it belongs to.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ..core.config import EngineConfig
from ..core.models import Block, Evidence
from ..core.protocols import EmbeddingProvider, Repository, VectorIndex
from .result import Trace


def block_embedding_text(block: Block, node_title: str | None, max_chars: int = 2000) -> str:
    head = f"{node_title}\n" if node_title else ""
    return (head + block.content)[:max_chars]


class VectorIndexer:
    """Keeps a VectorIndex in step with the Repository (blocks stay the source of truth)."""

    def __init__(
        self, repo: Repository, index: VectorIndex, embedder: EmbeddingProvider, batch: int = 64
    ) -> None:
        self.repo = repo
        self.index = index
        self.embedder = embedder
        self.batch = batch

    def index_document(self, document_id: str) -> int:
        self.index.delete_document(document_id)
        blocks = self.repo.get_document_blocks(document_id)
        titles: dict[str, str] = {}
        for b in blocks:
            if b.node_id and b.node_id not in titles:
                n = self.repo.get_node(b.node_id)
                titles[b.node_id] = n.title if n else ""
        for i in range(0, len(blocks), self.batch):
            chunk = blocks[i : i + self.batch]
            texts = [block_embedding_text(b, titles.get(b.node_id or "")) for b in chunk]
            vecs = self.embedder.embed_texts(texts)
            self.index.upsert([(b.id, b.document_id, v) for b, v in zip(chunk, vecs, strict=True)])
        return len(blocks)

    def delete_document(self, document_id: str) -> None:
        self.index.delete_document(document_id)

    def reindex_all(self) -> int:
        self.index.rebuild()
        return sum(self.index_document(d.id) for d in self.repo.list_documents())


class VectorRetriever:
    name = "vector"

    def __init__(
        self,
        repo: Repository,
        index: VectorIndex,
        embedder: EmbeddingProvider,
        config: EngineConfig | None = None,
    ) -> None:
        self.repo = repo
        self.index = index
        self.embedder = embedder
        self.config = config or EngineConfig()

    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int | None = None,
    ) -> list[Evidence]:
        scope = self.repo.get_subtree_ids(node_id) if node_id else None
        return self.vector_search(query, document_id=document_id, node_ids=scope, limit=limit)

    def vector_search(
        self,
        query: str,
        document_id: str | None = None,
        node_ids: Sequence[str] | None = None,
        limit: int | None = None,
        trace: Trace | None = None,
    ) -> list[Evidence]:
        limit = limit or self.config.fts_limit
        t0 = time.perf_counter()
        qv = self.embedder.embed_query(query)
        t1 = time.perf_counter()
        block_ids: list[str] | None = None
        if node_ids is not None:
            block_ids = [b.id for nid in node_ids for b in self.repo.get_blocks(nid, limit=None)]
        hits = self.index.search(qv, k=limit, document_id=document_id, block_ids=block_ids)
        t2 = time.perf_counter()
        if trace is not None:
            trace.add(
                "vector",
                model=self.embedder.model_name,
                document_id=document_id,
                scope_blocks=None if block_ids is None else len(block_ids),
                hits=len(hits),
                embed_ms=round((t1 - t0) * 1000, 2),
                search_ms=round((t2 - t1) * 1000, 2),
            )
        titles: dict[str, str] = {}
        out: list[Evidence] = []
        for block_id, sim in hits:
            b = self.repo.get_block(block_id)
            if b is None:  # index out of date: blocks are the source of truth
                continue
            title = None
            if b.node_id:
                if b.node_id not in titles:
                    n = self.repo.get_node(b.node_id)
                    titles[b.node_id] = n.title if n else ""
                title = titles[b.node_id]
            out.append(
                Evidence(
                    document_id=b.document_id,
                    node_id=b.node_id,
                    block_id=b.id,
                    content=b.content,
                    source="vector",
                    score=round(sim, 4),
                    metadata={
                        "node_title": title,
                        "page": b.page,
                        "start_offset": b.start_offset,
                        "end_offset": b.end_offset,
                        "block_type": b.block_type,
                        "similarity": round(sim, 4),
                    },
                )
            )
        return out
