"""Traditional (structure-blind) RAG baselines.

    document text -> fixed-size token chunks (600 tokens, 100 overlap) -> BM25 / embeddings

The chunker sees only the raw extracted text - no headings, no tree, no blocks - exactly what a
standard RAG pipeline gets from a parser. BM25 uses the same SQLite FTS5 tokenizer and CJK
segmentation as TreeEngine's block index and the vector side uses the same embedding model, so
the only difference between ``rag_*`` and TreeEngine strategies is the unit (chunk vs block)
and the structure.

Chunks are mapped back to blocks only for *evaluation* (pages a chunk covers, the section its
largest part comes from) - the retrieval itself never uses them.
"""

from __future__ import annotations

import bisect
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from treeengine.core.models import Evidence
from treeengine.core.protocols import EmbeddingProvider, Repository
from treeengine.core.text import query_terms
from treeengine.storage.fts5 import fts_match_expr, segment_for_index

from ..metrics.cost import token_spans


@dataclass
class Chunk:
    id: str
    document_id: str
    start: int
    end: int
    text: str
    pages: list[int]
    node_id: str | None


def chunk_spans(text: str, size: int = 600, overlap: int = 100) -> list[tuple[int, int]]:
    """Character spans of windows of ``size`` tokens advancing by ``size - overlap`` tokens."""
    if size <= overlap:
        raise ValueError("chunk size must exceed overlap")
    spans = token_spans(text)
    out = []
    step = size - overlap
    for i in range(0, max(1, len(spans)), step):
        window = spans[i : i + size]
        if not window:
            break
        out.append((window[0][0], window[-1][1]))
        if i + size >= len(spans):
            break
    return out


class _BlockMap:
    """Character offset -> (pages, dominant section) using the document's blocks."""

    def __init__(self, blocks: Sequence[Any]) -> None:
        rows = sorted(
            (b.start_offset, b.end_offset, b.page, b.node_id)
            for b in blocks
            if b.start_offset is not None and b.end_offset is not None
        )
        self.rows = rows
        self.starts = [r[0] for r in rows]

    def locate(self, start: int, end: int) -> tuple[list[int], str | None]:
        i = max(0, bisect.bisect_right(self.starts, start) - 1)
        pages: set[int] = set()
        best: tuple[int, str | None] = (0, None)
        while i < len(self.rows) and self.rows[i][0] < end:
            s, e, page, node = self.rows[i]
            ov = min(e, end) - max(s, start)
            if ov > 0:
                if page is not None:
                    pages.add(int(page))
                if ov > best[0]:
                    best = (ov, node)
            i += 1
        return sorted(pages), best[1]


class ChunkStore:
    def __init__(
        self, repo: Repository, document_ids: Sequence[str], size: int = 600, overlap: int = 100
    ) -> None:
        self.size, self.overlap = size, overlap
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        for d in document_ids:
            doc = repo.get_document(d)
            if doc is None or not doc.text:
                continue
            bmap = _BlockMap(repo.get_document_blocks(d))
            for i, (s, e) in enumerate(chunk_spans(doc.text, size, overlap)):
                text = doc.text[s:e].strip()
                if not text:
                    continue
                pages, node = bmap.locate(s, e)
                c = Chunk(f"{d}:c{i}", d, s, e, text, pages, node)
                self.chunks.append(c)
                self.by_id[c.id] = c
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, "
            "document_id UNINDEXED, content, tokenize='porter unicode61')"
        )
        self.conn.executemany(
            "INSERT INTO chunks_fts VALUES (?, ?, ?)",
            [(c.id, c.document_id, segment_for_index(c.text)) for c in self.chunks],
        )
        self.vectors: Any = None
        self.embedder: EmbeddingProvider | None = None

    def size_bytes(self) -> int:
        return sum(len(c.text.encode("utf-8")) for c in self.chunks)

    # ------------------------------------------------------------------ search
    def bm25(self, query: str, document_id: str | None, limit: int) -> list[tuple[Chunk, float]]:
        match = fts_match_expr(query_terms(query))
        if not match:
            return []
        sql = "SELECT chunk_id, bm25(chunks_fts) AS r FROM chunks_fts WHERE chunks_fts MATCH ?"
        params: list[Any] = [match]
        if document_id is not None:
            sql += " AND document_id = ?"
            params.append(document_id)
        try:
            rows = self.conn.execute(sql + " ORDER BY r LIMIT ?", [*params, limit]).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(self.by_id[cid], -float(r)) for cid, r in rows]

    def index_vectors(self, index: Any, embedder: EmbeddingProvider, batch: int = 64) -> None:
        self.vectors, self.embedder = index, embedder
        for i in range(0, len(self.chunks), batch):
            part = self.chunks[i : i + batch]
            vecs = embedder.embed_texts([c.text for c in part])
            index.upsert([(c.id, c.document_id, v) for c, v in zip(part, vecs, strict=True)])

    def vector(self, query: str, document_id: str | None, limit: int) -> list[tuple[Chunk, float]]:
        if self.vectors is None or self.embedder is None:
            raise RuntimeError("chunk vectors not built")
        qv = self.embedder.embed_query(query)
        hits = self.vectors.search(qv, k=limit, document_id=document_id)
        return [(self.by_id[cid], s) for cid, s in hits if cid in self.by_id]

    @staticmethod
    def evidence(hits: Sequence[tuple[Chunk, float]], source: str) -> list[Evidence]:
        return [
            Evidence(
                document_id=c.document_id,
                node_id=c.node_id,
                block_id=c.id,
                content=c.text,
                source=source,
                score=round(s, 4),
                metadata={
                    "chunk": True,
                    "pages": c.pages,
                    "page": c.pages[0] if c.pages else None,
                    "start_offset": c.start,
                    "end_offset": c.end,
                },
            )
            for c, s in hits
        ]


__all__ = ["Chunk", "ChunkStore", "chunk_spans"]
