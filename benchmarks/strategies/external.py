"""Part A benchmark: "another RAG stack" with and without TreeEngine underneath.

The other stack is a typical LangChain-style pipeline built from what such a stack would use
itself, independent of TreeEngine's parsing:

    parser    pypdf ``extract_text()`` per page (plain files: their raw text)
    splitter  LangChain ``RecursiveCharacterTextSplitter`` (1000 chars, 200 overlap)
    store     Qdrant (local mode) through ``treeengine.integrations.vectorstores.QdrantStore``
    embedder  the run's embedder (same model as TreeEngine's, so only the unit differs)
    retrieval vector top-k, or BM25 + vector RRF (``hybrid``), optionally a cross-encoder
              rerank of the top 30 (``_rr``)

Strategies:

    ext_vector, ext_hybrid, ext_hybrid_rr      their chunks, read as they are (raw)
    ext_vector+te, ext_hybrid+te, ext_hybrid_rr+te
                                               the same chunks, aligned with TreeEngine blocks
                                               (``integrations.align``) and handed to TreeEngine
                                               as block anchors: every TreeEngine reading
                                               policy (auto, gate+retry ...) applies

The chunk ids carry the source path only (no TreeEngine ids) - alignment finds the document
by path and the text. Indexes are built per document on first use (only the documents the
questions ask about). Chunk pages come from the other parser itself (for page recall).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from treeengine.core.models import Evidence
from treeengine.core.text import query_terms
from treeengine.storage.fts5 import fts_match_expr, segment_for_index

SIZE, OVERLAP = 1000, 200
RERANK = "ce:Xenova/ms-marco-MiniLM-L-12-v2"


@dataclass
class ExtChunk:
    id: str
    uri: str
    document_id: str  # TreeEngine's id - evaluation bookkeeping only, never used to map
    text: str
    pages: list[int]
    start: int


def parse(uri: str) -> list[tuple[int | None, str]]:
    """(page, text) pairs from the other stack's own parser."""
    if uri.lower().endswith(".pdf"):
        from pypdf import PdfReader

        out = []
        for i, p in enumerate(PdfReader(uri).pages, 1):
            try:
                out.append((i, p.extract_text() or ""))
            except Exception:  # noqa: BLE001 - a broken page
                out.append((i, ""))
        return out
    with open(uri, encoding="utf-8", errors="replace") as f:
        return [(None, f.read())]


def cached_parse(uri: str, cache_dir: Any) -> list[tuple[int | None, str]]:
    """``parse`` once per file version (pypdf is the slow part: ~0.2 s a page)."""
    if cache_dir is None:
        return parse(uri)
    import hashlib
    import json
    import os
    from pathlib import Path

    st = os.stat(uri)
    key = hashlib.sha256(f"{uri}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20]
    f = Path(cache_dir) / "ext-pages" / f"{key}.json"
    if f.exists():
        return [(p, t) for p, t in json.loads(f.read_text(encoding="utf-8"))]
    pages = parse(uri)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(pages), encoding="utf-8")
    return pages


class ExternalRAG:
    def __init__(self, ws: Any) -> None:
        from treeengine.integrations.vectorstores.qdrant import QdrantStore

        self.ws = ws
        self.repo = ws.repo
        self.store = QdrantStore(collection="ext")
        self.chunks: dict[str, ExtChunk] = {}
        self.by_doc: dict[str, list[str]] = {}
        self.fts = sqlite3.connect(":memory:", check_same_thread=False)
        self.fts.execute(
            "CREATE VIRTUAL TABLE c USING fts5(chunk_id UNINDEXED, uri UNINDEXED, content, "
            "tokenize='porter unicode61')"
        )
        self.aligner: Any = None
        self.stats = {"documents": 0, "chunks": 0, "seconds": 0.0, "aligned": 0, "exact": 0}

    # ------------------------------------------------------------------ indexing
    def ensure(self, document_id: str) -> str | None:
        if document_id in self.by_doc:
            return self.uri(document_id)
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from treeengine.integrations.vectorstores import VectorRecord

        t0 = time.perf_counter()
        doc = self.repo.get_document(document_id)
        if doc is None or not doc.uri:
            self.by_doc[document_id] = []
            return None
        pages = cached_parse(doc.uri, getattr(self.ws, "cache_dir", None))
        text, starts = "", []
        for p, t in pages:
            starts.append((len(text), p))
            text += t + "\n"
        split = RecursiveCharacterTextSplitter(
            chunk_size=SIZE, chunk_overlap=OVERLAP, add_start_index=True
        )
        made: list[ExtChunk] = []
        for i, d in enumerate(split.create_documents([text])):
            s = int(d.metadata["start_index"])
            e = s + len(d.page_content)
            bounds = [(st, starts[j + 1][0] if j + 1 < len(starts) else len(text), p)
                      for j, (st, p) in enumerate(starts)]  # fmt: skip
            pg = [p for st, en, p in bounds if p is not None and st < e and en > s]
            made.append(ExtChunk(f"ext:{doc.uri}#{i}", doc.uri, document_id, d.page_content, pg, s))
        emb = self.ws.embedder
        for i in range(0, len(made), 64):
            part = made[i : i + 64]
            vecs = emb.embed_texts([c.text for c in part])
            self.store.upsert(
                [
                    VectorRecord(
                        c.id,
                        v,
                        c.text,
                        {
                            "document_id": c.uri,
                            "chunk_id": c.id,
                            "page": c.pages[0] if c.pages else None,
                        },
                    )
                    for c, v in zip(part, vecs, strict=True)
                ]  # fmt: skip
            )
        self.fts.executemany(
            "INSERT INTO c VALUES (?, ?, ?)",
            [(c.id, c.uri, segment_for_index(c.text)) for c in made],
        )
        for c in made:
            self.chunks[c.id] = c
        self.by_doc[document_id] = [c.id for c in made]
        # TreeEngine side: align the other stack's chunks with the blocks (stored once)
        rep = self._aligner().align(
            [{"text": c.text, "chunk_id": c.id, "uri": c.uri} for c in made]
        )
        s = rep.summary()
        self.stats["aligned"] += rep.aligned
        self.stats["exact"] += rep.exact
        self.stats["documents"] += 1
        self.stats["chunks"] += len(made)
        self.stats["seconds"] += time.perf_counter() - t0
        self.ws.log(
            f"ext: {doc.uri} {len(made)} chunks, aligned {s['aligned_share']}, "
            f"exact {s['exact_share']}, {time.perf_counter() - t0:.0f}s"
        )
        return doc.uri

    def _aligner(self) -> Any:
        if self.aligner is None:
            from treeengine.integrations.align import ChunkAligner

            self.aligner = ChunkAligner(SimpleNamespace(repo=self.repo))
        return self.aligner

    def uri(self, document_id: str) -> str | None:
        ids = self.by_doc.get(document_id) or []
        return self.chunks[ids[0]].uri if ids else None

    # ------------------------------------------------------------------ search
    def vector(self, q: str, d: str, k: int) -> list[tuple[ExtChunk, float]]:
        uri = self.ensure(d)
        if uri is None:
            return []
        hits = self.store.search(self.ws.embedder.embed_query(q), k, document_id=uri)
        return [(self.chunks[h.id], h.score) for h in hits if h.id in self.chunks]

    def bm25(self, q: str, d: str, k: int) -> list[tuple[ExtChunk, float]]:
        uri = self.ensure(d)
        match = fts_match_expr(query_terms(q))
        if uri is None or not match:
            return []
        try:
            rows = self.fts.execute(
                "SELECT chunk_id, bm25(c) r FROM c WHERE c MATCH ? AND uri = ? ORDER BY r LIMIT ?",
                (match, uri, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(self.chunks[c], -float(r)) for c, r in rows]

    def hybrid(self, q: str, d: str, k: int) -> list[tuple[ExtChunk, float]]:
        a, b = self.bm25(q, d, k), self.vector(q, d, k)
        score: dict[str, float] = {}
        for lst in (a, b):
            for r, (c, _) in enumerate(lst):
                score[c.id] = score.get(c.id, 0.0) + 1.0 / (60 + r + 1)
        order = sorted(score, key=lambda x: -score[x])[:k]
        return [(self.chunks[c], score[c]) for c in order]

    def rerank(self, q: str, hits: list[tuple[ExtChunk, float]], top: int = 30):
        from treeengine.retrieval.rerank import rerank_scores

        head, tail = hits[:top], hits[top:]
        sc = rerank_scores(RERANK, q, [c.text for c, _ in head])
        head = sorted(zip(head, sc, strict=True), key=lambda x: -x[1])
        return [(c, s) for (c, _), s in head] + tail

    # ------------------------------------------------------------------ evidence
    @staticmethod
    def raw(hits: Sequence[tuple[ExtChunk, float]], source: str) -> list[Evidence]:
        return [
            Evidence(c.document_id, None, c.id, c.text, source, round(s, 4),
                     {"chunk": True, "pages": c.pages, "page": c.pages[0] if c.pages else None})
            for c, s in hits
        ]  # fmt: skip

    def mapped(self, hits: Sequence[tuple[ExtChunk, float]], source: str) -> list[Evidence]:
        """Their chunks -> TreeEngine blocks via the stored alignment (rank order, deduped);
        an unaligned chunk passes through as text."""
        from treeengine.integrations.hits import resolve_chunk

        out: list[Evidence] = []
        seen: set[str] = set()
        for r, (c, s) in enumerate(hits, 1):
            anchors = resolve_chunk(self.repo, {"chunk_id": c.id, "text": c.text}, rank=r)
            if not anchors:
                out += self.raw([(c, s)], source)
                continue
            for a in anchors:
                if a.block_id in seen or a.evidence is None:
                    continue
                seen.add(a.block_id)
                ev = a.evidence
                out.append(
                    Evidence(ev.document_id, ev.node_id, ev.block_id, ev.content, source,
                             round(s, 4), {**(ev.metadata or {}), "chunk_id": c.id})
                )  # fmt: skip
        return out


def shared(ws: Any) -> ExternalRAG:
    """The run's one ExternalRAG (index built per document on first use)."""
    if "ext" not in ws._resources:
        ws._resources["ext"] = ExternalRAG(ws)
    return ws._resources["ext"]  # type: ignore[no-any-return]


def ext_table(ws: Any) -> dict[str, tuple[Any, tuple[str, ...]]]:
    def ext() -> ExternalRAG:
        return shared(ws)

    def make(kind: str, te: bool, rr: bool) -> Any:
        name = f"ext_{kind}{'_rr' if rr else ''}{'+te' if te else ''}"

        def run(q: str, d: str | None, k: int, t: Any) -> list[Evidence]:
            if d is None:
                return []
            x = ext()
            hits = x.vector(q, d, max(k, 30)) if kind == "vector" else x.hybrid(q, d, max(k, 30))
            if rr:
                hits = x.rerank(q, hits)
            hits = hits[:k]
            return x.mapped(hits, name) if te else x.raw(hits, name)

        return name, run

    out: dict[str, tuple[Any, tuple[str, ...]]] = {}
    for kind in ("vector", "hybrid"):
        for rr in (False, True):
            for te in (False, True):
                n, fn = make(kind, te, rr)
                out[n] = (fn, ())
    return out


EXT = tuple(
    f"ext_{k}{r}{t}" for k in ("vector", "hybrid") for r in ("", "_rr") for t in ("", "+te")
)

__all__ = ["EXT", "ExternalRAG", "ext_table", "parse"]
