"""Chunk alignment: another RAG system's chunks -> TreeEngine blocks.

An existing RAG pipeline cut its documents with its own parser and splitter; its chunks carry
no TreeEngine block ids. ``ChunkAligner`` locates every chunk inside the same document as
TreeEngine ingested it and records which blocks it covers, once. After that a hit that carries
the chunk id (``metadata["chunk_id"]`` or ``chunk_id``) maps exactly - in ``build_context``,
``map_hits``, the LangChain / LlamaIndex / Haystack post-processors - and strict mapping can be
used with chunks that were never TreeEngine's.

Locating a chunk, in order (``method`` in the record):

    offsets   the chunk's start / end offsets agree with TreeEngine's text
    exact     the chunk occurs in the document text (compared without whitespace and case,
              Unicode-normalised: different PDF extractors break lines and space words
              differently, but produce the same characters)
    probes    several 32-character probes of the chunk are found in order (a chunk whose
              parser dropped or changed a few characters); quality = share of probes found
    fts       the best full-text match of the chunk's words (approximate)

``quality`` is 1.0 for offsets / exact, the probe share for probes, 0.3 for fts. The record
also keeps the document's ``source_revision`` at alignment time: a later re-ingest with changed
text makes the alignment ``stale`` (re-align, or the strict mode drops the chunk).
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk_alignment (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    block_ids TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    method TEXT NOT NULL,
    quality REAL NOT NULL,
    source_revision TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunk_alignment_doc ON chunk_alignment(document_id);
"""

PROBE = 32
QUALITY_FTS = 0.3


@dataclass
class Chunk:
    """A chunk of another system. ``chunk_id`` defaults to a hash of the text."""

    text: str
    chunk_id: str | None = None
    document_id: str | None = None
    uri: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    @property
    def key(self) -> str:
        return self.chunk_id or chunk_key(self.text)


@dataclass
class Alignment:
    chunk_id: str
    document_id: str | None
    block_ids: list[str]
    start_offset: int | None
    end_offset: int | None
    method: str  # offsets | exact | probes | fts | none
    quality: float
    source_revision: str | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.block_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignReport:
    alignments: list[Alignment]

    @property
    def aligned(self) -> int:
        return sum(1 for a in self.alignments if a.ok)

    @property
    def exact(self) -> int:
        return sum(1 for a in self.alignments if a.method in ("offsets", "exact"))

    def summary(self) -> dict[str, Any]:
        n = len(self.alignments) or 1
        by: dict[str, int] = {}
        for a in self.alignments:
            by[a.method] = by.get(a.method, 0) + 1
        return {
            "chunks": len(self.alignments),
            "aligned": self.aligned,
            "exact": self.exact,
            "aligned_share": round(self.aligned / n, 3),
            "exact_share": round(self.exact / n, 3),
            "mean_quality": round(sum(a.quality for a in self.alignments) / n, 3),
            "by_method": by,
        }


def chunk_key(text: str) -> str:
    return "chk_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _squash(text: str) -> tuple[str, list[int]]:
    """Text without whitespace, NFKC + casefolded, and the original index of every kept char."""
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        for c in unicodedata.normalize("NFKC", ch).casefold():
            if not c.isspace():
                out.append(c)
                idx.append(i)
    return "".join(out), idx


def as_chunk(item: Any) -> Chunk:
    """A Chunk, a dict (``text`` / ``page_content`` / ``content``, ids at the top or under
    ``metadata``), or a LangChain / LlamaIndex / Haystack document object."""
    if isinstance(item, Chunk):
        return item
    if hasattr(item, "node") and hasattr(item, "score"):  # LlamaIndex NodeWithScore
        ch = as_chunk(item.node)
        ch.score = None if item.score is None else float(item.score)
        return ch
    if isinstance(item, Mapping):
        meta = dict(item.get("metadata") or item.get("meta") or {})
        get = lambda k: item.get(k, meta.get(k))  # noqa: E731
        text = get("text") or get("page_content") or get("content") or ""
        return Chunk(
            text=str(text),
            chunk_id=_s(get("chunk_id") or get("id") or get("id_")),
            document_id=_s(get("document_id") or get("doc_id")),
            uri=_s(get("uri") or get("source") or get("file_path")),
            start_offset=_i(get("start_offset") or get("start_char_idx")),
            end_offset=_i(get("end_offset") or get("end_char_idx")),
            metadata=meta,
            score=_f(get("score")),
        )
    text = (
        getattr(item, "page_content", None)
        or getattr(item, "content", None)
        or (item.get_content() if hasattr(item, "get_content") else None)
        or getattr(item, "text", None)
        or ""
    )
    meta = dict(getattr(item, "metadata", None) or getattr(item, "meta", None) or {})
    data: dict[str, Any] = {"text": text, "metadata": meta}
    for attr in ("id", "id_", "node_id"):
        v = getattr(item, attr, None)
        if isinstance(v, str) and v and "chunk_id" not in meta:
            data["id"] = v
            break
    if getattr(item, "score", None) is not None:
        data["score"] = item.score
    for attr in ("start_char_idx", "end_char_idx"):
        v = getattr(item, attr, None)
        if v is not None:
            data[attr] = v
    return as_chunk(data)


def _s(v: Any) -> str | None:
    return None if v is None or v == "" else str(v)


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


class ChunkAligner:
    """Align external chunks to an engine's blocks and keep the result in its database."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.repo = engine.repo
        conn = getattr(self.repo, "conn", None)
        if conn is None:
            raise TypeError("chunk alignment is stored in the SQLite repository")
        self.conn = conn
        conn.executescript(_SCHEMA)
        self._docs: dict[str, tuple[str, list[int], Any]] = {}
        self._spans: dict[str, list[tuple[int, int, str]]] = {}

    # ------------------------------------------------------------------ public
    def align(
        self,
        chunks: Iterable[Any],
        *,
        document_id: str | None = None,
        store: bool = True,
        min_quality: float = 0.0,
    ) -> AlignReport:
        out = []
        for item in chunks:
            ch = as_chunk(item)
            a = self.align_one(ch, document_id=document_id)
            if a.quality < min_quality:
                why = f"quality {a.quality} < {min_quality}"
                a = Alignment(ch.key, a.document_id, [], None, None, "none", 0.0,
                              a.source_revision, why)  # fmt: skip
            out.append(a)
        if store:
            self._save([a for a in out if a.ok])
        return AlignReport(out)

    def lookup(self, chunk_id: str) -> Alignment | None:
        r = self.conn.execute(
            "SELECT chunk_id, document_id, block_ids, start_offset, end_offset, method, quality, "
            "source_revision FROM chunk_alignment WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if r is None:
            return None
        return Alignment(r[0], r[1], json.loads(r[2]), r[3], r[4], r[5], float(r[6]), r[7])

    def forget(self, *, document_id: str | None = None, chunk_ids: Sequence[str] = ()) -> int:
        with self.repo.transaction():
            n = 0
            if document_id is not None:
                n += self.conn.execute(
                    "DELETE FROM chunk_alignment WHERE document_id = ?", (document_id,)
                ).rowcount
            for c in chunk_ids:
                n += self.conn.execute(
                    "DELETE FROM chunk_alignment WHERE chunk_id = ?", (c,)
                ).rowcount
        return n

    # ------------------------------------------------------------------ one chunk
    def align_one(self, ch: Chunk, *, document_id: str | None = None) -> Alignment:
        doc_id = self._document(ch, document_id)
        if doc_id is None:
            return self._fts(ch) or Alignment(
                ch.key, None, [], None, None, "none", 0.0, None, "document not found"
            )
        squashed, idx, d = self._doc(doc_id)
        rev = self._revision(doc_id)
        text = d.text or ""
        if ch.start_offset is not None and ch.end_offset is not None:
            s, e = int(ch.start_offset), int(ch.end_offset)
            if 0 <= s < e <= len(text) and _squash(text[s:e])[0] == _squash(ch.text)[0]:
                return self._record(ch, doc_id, s, e, "offsets", 1.0, rev)
        target, _ = _squash(ch.text)
        if not target:
            return Alignment(ch.key, doc_id, [], None, None, "none", 0.0, rev, "empty chunk")
        pos = squashed.find(target)
        if pos >= 0:
            if squashed.find(target, pos + 1) >= 0 and len(target) < 200:
                reason = "the chunk text occurs more than once; the first occurrence was used"
            else:
                reason = None
            s, e = idx[pos], idx[pos + len(target) - 1] + 1
            a = self._record(ch, doc_id, s, e, "exact", 1.0, rev)
            a.reason = reason
            return a
        probes = self._probes(target)
        found: list[tuple[int, int]] = []
        cursor = 0
        for p in probes:
            j = squashed.find(p, cursor)
            if j < 0:
                continue
            found.append((j, j + len(p)))
            cursor = j + len(p)
        if len(found) >= 2 or (found and len(probes) == 1):
            q = round(len(found) / len(probes), 3)
            s, e = idx[found[0][0]], idx[found[-1][1] - 1] + 1
            if e - s <= 3 * max(1, len(ch.text)):  # probes must stay together
                return self._record(ch, doc_id, s, e, "probes", q, rev)
        return self._fts(ch, doc_id) or Alignment(
            ch.key, doc_id, [], None, None, "none", 0.0, rev, "text not found in the document"
        )

    # ------------------------------------------------------------------ helpers
    def _document(self, ch: Chunk, default: str | None) -> str | None:
        if ch.document_id and self.repo.get_document(ch.document_id) is not None:
            return ch.document_id
        if ch.uri:
            d = self.repo.find_document_by_uri(ch.uri)
            if d is None:  # a path the other system stored differently: match the file name
                name = ch.uri.replace("\\", "/").rsplit("/", 1)[-1]
                for doc in self.repo.list_documents():
                    if (doc.uri or "").replace("\\", "/").rsplit("/", 1)[-1] == name:
                        d = doc
                        break
            if d is not None:
                return str(d.id)
        if default:
            return default
        docs = self.repo.list_documents()
        return str(docs[0].id) if len(docs) == 1 else None

    def _doc(self, doc_id: str) -> tuple[str, list[int], Any]:
        if doc_id not in self._docs:
            d = self.repo.get_document(doc_id)
            sq, idx = _squash(d.text or "")
            self._docs[doc_id] = (sq, idx, d)
        return self._docs[doc_id]

    def _revision(self, doc_id: str) -> str | None:
        get = getattr(self.repo, "get_index_status", None)
        rec = get(doc_id) if get is not None else None
        return rec.get("source_revision") if rec else None

    @staticmethod
    def _probes(target: str) -> list[str]:
        if len(target) <= PROBE:
            return [target]
        n = min(8, max(2, len(target) // PROBE))
        step = (len(target) - PROBE) / (n - 1)
        return [target[int(i * step) : int(i * step) + PROBE] for i in range(n)]

    def _record(
        self, ch: Chunk, doc_id: str, s: int, e: int, method: str, q: float, rev: str | None
    ) -> Alignment:
        import bisect

        if doc_id not in self._spans:  # (start, end, id) sorted by start, loaded once
            self._spans[doc_id] = sorted(
                (b.start_offset, b.end_offset, b.id)
                for b in self.repo.get_document_blocks(doc_id)
                if b.start_offset is not None and b.end_offset is not None
            )
        rows = self._spans[doc_id]
        i = max(0, bisect.bisect_right([r[0] for r in rows], s) - 1) if rows else 0
        blocks = []
        while i < len(rows) and rows[i][0] < e:
            if min(e, rows[i][1]) > max(s, rows[i][0]):
                blocks.append(SimpleNamespace(id=rows[i][2]))
            i += 1
        if not blocks:
            return Alignment(ch.key, doc_id, [], s, e, "none", 0.0, rev, "no block at the span")
        return Alignment(ch.key, doc_id, [b.id for b in blocks], s, e, method, q, rev)

    def _fts(self, ch: Chunk, doc_id: str | None = None) -> Alignment | None:
        from ..core.text import query_terms

        terms = query_terms(" ".join(ch.text.split()[:40]))[:12]
        if not terms or not hasattr(self.repo, "fts_query"):
            return None
        rows = self.repo.fts_query(terms, mode="or", document_id=doc_id, limit=1)
        if not rows:
            return None
        b = rows[0][0]
        return Alignment(
            ch.key,
            b.document_id,
            [b.id],
            b.start_offset,
            b.end_offset,
            "fts",
            QUALITY_FTS,
            self._revision(b.document_id),
            "located by full-text search only",
        )

    def _save(self, rows: Sequence[Alignment]) -> None:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        with self.repo.transaction():
            self.conn.executemany(
                "INSERT OR REPLACE INTO chunk_alignment VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        a.chunk_id,
                        a.document_id,
                        json.dumps(a.block_ids),
                        a.start_offset,
                        a.end_offset,
                        a.method,
                        a.quality,
                        a.source_revision,
                        now,
                    )
                    for a in rows
                ],
            )


def stored_alignment(repo: Any, chunk_id: str) -> Alignment | None:
    """The stored alignment of a chunk id, or None (also when the table does not exist)."""
    conn = getattr(repo, "conn", None)
    if conn is None:
        return None
    try:
        r = conn.execute(
            "SELECT chunk_id, document_id, block_ids, start_offset, end_offset, method, quality, "
            "source_revision FROM chunk_alignment WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
    except Exception:  # noqa: BLE001 - no alignment table yet
        return None
    if r is None:
        return None
    return Alignment(r[0], r[1], json.loads(r[2]), r[3], r[4], r[5], float(r[6]), r[7])


__all__ = [
    "AlignReport",
    "Alignment",
    "Chunk",
    "ChunkAligner",
    "as_chunk",
    "chunk_key",
    "stored_alignment",
]
