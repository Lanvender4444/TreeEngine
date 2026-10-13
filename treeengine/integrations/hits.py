"""External hits -> TreeEngine anchors, so any retriever's results can use Context Reconstruction.

An external retriever (Pinecone, Qdrant, Elasticsearch, a home-grown index ...) returns pieces
of text with some identifiers. ``resolve_hits`` maps each one back to a stable block of a
document TreeEngine has ingested, trying in order:

    1. ``block_id``                       an id TreeEngine handed out
    2. document + ``start_offset``/``end_offset``  the block overlapping the range most
    3. document + ``text``                the block the text starts in (normalised substring)
    4. document + ``page``                the first block of that page
    5. ``text`` alone                     FTS for the text's words, verified by substring

The document comes from ``document_id``, else ``uri`` / ``source`` (the ingested file), else
the caller's default. A hit that cannot be mapped passes through as plain text (it still gets
a span of its own, within the budget).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.anchor import Anchor, BlockAnchor
from ..core.models import Block, Evidence
from ..core.protocols import Repository
from ..core.text import query_terms


@dataclass
class ExternalHit:
    text: str = ""
    document_id: str | None = None
    block_id: str | None = None
    page: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    uri: str | None = None  # the source file / URL the document was ingested from
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_FIELDS = set(ExternalHit.__dataclass_fields__)
_ALIASES = {"content": "text", "page_content": "text", "source": "uri", "doc_id": "document_id"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def as_external(hit: Any) -> ExternalHit:
    """A dict (``text`` / ``content`` / ``page_content``; ids and positions at the top level or
    under ``metadata``), an object with those attributes, or an ExternalHit."""
    if isinstance(hit, ExternalHit):
        return hit
    if isinstance(hit, Mapping):
        data: dict[str, Any] = {}
        meta = dict(hit.get("metadata") or {})
        for src in (meta, hit):
            for k, v in src.items():
                k = _ALIASES.get(k, k)
                if k in _FIELDS and k != "metadata" and v is not None:
                    data[k] = v
        data["metadata"] = meta
        return ExternalHit(**data)
    # LangChain / LlamaIndex style objects
    text = getattr(hit, "page_content", None) or getattr(hit, "text", None) or ""
    meta = dict(getattr(hit, "metadata", None) or {})
    return as_external({"text": text, "metadata": meta})


class _DocBlocks:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.cache: dict[str, list[Block]] = {}

    def __call__(self, document_id: str) -> list[Block]:
        if document_id not in self.cache:
            self.cache[document_id] = self.repo.get_document_blocks(document_id)
        return self.cache[document_id]


def _by_offsets(blocks: list[Block], start: int, end: int) -> Block | None:
    best, best_ov = None, 0
    for b in blocks:
        if b.start_offset is None or b.end_offset is None:
            continue
        ov = min(end, b.end_offset) - max(start, b.start_offset)
        if ov > best_ov:
            best, best_ov = b, ov
    return best


def _by_text(blocks: list[Block], text: str) -> Block | None:
    """The block the hit starts in: the hit's opening words inside a block, or a block's
    opening words inside the hit (a chunk spanning several blocks)."""
    t = _norm(text)
    if not t:
        return None
    head = t[:80]
    for b in blocks:
        if head and head in _norm(b.content):
            return b
    for b in blocks:
        bh = _norm(b.content)[:60]
        if len(bh) >= 20 and bh in t:
            return b
    return None


def resolve_hit(
    repo: Repository, hit: Any, *, document_id: str | None = None, rank: int = 0
) -> Anchor | Evidence:
    if isinstance(hit, (Anchor, Evidence)):
        return hit
    h = as_external(hit)
    blocks = _DocBlocks(repo)
    md = {**h.metadata, "external": True}
    if h.block_id:
        b = repo.get_block(h.block_id)
        if b is not None:
            return _anchor(b, h, rank, md, "block_id")
    doc = h.document_id
    if doc and repo.get_document(doc) is None:
        doc = None
    if doc is None and h.uri:
        d = repo.find_document_by_uri(h.uri)
        doc = d.id if d else None
    doc = doc or document_id
    if doc:
        bl = blocks(doc)
        hit_b = None
        how = ""
        if h.start_offset is not None:
            end = h.end_offset if h.end_offset is not None else h.start_offset + max(1, len(h.text))
            hit_b, how = _by_offsets(bl, int(h.start_offset), int(end)), "offsets"
        if hit_b is None and h.text:
            hit_b, how = _by_text(bl, h.text), "text"
        if hit_b is None and h.page is not None:
            hit_b = next((b for b in bl if b.page == int(h.page)), None)
            how = "page"
        if hit_b is not None:
            return _anchor(hit_b, h, rank, md, how)
    elif h.text:
        terms = query_terms(" ".join(h.text.split()[:30]))[:12]
        for b, _ in repo.fts_query(terms, mode="and", limit=20) if terms else []:
            if _by_text([b], h.text) is not None:
                return _anchor(b, h, rank, md, "text+fts")
    return Evidence(
        doc or "external",
        None,
        None,
        h.text,
        "external",
        h.score,
        {**md, "page": h.page, "unmapped": True},
    )


def _anchor(b: Block, h: ExternalHit, rank: int, md: dict[str, Any], how: str) -> Anchor:
    ev = Evidence(
        b.document_id,
        b.node_id,
        b.id,
        b.content,
        "external",
        h.score,
        {**md, "page": b.page, "mapped_by": how},
    )
    return BlockAnchor(b.document_id, b.id, b.node_id, "external", rank, h.score, ev.metadata, ev)


def resolve_hits(
    repo: Repository, hits: Iterable[Any], *, document_id: str | None = None
) -> list[Anchor | Evidence]:
    return [resolve_hit(repo, h, document_id=document_id, rank=i) for i, h in enumerate(hits, 1)]


__all__ = ["ExternalHit", "as_external", "resolve_hit", "resolve_hits"]
