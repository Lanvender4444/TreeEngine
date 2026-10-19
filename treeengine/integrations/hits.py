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

Every resolved hit records how it was mapped (TE-028) in its metadata:

    mapping_status   exact | approximate | unmapped | stale
    mapped_by        block_id | offsets | text | text+fts | page | none
    mapping_quality  1.0 (block id, consistent) ... 0.0 (unmapped)
    mapping_reason   why it is not exact (ambiguous text, page mismatch, unknown block id ...)

``strict=True`` only accepts exact mappings: the block id must exist and agree with the hit's
document and text; offsets and text must point at one block unambiguously; a
``source_revision`` must equal the document's current revision. Everything else is unmapped
(and, with ``on_unmapped="drop"`` - the strict default - left out of the context).
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
    source_revision: str | None = None  # "sha256:..." of the document the hit was taken from
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


QUALITY = {"block_id": 1.0, "offsets": 0.9, "text": 0.8, "text+fts": 0.6, "page": 0.3}


def _revision(repo: Repository, document_id: str) -> str | None:
    get = getattr(repo, "get_index_status", None)
    rec = get(document_id) if get is not None else None
    return rec.get("source_revision") if rec else None


def _text_blocks(blocks: list[Block], text: str) -> list[Block]:
    """Every block containing the hit's opening words (to detect ambiguous text)."""
    head = _norm(text)[:80]
    return [b for b in blocks if head and head in _norm(b.content)] if head else []


def resolve_hit(
    repo: Repository,
    hit: Any,
    *,
    document_id: str | None = None,
    rank: int = 0,
    strict: bool = False,
) -> Anchor | Evidence:
    if isinstance(hit, (Anchor, Evidence)):
        return hit
    h = as_external(hit)
    blocks = _DocBlocks(repo)
    md = {**h.metadata, "external": True}

    def unmapped(reason: str, doc: str | None = None, status: str = "unmapped") -> Evidence:
        return Evidence(
            doc or h.document_id or "external",
            None,
            None,
            h.text,
            "external",
            h.score,
            {
                **md,
                "page": h.page,
                "unmapped": True,
                "mapped_by": "none",
                "mapping_status": status,
                "mapping_quality": 0.0,
                "mapping_reason": reason,
            },
        )

    def done(b: Block, how: str, problems: list[str]) -> Anchor | Evidence:
        rev = _revision(repo, b.document_id)
        if h.source_revision and rev and h.source_revision != rev:
            if strict:
                return unmapped(f"stale: hit from {h.source_revision}, now {rev}", b.document_id,
                                "stale")  # fmt: skip
            problems.append(f"stale revision {h.source_revision} (now {rev})")
        quality = QUALITY.get(how, 0.5) * (0.5 if problems else 1.0)
        status = (
            "exact" if not problems and how in ("block_id", "offsets", "text") else ("approximate")
        )
        if strict and status != "exact":
            return unmapped("; ".join(problems) or f"{how} is not an exact mapping", b.document_id)
        info = {
            "mapping_status": "stale" if any(p.startswith("stale") for p in problems) else status,
            "mapping_quality": round(quality, 2),
            "mapping_reason": "; ".join(problems) or None,
        }
        return _anchor(b, h, rank, {**md, **info}, how)

    if h.block_id:
        b = repo.get_block(h.block_id)
        if b is not None:
            issues: list[str] = []
            if h.document_id and h.document_id != b.document_id:
                issues.append(f"block {b.id} belongs to {b.document_id}, not {h.document_id}")
            if h.text and _by_text([b], h.text) is None:
                issues.append("hit text not found in the block")
            if h.page is not None and b.page is not None and int(h.page) != b.page:
                issues.append(f"page {h.page} != block page {b.page}")
            return done(b, "block_id", issues)
        if strict:
            return unmapped(f"unknown block id {h.block_id}")
    doc = h.document_id
    if doc and repo.get_document(doc) is None:
        if strict:
            return unmapped(f"unknown document {doc}")
        doc = None
    if doc is None and h.uri:
        d = repo.find_document_by_uri(h.uri)
        doc = d.id if d else None
    doc = doc or document_id
    if doc:
        bl = blocks(doc)
        problems: list[str] = []
        if h.start_offset is not None:
            end = h.end_offset if h.end_offset is not None else h.start_offset + max(1, len(h.text))
            b = _by_offsets(bl, int(h.start_offset), int(end))
            if b is not None:
                if h.text and _by_text([b], h.text) is None:
                    problems.append("hit text not found at these offsets")
                return done(b, "offsets", problems)
            problems.append("offsets outside every block")
        if h.text:
            cands = _text_blocks(bl, h.text)
            b = cands[0] if cands else _by_text(bl, h.text)
            if b is not None:
                if len(cands) > 1:
                    problems.append(f"ambiguous: text occurs in {len(cands)} blocks")
                if h.page is not None and b.page is not None and int(h.page) != b.page:
                    problems.append(f"page {h.page} != block page {b.page}")
                if not cands:
                    problems.append("matched a block opening inside a longer hit")
                return done(b, "text", problems)
        if h.page is not None:
            b = next((x for x in bl if x.page == int(h.page)), None)
            if b is not None:
                return done(b, "page", [*problems, "located by page only"])
        return unmapped("; ".join(problems) or "no block matches", doc)
    if h.text:
        terms = query_terms(" ".join(h.text.split()[:30]))[:12]
        for b, _ in repo.fts_query(terms, mode="and", limit=20) if terms else []:
            if _by_text([b], h.text) is not None:
                return done(b, "text+fts", ["no document id: found by full-text search"])
    return unmapped("no document id and no matching text")


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
    repo: Repository,
    hits: Iterable[Any],
    *,
    document_id: str | None = None,
    strict: bool = False,
    on_unmapped: str | None = None,
) -> list[Anchor | Evidence]:
    """Map hits to anchors. ``on_unmapped``: "passthrough" (default; strict: "drop") keeps
    an unmapped hit as plain text, "drop" leaves it out, "error" raises ``MappingError``."""
    mode = on_unmapped or ("drop" if strict else "passthrough")
    if mode not in ("passthrough", "drop", "error"):
        raise ValueError(f"on_unmapped must be passthrough | drop | error, not {mode!r}")
    out: list[Anchor | Evidence] = []
    for i, h in enumerate(hits, 1):
        r = resolve_hit(repo, h, document_id=document_id, rank=i, strict=strict)
        if isinstance(r, Evidence) and (r.metadata or {}).get("unmapped"):
            if mode == "drop":
                continue
            if mode == "error":
                from ..api.errors import MappingError

                raise MappingError(
                    f"hit {i} could not be mapped: {r.metadata.get('mapping_reason')}",
                    hit=i,
                    reason=r.metadata.get("mapping_reason"),
                    status=r.metadata.get("mapping_status"),
                )
        out.append(r)
    return out


def mapping_of(item: Any) -> dict[str, Any] | None:
    """The mapping record of a resolved hit (None for TreeEngine's own anchors)."""
    md = getattr(item, "metadata", None) or {}
    if not md.get("external"):
        return None
    return {
        "status": md.get("mapping_status", "approximate"),
        "mapped_by": md.get("mapped_by"),
        "quality": float(md.get("mapping_quality", 0.0)),
        "reason": md.get("mapping_reason"),
    }


__all__ = ["QUALITY", "ExternalHit", "as_external", "mapping_of", "resolve_hit", "resolve_hits"]
