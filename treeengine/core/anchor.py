"""Anchors: where a retriever says the document is worth reading.

Every retriever answers the same question - which positions of which documents deserve to be
read for this query - so every retriever produces anchors:

    lexical    FTS       -> BlockAnchor
    semantic   Vector    -> BlockAnchor
    structural Tree      -> NodeAnchor (a whole section is worth reading)
    expansion  Graph-lite-> BlockAnchor (source "graph")

Anchors are fused by rank (weighted RRF: raw BM25, cosine and LLM scores are never added) and
handed to the ContextBuilder, which turns them into ContextSpans. ``Evidence`` stays the
citable record of a hit (``Anchor.evidence``); a ``ContextSpan`` is what the model reads.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import Evidence

AnchorKind = Literal["block", "node"]


@dataclass
class Anchor:
    """A ranked document position. ``kind`` is "block" (one block) or "node" (a section).

    ``rank`` is 1-based within the list that produced it (after fusion: the fused rank);
    ``score`` is that retriever's own score, kept for display only - fusion uses ranks.
    ``evidence`` is the citable hit behind a block anchor (content, page, title)."""

    kind: AnchorKind
    document_id: str
    block_id: str | None = None
    node_id: str | None = None
    source: str = ""
    rank: int = 0
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: Evidence | None = None

    def __post_init__(self) -> None:
        if self.kind == "block" and not self.block_id:
            raise ValueError("a block anchor needs a block_id")
        if self.kind == "node" and not self.node_id:
            raise ValueError("a node anchor needs a node_id")

    @property
    def key(self) -> tuple[str, str]:
        return (
            ("block", str(self.block_id)) if self.kind == "block" else ("node", str(self.node_id))
        )

    @property
    def sources(self) -> list[str]:
        """Every retriever that proposed this position (several after fusion)."""
        return list(
            self.metadata.get("retrieval_paths") or [s for s in self.source.split("+") if s]
        )


def BlockAnchor(  # noqa: N802 - reads as a type at call sites
    document_id: str,
    block_id: str,
    node_id: str | None = None,
    source: str = "",
    rank: int = 0,
    score: float | None = None,
    metadata: dict[str, Any] | None = None,
    evidence: Evidence | None = None,
) -> Anchor:
    return Anchor(
        "block", document_id, block_id, node_id, source, rank, score, metadata or {}, evidence
    )


def NodeAnchor(  # noqa: N802
    document_id: str,
    node_id: str,
    source: str = "tree",
    rank: int = 0,
    score: float | None = None,
    metadata: dict[str, Any] | None = None,
    evidence: Evidence | None = None,
) -> Anchor:
    return Anchor("node", document_id, None, node_id, source, rank, score, metadata or {}, evidence)


RetrievalAnchor = Anchor  # V0.6 name


def anchor_from_evidence(e: Evidence, source: str | None = None, rank: int = 0) -> Anchor:
    """Evidence with a block id -> BlockAnchor; a node-level hit -> NodeAnchor."""
    src = source or e.source
    if e.block_id:
        return BlockAnchor(
            e.document_id, e.block_id, e.node_id, src, rank, e.score, dict(e.metadata), e
        )
    if e.node_id:
        return NodeAnchor(e.document_id, e.node_id, src, rank, e.score, dict(e.metadata), e)
    raise ValueError("evidence without a block or node id cannot anchor a context")


def anchors_from_evidence(evidence: Iterable[Evidence], source: str | None = None) -> list[Anchor]:
    out: list[Anchor] = []
    for e in evidence:
        if e.block_id or e.node_id:
            out.append(anchor_from_evidence(e, source, rank=len(out) + 1))
    return out


__all__ = [
    "Anchor",
    "AnchorKind",
    "BlockAnchor",
    "NodeAnchor",
    "RetrievalAnchor",
    "anchor_from_evidence",
    "anchors_from_evidence",
]
