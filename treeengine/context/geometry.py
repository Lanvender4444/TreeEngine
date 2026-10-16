"""Evidence geometry (V0.12): how the retrieved evidence is laid out in the document.

Deterministic - no LLM, no model, no dataset-specific condition. Input: ranked anchors (the
fused FTS + Vector list); output: features of the top-k hits and their clusters.

Clusters: hits of the same document are sorted by position and joined when they share a node
or the gap between them is at most ``MAX_GAP_CHARS`` characters (fixed, ~500 tokens): one
cluster = one evidence region worth reading as a coherent span.

Features (``EvidenceGeometry``):

  distinct_sections      distinct nodes among the top-k hits
  same_section_ratio     share of the top-k hits in the most common node
  page_span / offset_span  extent of the hits in the best hit's document
  cluster_count          clusters among the top-k hits
  largest_cluster_ratio  rank mass of the largest cluster / total rank mass
  top1_dominance         fused score of hit 1 / fused score of hit 2
  score_decay            fused score of hit k / fused score of hit 1
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.protocols import Repository

TOP_K = 10
MAX_GAP_CHARS = 2000
RRF_K = 60


@dataclass
class EvidenceCluster:
    document_id: str
    node_id: str | None  # the node shared by all its hits, else None
    block_ids: list[str]
    rank_min: int
    rank_max: int
    score_mass: float
    start_offset: int | None
    end_offset: int | None
    page_start: int | None
    page_end: int | None
    lo: int = 0  # block index range in the document (for the ContextBuilder)
    hi: int = 0
    ranks: list[int] = field(default_factory=list)


@dataclass
class EvidenceGeometry:
    top_k: int
    distinct_sections: int
    same_section_ratio: float
    page_span: int
    offset_span: int
    cluster_count: int
    largest_cluster_ratio: float
    top1_dominance: float
    score_decay: float
    clusters: list[EvidenceCluster] = field(default_factory=list)

    def features(self) -> dict[str, float]:
        return {
            "distinct_sections": self.distinct_sections,
            "same_section_ratio": self.same_section_ratio,
            "page_span": self.page_span,
            "offset_span": self.offset_span,
            "cluster_count": self.cluster_count,
            "largest_cluster_ratio": self.largest_cluster_ratio,
            "top1_dominance": self.top1_dominance,
            "score_decay": self.score_decay,
        }


def _hits(anchors: Sequence[Any], repo: Repository, top_k: int) -> list[tuple[int, Any, Any]]:
    """(1-based rank, anchor, block) of the first ``top_k`` block hits."""
    out: list[tuple[int, Any, Any]] = []
    seen: set[str] = set()
    for a in anchors:
        bid = getattr(a, "block_id", None)
        if not bid or bid in seen:
            continue
        b = repo.get_block(bid)
        if b is None:
            continue
        seen.add(bid)
        out.append((len(out) + 1, a, b))
        if len(out) >= top_k:
            break
    return out


def cluster_hits(
    hits: list[tuple[int, Any, Any]],
    index_of: dict[str, int] | None = None,
    max_gap_chars: int = MAX_GAP_CHARS,
) -> list[EvidenceCluster]:
    by_doc: dict[str, list[tuple[int, Any, Any]]] = {}
    for h in hits:
        by_doc.setdefault(h[2].document_id, []).append(h)
    clusters: list[EvidenceCluster] = []
    for doc, hs in by_doc.items():
        hs.sort(key=lambda h: (h[2].start_offset or 0, h[0]))
        groups: list[list[tuple[int, Any, Any]]] = []
        for h in hs:
            if groups:
                prev = groups[-1][-1][2]
                gap = (h[2].start_offset or 0) - (prev.end_offset or prev.start_offset or 0)
                same_node = h[2].node_id is not None and h[2].node_id == prev.node_id
                if same_node or gap <= max_gap_chars:
                    groups[-1].append(h)
                    continue
            groups.append([h])
        for g in groups:
            nodes = {h[2].node_id for h in g}
            pages = [h[2].page for h in g if h[2].page is not None]
            idx = [index_of[h[2].id] for h in g if index_of and h[2].id in index_of]
            clusters.append(
                EvidenceCluster(
                    document_id=doc,
                    node_id=next(iter(nodes)) if len(nodes) == 1 else None,
                    block_ids=[h[2].id for h in g],
                    rank_min=min(h[0] for h in g),
                    rank_max=max(h[0] for h in g),
                    score_mass=sum(1.0 / (RRF_K + h[0]) for h in g),
                    start_offset=min((h[2].start_offset or 0) for h in g),
                    end_offset=max((h[2].end_offset or 0) for h in g),
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    lo=min(idx) if idx else 0,
                    hi=max(idx) if idx else 0,
                    ranks=sorted(h[0] for h in g),
                )
            )
    clusters.sort(key=lambda c: c.rank_min)
    return clusters


def analyze_geometry(
    anchors: Sequence[Any],
    repo: Repository,
    top_k: int = TOP_K,
    index_of: dict[str, int] | None = None,
) -> EvidenceGeometry:
    hits = _hits(anchors, repo, top_k)
    if not hits:
        return EvidenceGeometry(0, 0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0)
    nodes = Counter(h[2].node_id for h in hits)
    doc = hits[0][2].document_id
    same_doc = [h[2] for h in hits if h[2].document_id == doc]
    pages = [b.page for b in same_doc if b.page is not None]
    starts = [b.start_offset for b in same_doc if b.start_offset is not None]
    ends = [b.end_offset for b in same_doc if b.end_offset is not None]
    clusters = cluster_hits(hits, index_of)
    total = sum(c.score_mass for c in clusters) or 1.0
    scores = [float(getattr(h[1], "score", 0.0) or 0.0) for h in hits]
    return EvidenceGeometry(
        top_k=len(hits),
        distinct_sections=len(nodes),
        same_section_ratio=nodes.most_common(1)[0][1] / len(hits),
        page_span=(max(pages) - min(pages)) if pages else 0,
        offset_span=(max(ends) - min(starts)) if starts and ends else 0,
        cluster_count=len(clusters),
        largest_cluster_ratio=max(c.score_mass for c in clusters) / total,
        top1_dominance=(scores[0] / scores[1]) if len(scores) > 1 and scores[1] > 0 else 1.0,
        score_decay=(scores[-1] / scores[0]) if scores and scores[0] > 0 else 0.0,
        clusters=clusters,
    )


__all__ = ["EvidenceCluster", "EvidenceGeometry", "analyze_geometry", "cluster_hits"]
