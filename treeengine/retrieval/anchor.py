"""Anchor fusion (retrieval side). The Anchor data model lives in ``core.anchor`` so the
ContextBuilder can consume anchors without depending on any retriever."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..core.anchor import (
    Anchor,
    AnchorKind,
    BlockAnchor,
    NodeAnchor,
    RetrievalAnchor,
    anchor_from_evidence,
    anchors_from_evidence,
)


def fuse_anchors(
    lists: Mapping[str, Sequence[Anchor]],
    weights: dict[str, float] | None = None,
    k: int = 60,
    limit: int | None = None,
) -> list[Anchor]:
    """Weighted Reciprocal Rank Fusion over anchors of any kind: score = sum w / (k + rank).
    The same position from several lists becomes one anchor that keeps every list's rank
    (``metadata["signals"]``) and the paths that found it (``metadata["retrieval_paths"]``).
    Ties keep first-seen order, exactly like ``EvidenceMerger.rrf`` on the same lists."""
    weights = weights or {}
    fused: dict[tuple[str, str], float] = {}
    signals: dict[tuple[str, str], dict[str, Any]] = {}
    first: dict[tuple[str, str], Anchor] = {}
    for name, items in lists.items():
        w = weights.get(name, 1.0)
        if w <= 0:
            continue
        seen: set[tuple[str, str]] = set()
        for rank, a in enumerate(items):
            key = a.key
            if key in seen:
                continue
            seen.add(key)
            fused[key] = fused.get(key, 0.0) + w / (k + rank + 1)
            signals.setdefault(key, {})[name] = {"rank": rank + 1, "score": a.score}
            first.setdefault(key, a)
    order = sorted(fused, key=lambda key: -fused[key])
    if limit is not None:
        order = order[:limit]
    out = []
    for i, key in enumerate(order, 1):
        base = first[key]
        paths = sorted(signals[key])
        md = {**base.metadata, "signals": signals[key], "fusion": "rrf", "retrieval_paths": paths}
        ev = base.evidence
        if ev is not None:
            ev = replace(
                ev,
                source="+".join(paths) if len(paths) > 1 else paths[0],
                score=round(fused[key], 6),
                metadata={
                    **ev.metadata,
                    "signals": signals[key],
                    "fusion": "rrf",
                    "retrieval_paths": paths,
                },
            )
        out.append(
            replace(
                base,
                source="+".join(paths) if len(paths) > 1 else paths[0],
                rank=i,
                score=round(fused[key], 6),
                metadata=md,
                evidence=ev,
            )
        )
    return out


def absorb_contained(
    anchors: Sequence[Anchor], section_of: Callable[[Anchor], set[str]]
) -> list[Anchor]:
    """Give a NodeAnchor credit for the evidence it contains.

    In plain RRF a NodeAnchor never shares a key with a block, so it only ever gets its own
    tree term and ranks below every block the fast paths agree on - the section is never read
    within a budget. Here each NodeAnchor (in fused order) takes over the best-ranked block
    anchor inside its section (``section_of(node)`` = the node ids it covers): its score becomes
    that block's fused score plus its own tree term - the same credit a block gets for being
    found by two paths - and the block is read as the node's focus
    (``metadata["focus_block_id"]``: the ContextBuilder grows the span around it within the
    section). A node with no evidence inside keeps its tree-only score. Blocks other than the
    focus stay in the list; overlapping spans are merged later."""
    out = list(anchors)
    if not any(a.kind == "node" for a in out):
        return out
    taken: set[tuple[str, str]] = set()
    nodes: dict[tuple[str, str], Anchor] = {}
    for n in [a for a in out if a.kind == "node"]:
        ids = section_of(n)
        best = next(
            (
                a
                for a in out
                if a.kind == "block"
                and a.key not in taken
                and a.document_id == n.document_id
                and a.node_id in ids
            ),
            None,
        )
        if best is None:
            continue
        taken.add(best.key)
        paths = sorted(set(n.sources) | set(best.sources))
        md = {
            **n.metadata,
            "focus_block_id": best.block_id,
            "retrieval_paths": paths,
            "absorbed": {"block_id": best.block_id, "signals": best.metadata.get("signals")},
        }
        score = round((best.score or 0.0) + (n.score or 0.0), 6)
        ev = n.evidence
        if ev is not None:
            ev = replace(ev, source="+".join(paths), score=score, metadata={**ev.metadata, **md})
        nodes[n.key] = replace(n, source="+".join(paths), score=score, metadata=md, evidence=ev)
    merged = [nodes.get(a.key, a) for a in out if a.key not in taken]
    merged.sort(key=lambda a: -(a.score or 0.0))  # stable: ties keep fused order
    return [replace(a, rank=i) for i, a in enumerate(merged, 1)]


__all__ = [
    "Anchor",
    "AnchorKind",
    "BlockAnchor",
    "NodeAnchor",
    "RetrievalAnchor",
    "absorb_contained",
    "anchor_from_evidence",
    "anchors_from_evidence",
    "fuse_anchors",
]
