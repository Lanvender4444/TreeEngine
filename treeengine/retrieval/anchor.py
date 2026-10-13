"""Anchor fusion (retrieval side). The Anchor data model lives in ``core.anchor`` so the
ContextBuilder can consume anchors without depending on any retriever."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


__all__ = [
    "Anchor",
    "AnchorKind",
    "BlockAnchor",
    "NodeAnchor",
    "RetrievalAnchor",
    "anchor_from_evidence",
    "anchors_from_evidence",
    "fuse_anchors",
]
