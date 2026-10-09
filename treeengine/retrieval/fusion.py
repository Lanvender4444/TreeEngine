"""Evidence fusion: the same block found by several retrievers becomes ONE Evidence that keeps
every retriever's signal, instead of naive list concatenation.

V0.3 uses Reciprocal Rank Fusion (Cormack et al., 2009): score = sum 1 / (k + rank). It needs no
score calibration between BM25, cosine similarity and tree scores.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..core.models import Evidence
from .result import Trace


def _key(e: Evidence) -> tuple[str, str]:
    return ("block", e.block_id) if e.block_id else ("node", e.node_id or "")


class EvidenceMerger:
    def __init__(self, k: int = 60, weights: dict[str, float] | None = None) -> None:
        self.k = k
        self.weights = weights or {}

    def rrf(
        self,
        lists: dict[str, list[Evidence]],
        limit: int | None = None,
        trace: Trace | None = None,
    ) -> list[Evidence]:
        fused: dict[tuple[str, str], float] = {}
        signals: dict[tuple[str, str], dict[str, Any]] = {}
        first: dict[tuple[str, str], Evidence] = {}
        for name, items in lists.items():
            seen: set[tuple[str, str]] = set()
            for rank, e in enumerate(items):
                key = _key(e)
                if key in seen:
                    continue
                seen.add(key)
                w = self.weights.get(name, 1.0)
                fused[key] = fused.get(key, 0.0) + w / (self.k + rank + 1)
                signals.setdefault(key, {})[name] = {"rank": rank + 1, "score": e.score}
                first.setdefault(key, e)
        order = sorted(fused, key=lambda key: -fused[key])
        if limit is not None:
            order = order[:limit]
        out = []
        for key in order:
            base = first[key]
            sources = sorted(signals[key])
            out.append(
                replace(
                    base,
                    source="+".join(sources) if len(sources) > 1 else sources[0],
                    score=round(fused[key], 6),
                    metadata={**base.metadata, "signals": signals[key], "fusion": "rrf"},
                )
            )
        if trace is not None:
            trace.add(
                "fusion",
                method="rrf",
                k=self.k,
                weights=self.weights or None,
                inputs={name: len(items) for name, items in lists.items()},
                output=len(out),
                overlap=sum(1 for key in order if len(signals[key]) > 1),
            )
        return out
