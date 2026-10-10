"""Benchmark side of context reconstruction (Retrieval Unit != Reading Unit).

The reconstruction itself lives in :mod:`treeengine.context`; this module fixes the benchmark's
choices so every run is comparable:

* tokens are counted with tiktoken cl100k (``metrics.cost.count_tokens``), like every other
  token figure in the reports;
* the best-ranked item is kept even when it alone exceeds the budget (a 600-token chunk at a
  500-token budget), so no strategy answers from an empty context;
* spans are given to the answer model in rank order of their best anchor.

Modes (``MODES``): raw, neighbor1, neighbor2, adaptive600, section600 and ``auto`` (the V0.5
default policy: section600 at budgets <= 1000 tokens, adaptive600 above). Traditional chunks
have no block geometry and are only used ``raw``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from treeengine.context import BlockContextBuilder, ContextSpan
from treeengine.context.span import evidence_id
from treeengine.core.models import Evidence

from .metrics.cost import count_tokens

MODES = ["raw", "neighbor1", "neighbor2", "adaptive600", "section600", "auto"]


class ContextBuilder:
    def __init__(self, repo: Any) -> None:
        self.inner = BlockContextBuilder(
            repo, token_counter=count_tokens, order="rank", allow_first_overflow=True
        )

    def spans(self, evidence: Sequence[Evidence], mode: str, budget: int) -> list[ContextSpan]:
        return self.inner.build(evidence, token_budget=budget, policy=mode)

    def build(
        self, evidence: Sequence[Evidence], mode: str, budget: int
    ) -> tuple[list[Evidence], int]:
        """-> (what the answer model reads, as Evidence items; total tokens)."""
        spans = self.spans(evidence, mode, budget)
        return as_evidence(evidence, spans), sum(s.token_count for s in spans)


def as_evidence(evidence: Sequence[Evidence], spans: Sequence[ContextSpan]) -> list[Evidence]:
    """Spans -> Evidence; pass-through items come back as the original Evidence objects."""
    by_id = {evidence_id(e): e for e in evidence}
    out = []
    for s in spans:
        passthrough = len(s.source_evidence_ids) == 1 and (
            not s.block_ids or s.block_ids == [s.source_evidence_ids[0]]
        )
        orig = by_id.get(s.source_evidence_ids[0]) if passthrough else None
        out.append(orig if orig is not None and orig.content == s.content else s.to_evidence())
    return out


def fragmentation(spans: Sequence[ContextSpan]) -> dict[str, float]:
    """Reading-context shape: how many disconnected pieces, how long, how many cross sections."""
    if not spans:
        return {"span_count": 0, "avg_span_tokens": 0, "max_span_tokens": 0}
    sizes = [s.token_count for s in spans]
    blocks = [b for s in spans for b in s.block_ids]
    return {
        "span_count": len(spans),
        "avg_span_tokens": sum(sizes) / len(sizes),
        "max_span_tokens": max(sizes),
        "section_crossings": sum(s.section_crossings for s in spans),
        "duplicate_ratio": (1 - len(set(blocks)) / len(blocks)) if blocks else 0.0,
    }


__all__ = ["MODES", "ContextBuilder", "as_evidence", "fragmentation"]
