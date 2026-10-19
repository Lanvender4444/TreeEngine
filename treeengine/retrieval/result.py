"""Traceable execution: every managed retrieval can explain what it did and what it cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..context.span import ContextSpan
from ..core.models import Evidence


class Trace:
    """Append-only list of execution steps. Retrievers accept ``trace=None`` (no-op)."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def add(self, step: str, **data: Any) -> None:
        self.steps.append({"step": step, **data})

    def of(self, step: str) -> list[dict[str, Any]]:
        return [s for s in self.steps if s["step"] == step]


@dataclass
class SearchStats:
    latency_ms: float = 0.0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_tokens_estimated: bool = True
    llm_calls_by_purpose: dict[str, int] = field(default_factory=dict)
    fts_queries: int = 0
    vector_queries: int = 0
    tree_searches: int = 0
    visited_nodes: int = 0  # nodes loaded during tree traversal (summed over documents)
    total_nodes: int = 0  # size of the trees that were traversed
    tree_steps: int = 0  # tree-reasoning steps of the retrieval controller
    candidate_count: int = 0  # evidence items after fusion
    context_tokens: int = 0  # tokens of the reading context (context_spans)

    @property
    def visited_ratio(self) -> float | None:
        return self.visited_nodes / self.total_nodes if self.total_nodes else None


@dataclass
class SearchResult:
    query: str
    evidence: list[Evidence]
    plan: Any  # RetrievalPlan
    trace: list[dict[str, Any]]
    stats: SearchStats
    # what the answer model should read: continuous spans rebuilt around the evidence (filled
    # when a context budget is given; Evidence stays the retrieval record / citations)
    context_spans: list[ContextSpan] = field(default_factory=list)
    # V0.9: the retrieval decisions - ranked positions worth reading (BlockAnchor / NodeAnchor,
    # after fusion). ``evidence`` is their citable record; ``context_spans`` what to read.
    anchors: list[Any] = field(default_factory=list)
    # SDK v1 (TE-021 / TE-022): the preset used, capabilities it had to do without, and the
    # source revision of each document in the result (for stable citations)
    preset: str | None = None
    degradations: list[str] = field(default_factory=list)
    source_revisions: dict[str, str | None] = field(default_factory=dict)

    @property
    def query_type(self) -> str:
        return str(getattr(self.plan, "query_type", ""))

    def to_v1(self, *, include_trace: bool = False, max_evidence: int = 20) -> Any:
        """This result as a versioned ``RetrievalResultV1`` (``.to_dict()`` for JSON)."""
        from ..api.dto import retrieval_v1

        return retrieval_v1(None, self, include_trace=include_trace, max_evidence=max_evidence)
