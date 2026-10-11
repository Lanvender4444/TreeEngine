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

    @property
    def query_type(self) -> str:
        return str(getattr(self.plan, "query_type", ""))
