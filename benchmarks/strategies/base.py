"""Benchmark strategy interface.

Every strategy - TreeEngine's own retrievers, the traditional chunk-RAG baselines, full context -
answers the same two calls and returns the same records, so the retrieval metrics, the answer
model and the judge see no difference between them except the evidence.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from treeengine.core.models import Evidence
from treeengine.retrieval.result import Trace

from ..metrics.cost import count_tokens
from ..metrics.retrieval import CONTEXT_K


@dataclass
class RetrievalRun:
    evidence: list[Evidence]
    latency_ms: float = 0.0
    context_tokens: int = 0  # tokens of the top CONTEXT_K evidence items
    llm_calls: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0
    input_tokens: int = 0  # LLM tokens spent by retrieval itself (LLM navigation / planner)
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_ids(self) -> list[str] | None:
        return self.metadata.get("target_ids")

    @property
    def visited_ratio(self) -> float | None:
        return self.metadata.get("visited_ratio")

    @property
    def tree_depth(self) -> int | None:
        return self.metadata.get("tree_depth")

    @property
    def nodes_expanded(self) -> int | None:
        return self.metadata.get("nodes_expanded")


@dataclass
class IndexStats:
    """What it costs to build the index a strategy needs (shared indexes are counted in full
    for every strategy that needs them: it is the cost of deploying that strategy alone)."""

    strategy: str
    documents: int = 0
    blocks: int = 0  # indexed units: blocks, chunks, vectors
    elapsed_seconds: float = 0.0
    llm_calls: int = 0
    llm_tokens: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0
    index_bytes: int = 0
    estimated_cost: float | None = None
    parts: list[str] = field(default_factory=list)

    def __add__(self, other: IndexStats) -> IndexStats:
        cost = None
        if self.estimated_cost is not None or other.estimated_cost is not None:
            cost = (self.estimated_cost or 0.0) + (other.estimated_cost or 0.0)
        return IndexStats(
            strategy=self.strategy,
            documents=max(self.documents, other.documents),
            blocks=self.blocks + other.blocks,
            elapsed_seconds=self.elapsed_seconds + other.elapsed_seconds,
            llm_calls=self.llm_calls + other.llm_calls,
            llm_tokens=self.llm_tokens + other.llm_tokens,
            embedding_calls=self.embedding_calls + other.embedding_calls,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            index_bytes=self.index_bytes + other.index_bytes,
            estimated_cost=cost,
            parts=self.parts + other.parts,
        )


@runtime_checkable
class BenchmarkStrategy(Protocol):
    name: str
    family: str  # "treeengine" | "traditional" | "baseline"

    def index(self, documents: Sequence[str] | None = None) -> IndexStats: ...

    def retrieve(
        self, query: str, *, document_id: str | None = None, limit: int = 5
    ) -> RetrievalRun: ...


def tree_metadata(trace: Trace) -> dict[str, Any]:
    """Tree-specific metrics from the trace: targets, visited ratio, depth, nodes expanded."""
    steps = trace.of("tree")
    if not steps:
        return {}
    visited = sum(s["visited_nodes"] for s in steps)
    total = sum(s["total_nodes"] for s in steps)
    levels = [lv for s in steps for lv in s.get("levels", [])]
    return {
        "target_ids": [t for s in steps for t in s.get("target_ids", [])],
        "visited_ratio": visited / total if total else None,
        "tree_depth": max((lv["level"] for lv in levels), default=-1) + 1,
        "nodes_expanded": sum(len(lv.get("selected", [])) for lv in levels),
    }


Search = Callable[[str, str | None, int, Trace], list[Evidence]]


class Strategy:
    """A named search function plus the indexes it needs from the Workspace.

    ``run`` handles the bookkeeping every strategy shares: latency, LLM / embedding usage
    deltas, context size and tree metrics."""

    qa_only = False

    def __init__(
        self,
        name: str,
        family: str,
        workspace: Any,
        search: Search,
        requires: Sequence[str] = ("corpus",),
        description: str = "",
    ) -> None:
        self.name = name
        self.family = family
        self.ws = workspace
        self.search = search
        self.requires = tuple(requires)
        self.description = description

    def index(self, documents: Sequence[str] | None = None) -> IndexStats:
        total = IndexStats(self.name)
        for r in self.requires:
            total = total + self.ws.resource_stats(r)
        total.strategy = self.name
        return total

    def retrieve(
        self, query: str, *, document_id: str | None = None, limit: int = 5
    ) -> RetrievalRun:
        ws = self.ws
        ws.require(self.requires)  # build indexes outside the timed / metered section
        llm0 = ws.llm_usage()
        emb0 = ws.embedding_usage()
        t0 = time.perf_counter()
        trace = Trace()
        evidence = self.search(query, document_id, limit, trace)[:limit]
        ms = (time.perf_counter() - t0) * 1000
        llm = ws.llm_usage().since(llm0) if llm0 is not None else None
        emb = ws.embedding_usage().since(emb0) if emb0 is not None else None
        sizes = [count_tokens(e.content) for e in evidence]
        return RetrievalRun(
            evidence=evidence,
            latency_ms=ms,
            context_tokens=sum(sizes[:CONTEXT_K]),
            llm_calls=llm.calls if llm else 0,
            input_tokens=llm.input_tokens if llm else 0,
            output_tokens=llm.output_tokens if llm else 0,
            embedding_calls=emb.calls if emb else 0,
            embedding_tokens=emb.tokens if emb else 0,
            metadata={**tree_metadata(trace), "evidence_tokens": sizes},
        )


__all__ = [
    "BenchmarkStrategy",
    "IndexStats",
    "RetrievalRun",
    "Search",
    "Strategy",
    "tree_metadata",
]
