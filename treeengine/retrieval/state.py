"""Mutable state of one controlled retrieval: observations, anchors, budgets spent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..core.models import Evidence
from .result import Trace


@dataclass
class RetrievalAnchor:
    """Where a path pointed: a block (FTS / Vector hits) or a node (tree reasoning). Internal to
    the controller; node anchors are resolved to blocks before anything is read."""

    kind: Literal["block", "node"]
    document_id: str
    block_id: str | None = None
    node_id: str | None = None
    source: str = ""
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalState:
    query: str
    document_id: str | None = None
    lists: dict[str, list[Evidence]] = field(default_factory=dict)  # path -> ranked evidence
    node_anchors: list[RetrievalAnchor] = field(default_factory=list)
    visited_nodes: set[str] = field(default_factory=set)
    used_retrievers: list[str] = field(default_factory=list)
    step: int = 0
    llm_calls: int = 0
    tree_steps: int = 0
    fts_queries: int = 0
    vector_queries: int = 0
    stop_reason: str = ""
    trace: Trace = field(default_factory=Trace)

    def add(self, path: str, evidence: list[Evidence]) -> None:
        """Append a path's hits (several steps of one path accumulate in one ranked list)."""
        have = {(e.document_id, e.block_id, e.node_id) for e in self.lists.get(path, [])}
        new = [e for e in evidence if (e.document_id, e.block_id, e.node_id) not in have]
        self.lists.setdefault(path, []).extend(new)
        if path not in self.used_retrievers:
            self.used_retrievers.append(path)

    def all_evidence(self) -> list[Evidence]:
        seen: set[tuple[str, str | None, str | None]] = set()
        out = []
        for items in self.lists.values():
            for e in items:
                key = (e.document_id, e.block_id, e.node_id)
                if key not in seen:
                    seen.add(key)
                    out.append(e)
        return out


__all__ = ["RetrievalAnchor", "RetrievalState"]
