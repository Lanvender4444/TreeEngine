"""Mutable state of one controlled retrieval: ranked anchors per path, budgets spent."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.models import Evidence
from .anchor import Anchor, RetrievalAnchor
from .result import Trace


@dataclass
class RetrievalState:
    query: str
    document_id: str | None = None
    lists: dict[str, list[Anchor]] = field(default_factory=dict)  # path -> ranked anchors
    visited_nodes: set[str] = field(default_factory=set)
    used_retrievers: list[str] = field(default_factory=list)
    step: int = 0
    llm_calls: int = 0
    tree_steps: int = 0
    fts_queries: int = 0
    vector_queries: int = 0
    stop_reason: str = ""
    trace: Trace = field(default_factory=Trace)

    def add(self, path: str, anchors: list[Anchor]) -> None:
        """Append a path's anchors (several steps of one path accumulate in one ranked list)."""
        have = {a.key for a in self.lists.get(path, [])}
        new = [a for a in anchors if a.key not in have]
        lst = self.lists.setdefault(path, [])
        for a in new:
            a.rank = len(lst) + 1
            lst.append(a)
        if path not in self.used_retrievers:
            self.used_retrievers.append(path)

    def all_anchors(self) -> list[Anchor]:
        return [a for items in self.lists.values() for a in items]

    @property
    def node_anchors(self) -> list[Anchor]:
        return [a for items in self.lists.values() for a in items if a.kind == "node"]

    def all_evidence(self) -> list[Evidence]:
        seen: set[tuple[str, str]] = set()
        out = []
        for items in self.lists.values():
            for a in items:
                if a.key not in seen and a.evidence is not None:
                    seen.add(a.key)
                    out.append(a.evidence)
        return out


__all__ = ["RetrievalAnchor", "RetrievalState"]
