"""Baseline A: the whole document goes to the answer model (no retrieval).

Answers two questions: is retrieval needed at all, and how much context / cost does it save?
Only meaningful for single-document questions whose document fits the answer model's context;
anything over ``max_tokens`` is skipped (and reported as skipped, not as wrong).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from treeengine.core.models import Evidence

from ..metrics.cost import count_tokens
from .base import IndexStats, RetrievalRun


class FullContext:
    name = "full_context"
    family = "baseline"
    qa_only = True
    requires = ("corpus",)

    def __init__(self, ws: Any, max_tokens: int = 120_000) -> None:
        self.ws = ws
        self.max_tokens = max_tokens
        self._tokens: dict[str, int] = {}

    def index(self, documents: Sequence[str] | None = None) -> IndexStats:
        return IndexStats(self.name, documents=len(self.ws.doc_ids), parts=["parse only"])

    def doc_tokens(self, document_id: str) -> int:
        if document_id not in self._tokens:
            doc = self.ws.repo.get_document(document_id)
            self._tokens[document_id] = count_tokens(doc.text or "") if doc else 0
        return self._tokens[document_id]

    def retrieve(
        self, query: str, *, document_id: str | None = None, limit: int = 5
    ) -> RetrievalRun:
        t0 = time.perf_counter()
        if document_id is None:
            return RetrievalRun([], metadata={"skipped": "corpus-wide question"})
        n = self.doc_tokens(document_id)
        if n > self.max_tokens:
            return RetrievalRun([], metadata={"skipped": f"{n} tokens > {self.max_tokens}"})
        doc = self.ws.repo.get_document(document_id)
        ev = Evidence(document_id, None, f"{document_id}:full", doc.text, "full_context", 1.0, {})
        return RetrievalRun([ev], latency_ms=(time.perf_counter() - t0) * 1000, context_tokens=n)


__all__ = ["FullContext"]
