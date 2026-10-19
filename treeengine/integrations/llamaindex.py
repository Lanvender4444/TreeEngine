"""LlamaIndex adapter: ``TreeEngineLlamaRetriever`` (needs ``llama-index-core``).

Each ContextSpan becomes a ``NodeWithScore(TextNode(text, metadata))``; the score is the span's
rank position (1.0 for the first, decreasing), since TreeEngine fuses by rank. ``metadata`` is
the same flat citation record as the LangChain adapter's. Tested with llama-index-core 0.12+.
"""

from __future__ import annotations

from typing import Any

from . import to_documents

try:  # pragma: no cover - import guard
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as e:  # pragma: no cover
    from ..api.errors import DependencyMissingError

    raise DependencyMissingError(
        "the LlamaIndex adapter needs llama-index-core: pip install 'treeengine[llamaindex]'"
    ) from e


class TreeEngineLlamaRetriever(BaseRetriever):
    def __init__(
        self,
        engine: Any,
        document_id: str | None = None,
        context_budget: int = 2000,
        policy: Any = None,
        preset: str | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.document_id = document_id
        self.context_budget = context_budget
        self.policy = policy
        self.preset = preset

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        q = query_bundle.query_str
        if self.policy is not None:
            from dataclasses import replace

            pol = replace(self.policy, context_budget=self.context_budget)
            res = self.engine.retrieve(q, document_id=self.document_id, policy=pol)
        else:
            res = self.engine.retrieve(
                q,
                document_id=self.document_id,
                preset=self.preset or "balanced",
                context_budget=self.context_budget,
            )
        recs = to_documents(res)
        return [
            NodeWithScore(
                node=TextNode(text=r["text"], metadata=r["metadata"]), score=1.0 / (i + 1)
            )
            for i, r in enumerate(recs)
        ]


__all__ = ["TreeEngineLlamaRetriever"]
