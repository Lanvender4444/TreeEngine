"""LlamaIndex adapter: ``TreeEngineLlamaRetriever`` (needs ``llama-index-core``).

Each ContextSpan becomes a ``NodeWithScore(TextNode(text, metadata))``; the score is the span's
rank position (1.0 for the first, decreasing), since TreeEngine fuses by rank.
"""

from __future__ import annotations

from typing import Any

from . import to_documents

try:  # pragma: no cover - import guard
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the LlamaIndex adapter needs llama-index-core: pip install llama-index-core"
    ) from e


class TreeEngineLlamaRetriever(BaseRetriever):
    def __init__(
        self,
        engine: Any,
        document_id: str | None = None,
        context_budget: int = 2000,
        policy: Any = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.document_id = document_id
        self.context_budget = context_budget
        self.policy = policy

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        from ..retrieval.policy import RetrievalPolicy

        pol = self.policy or RetrievalPolicy(context_budget=self.context_budget)
        res = self.engine.retrieve(query_bundle.query_str, document_id=self.document_id, policy=pol)
        recs = to_documents(res)
        return [
            NodeWithScore(
                node=TextNode(text=r["text"], metadata=r["metadata"]), score=1.0 / (i + 1)
            )
            for i, r in enumerate(recs)
        ]


__all__ = ["TreeEngineLlamaRetriever"]
