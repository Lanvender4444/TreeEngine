"""LlamaIndex adapter: ``TreeEngineLlamaRetriever`` (needs ``llama-index-core``).

Each ContextSpan becomes a ``NodeWithScore(TextNode(text, metadata))``; the score is the span's
rank position (1.0 for the first, decreasing), since TreeEngine fuses by rank. ``metadata`` is
the same flat citation record as the LangChain adapter's. Tested with llama-index-core 0.12+.

``TreeEngineNodePostprocessor`` sits behind any LlamaIndex retriever / query engine
(``node_postprocessors=[TreeEngineNodePostprocessor(engine)]``): the retrieved nodes are
aligned with TreeEngine's blocks and replaced by reconstructed context spans.
"""

from __future__ import annotations

from typing import Any

from . import to_documents

try:  # pragma: no cover - import guard
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
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


class TreeEngineNodePostprocessor(BaseNodePostprocessor):
    engine: Any = None
    token_budget: int = 2000
    document_id: str | None = None
    policy: str = "auto"
    strict: bool = False
    align: bool = True
    min_quality: float = 0.0

    def __init__(self, engine: Any = None, **kw: Any) -> None:
        super().__init__(engine=engine, **kw)

    @classmethod
    def class_name(cls) -> str:
        return "TreeEngineNodePostprocessor"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        from .postprocess import Reconstructor

        rec = Reconstructor(
            self.engine,
            token_budget=self.token_budget,
            policy=self.policy,
            document_id=self.document_id,
            align=self.align,
            min_quality=self.min_quality,
            strict=self.strict,
        )
        q = query_bundle.query_str if query_bundle is not None else None
        return [
            NodeWithScore(
                node=TextNode(text=r["text"], metadata=r["metadata"]), score=1.0 / (i + 1)
            )
            for i, r in enumerate(rec.records(nodes, q))
        ]


__all__ = ["TreeEngineLlamaRetriever", "TreeEngineNodePostprocessor"]
