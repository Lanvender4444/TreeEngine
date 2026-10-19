"""LangChain adapter: ``TreeEngineRetriever`` (needs ``langchain-core``; core TreeEngine does not).

    from treeengine.integrations.langchain import TreeEngineRetriever
    retriever = TreeEngineRetriever(engine=engine, document_id=doc.id, context_budget=2000)
    docs = retriever.invoke("Why did the operating margin fall?")

Each returned ``Document`` is one ContextSpan: ``page_content`` = the span text, ``metadata`` =
document / node / pages / offsets / block ids / anchors / source revision (a citation). The
query goes through ``engine.retrieve`` with ``preset`` ("lite" | "balanced" | "quality"; default
balanced) unless an explicit ``policy`` is given. Tested with langchain-core 0.3 and 1.x.
"""

from __future__ import annotations

from typing import Any

from . import to_documents

try:  # pragma: no cover - import guard
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:  # pragma: no cover
    from ..api.errors import DependencyMissingError

    raise DependencyMissingError(
        "the LangChain adapter needs langchain-core: pip install 'treeengine[langchain]'"
    ) from e


class TreeEngineRetriever(BaseRetriever):
    engine: Any
    document_id: str | None = None
    context_budget: int = 2000
    preset: str | None = None  # lite | balanced | quality (default balanced)
    policy: Any = None  # a RetrievalPolicy overrides the preset

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.policy is not None:
            from dataclasses import replace

            pol = replace(self.policy, context_budget=self.context_budget)
            res = self.engine.retrieve(query, document_id=self.document_id, policy=pol)
        else:
            res = self.engine.retrieve(
                query,
                document_id=self.document_id,
                preset=self.preset or "balanced",
                context_budget=self.context_budget,
            )
        return [Document(page_content=r["text"], metadata=r["metadata"]) for r in to_documents(res)]


__all__ = ["TreeEngineRetriever"]
