"""LangChain adapter: ``TreeEngineRetriever`` (needs ``langchain-core``; core TreeEngine does not).

    from treeengine.integrations.langchain import TreeEngineRetriever
    retriever = TreeEngineRetriever(engine=engine, document_id=doc.id, context_budget=2000)
    docs = retriever.invoke("Why did the operating margin fall?")

Each returned ``Document`` is one ContextSpan: ``page_content`` = the span text, ``metadata`` =
document / node / pages / block ids / anchors. The query goes through ``engine.retrieve`` with
the benchmark-validated default policy (FTS + Vector, auto context) unless ``policy`` is given.
"""

from __future__ import annotations

from typing import Any

from . import to_documents

try:  # pragma: no cover - import guard
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the LangChain adapter needs langchain-core: pip install 'treeengine[langchain]'"
    ) from e


class TreeEngineRetriever(BaseRetriever):
    engine: Any
    document_id: str | None = None
    context_budget: int = 2000
    policy: Any = None  # RetrievalPolicy; default: RetrievalPolicy() (FTS + Vector, auto)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        from ..retrieval.policy import RetrievalPolicy

        pol = self.policy or RetrievalPolicy(context_budget=self.context_budget)
        res = self.engine.retrieve(query, document_id=self.document_id, policy=pol)
        return [Document(page_content=r["text"], metadata=r["metadata"]) for r in to_documents(res)]


__all__ = ["TreeEngineRetriever"]
