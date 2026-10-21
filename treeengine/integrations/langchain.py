"""LangChain adapter: ``TreeEngineRetriever`` (needs ``langchain-core``; core TreeEngine does not).

    from treeengine.integrations.langchain import TreeEngineRetriever
    retriever = TreeEngineRetriever(engine=engine, document_id=doc.id, context_budget=2000)
    docs = retriever.invoke("Why did the operating margin fall?")

Each returned ``Document`` is one ContextSpan: ``page_content`` = the span text, ``metadata`` =
document / node / pages / offsets / block ids / anchors / source revision (a citation). The
query goes through ``engine.retrieve`` with ``preset`` ("lite" | "balanced" | "quality"; default
balanced) unless an explicit ``policy`` is given. Tested with langchain-core 0.3 and 1.x.

``TreeEngineCompressor`` goes the other way: behind *your* retriever (any vector store), as a
document compressor - your chunks in, TreeEngine's reconstructed spans out:

    from langchain.retrievers import ContextualCompressionRetriever
    retriever = ContextualCompressionRetriever(
        base_compressor=TreeEngineCompressor(engine=engine, token_budget=2000),
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
    )

Chunks are aligned with TreeEngine's blocks on first sight (``integrations.align``) - the same
files must have been ingested into the engine.
"""

from __future__ import annotations

from typing import Any

from . import to_documents

try:  # pragma: no cover - import guard
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
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


class TreeEngineCompressor(BaseDocumentCompressor):
    engine: Any
    token_budget: int = 2000
    document_id: str | None = None
    policy: str = "auto"
    strict: bool = False
    align: bool = True
    min_quality: float = 0.0

    def compress_documents(
        self, documents: Any, query: str, callbacks: Any = None
    ) -> list[Document]:
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
        return [
            Document(page_content=r["text"], metadata=r["metadata"])
            for r in rec.records(list(documents), query)
        ]


__all__ = ["TreeEngineCompressor", "TreeEngineRetriever"]
