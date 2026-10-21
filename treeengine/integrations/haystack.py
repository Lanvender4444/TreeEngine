"""Haystack 2.x component: ``TreeEngineReconstructor`` (needs ``haystack-ai``).

Put it after any retriever in a pipeline; it takes the retrieved ``documents`` (and the query)
and returns TreeEngine's reconstructed context spans as Haystack ``Document`` objects:

    pipe.add_component("retriever", InMemoryEmbeddingRetriever(store))
    pipe.add_component("treeengine", TreeEngineReconstructor(engine, token_budget=2000))
    pipe.connect("retriever.documents", "treeengine.documents")

``TreeEngineRetriever`` is the plain retriever: a query in, TreeEngine's own retrieval out.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import guard
    from haystack import Document, component
except ImportError as e:  # pragma: no cover
    from ..api.errors import DependencyMissingError

    raise DependencyMissingError(
        "the Haystack adapter needs haystack-ai: pip install 'treeengine[haystack]'"
    ) from e


def _docs(records: list[dict[str, Any]]) -> list[Document]:
    out = []
    for i, r in enumerate(records):
        meta = {k: v for k, v in r["metadata"].items() if v is not None}
        out.append(Document(content=r["text"], meta=meta, score=1.0 / (i + 1)))
    return out


@component
class TreeEngineReconstructor:
    def __init__(
        self,
        engine: Any,
        token_budget: int = 2000,
        document_id: str | None = None,
        policy: str = "auto",
        strict: bool = False,
        align: bool = True,
        min_quality: float = 0.0,
    ) -> None:
        from .postprocess import Reconstructor

        self.reconstructor = Reconstructor(
            engine,
            token_budget=token_budget,
            policy=policy,
            document_id=document_id,
            align=align,
            min_quality=min_quality,
            strict=strict,
        )

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], query: str | None = None) -> dict[str, Any]:
        return {"documents": _docs(self.reconstructor.records(documents, query))}


@component
class TreeEngineRetriever:
    def __init__(
        self,
        engine: Any,
        document_id: str | None = None,
        context_budget: int = 2000,
        preset: str = "balanced",
    ) -> None:
        self.engine = engine
        self.document_id = document_id
        self.context_budget = context_budget
        self.preset = preset

    @component.output_types(documents=list[Document])
    def run(self, query: str) -> dict[str, Any]:
        from . import to_documents

        res = self.engine.retrieve(
            query,
            document_id=self.document_id,
            preset=self.preset,
            context_budget=self.context_budget,
        )
        return {"documents": _docs(to_documents(res))}


__all__ = ["TreeEngineReconstructor", "TreeEngineRetriever"]
