"""TreeEngine as a post-processor behind another retriever (framework neutral).

Another RAG system retrieves its own chunks (its splitter, its vector DB); ``Reconstructor``
turns them into TreeEngine context: each chunk is aligned once with the blocks of the same
document (``integrations.align``), then ``build_context`` rebuilds continuous, citeable spans
(the section around the hit, neighbouring paragraphs, the whole table) inside a token budget.

    rec = Reconstructor(engine, token_budget=2000)
    spans = rec(chunks, query="...")        # ContextSpans
    records = rec.records(chunks, query)    # [{"text", "metadata"}] for any framework

The LangChain compressor, LlamaIndex node post-processor and Haystack component are thin
wrappers around it. Chunks of documents TreeEngine has not ingested pass through unchanged
(``passthrough``) unless ``strict=True``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .align import as_chunk, stored_alignment


class Reconstructor:
    def __init__(
        self,
        engine: Any,
        *,
        token_budget: int = 2000,
        policy: str = "auto",
        document_id: str | None = None,
        align: bool = True,
        min_quality: float = 0.0,
        strict: bool = False,
        on_unmapped: str | None = None,
    ) -> None:
        self.engine = engine
        self.token_budget = token_budget
        self.policy = policy
        self.document_id = document_id
        self.align = align
        self.min_quality = min_quality
        self.strict = strict
        self.on_unmapped = on_unmapped
        self.last_alignment: dict[str, Any] | None = None

    def hits(self, items: Iterable[Any]) -> list[dict[str, Any]]:
        chunks = [as_chunk(i) for i in items]
        if self.align:
            todo = [c for c in chunks if stored_alignment(self.engine.repo, c.key) is None]
            if todo:
                rep = self.engine.align_chunks(
                    todo, document_id=self.document_id, min_quality=self.min_quality
                )
                self.last_alignment = rep.summary()
        out = []
        for c in chunks:
            meta = {k: v for k, v in c.metadata.items() if k != "chunk_id"}
            out.append(
                {
                    "text": c.text,
                    "chunk_id": c.key,
                    "document_id": c.document_id,
                    "uri": c.uri,
                    "start_offset": c.start_offset,
                    "end_offset": c.end_offset,
                    "score": c.score,
                    "metadata": meta,
                }
            )
        return out

    def __call__(self, items: Iterable[Any], query: str | None = None) -> list[Any]:
        return self.engine.build_context(
            self.hits(items),
            token_budget=self.token_budget,
            policy=self.policy,
            document_id=self.document_id,
            question=query,
            strict_mapping=self.strict,
            on_unmapped=self.on_unmapped,
        )

    def records(self, items: Iterable[Any], query: str | None = None) -> list[dict[str, Any]]:
        from . import span_record

        return [span_record(s) for s in self(items, query)]


__all__ = ["Reconstructor"]
